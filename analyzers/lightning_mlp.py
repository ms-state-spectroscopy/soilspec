from analyzers.analyzer import Analyzer
import numpy as np
import pandas as pd
from analyzers import utils
from analyzers.utils import CustomDataset

# from sklearn.cross_decomposition import PLSRegression
# import tensorflow as tf
# from tensorflow import keras
# from keras import callbacks, layers, Model, saving, regularizers
from tqdm import trange, tqdm

import os
from torch import optim, nn, utils, Tensor
import torch
import torch.utils.data as data_utils

# from torchvision.datasets import MNIST
# from torchvision.transforms import ToTensor
import lightning as L
from lightning.pytorch.utilities import CombinedLoader


class MultiTaskNetwork(nn.Module):
    def __init__(
        self,
        labels: list,
        input_dim: int = None,
        hidden_dim: int = 200,
        output_dim: int = 1,
    ):
        super(MultiTaskNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.labels: list[str] = labels
        self.task_idx = 0

        self.hidden = nn.Sequential(
            (
                nn.Linear(input_dim, hidden_dim)
                if input_dim is not None
                else nn.LazyLinear(hidden_dim)
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in labels])

    def getTaskLabel(self):
        return self.labels[self.task_idx]

    def setTask(self, label: str):
        self.task_idx = self.labels.index(label)

    def switchToNextTask(self):
        self.task_idx += 1
        self.task_idx %= len(self.labels)
        print(f"Switching to {self.labels[self.task_idx]}")

    def forward(self, x: torch.Tensor, task_label: str):

        self.setTask(task_label)

        x = self.hidden(x)

        # print(f"Switching to task {self.getTaskLabel()}")
        x = self.heads[self.task_idx](x)

        return x


# define the LightningModule
class LitMlp(L.LightningModule):
    def __init__(self, model: MultiTaskNetwork):
        super().__init__()
        torch.set_float32_matmul_precision("medium")
        self.model = model

    def training_step(self, batch, batch_idx):

        losses = []
        for label, (x, y) in batch.items():
            # print(label, x, y)
            x = x.view(x.size(0), -1)
            y_pred = self.model(x, label)
            losses.append(nn.functional.mse_loss(y_pred, y).reshape(1))

        # print(losses)
        loss = torch.sum(torch.cat(losses))
        # Logging to TensorBoard (if installed) by default
        self.log(f"train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):

        for label, (x, y) in batch.items():
            # print(label, x, y)
            x = x.view(x.size(0), -1)
            y_pred = self.model(x, label)
            test_loss = nn.functional.mse_loss(y_pred, y)
            self.log(f"test_loss_{self.model.getTaskLabel()}", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        # print(batch)

        val_losses = {}

        for label, (x, y) in batch.items():
            # print(label, x, y)
            x = x.view(x.size(0), -1)
            y_pred = self.model(x, label)
            val_loss = nn.functional.mse_loss(y_pred, y)
            val_losses[f"val_loss_{self.model.getTaskLabel()}"] = val_loss
            # self.log()

        tensorboard = self.logger.experiment
        tensorboard.add_scalars(f"val_loss", val_losses, self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    # def on_train_epoch_end(self):
    #     self.model.switchToNextTask()


class LightningMlpAnalyzer(Analyzer):
    def __init__(
        self,
        datasets,
        labels,
        verbose: int = 0,
        lr=1e-3,
        hidden_size=200,
        activation="relu",
        n_logits=3,
        input_size=None,
        checkpoint_path=None,
        max_train_epochs: int = 1000,
    ) -> None:
        super().__init__(verbose=verbose)

        self.datasets = datasets
        self.labels = labels

        self.activation = activation
        self.lr = lr
        self.hidden_size = hidden_size

        self.model = MultiTaskNetwork(labels, input_size, hidden_size)

        self.lit_model = LitMlp(self.model)

        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.lit_model = LitMlp.load_from_checkpoint(
                checkpoint_path=checkpoint_path, model=self.model
            )

        self.trainer = L.Trainer(
            accelerator="gpu", limit_val_batches=100, max_epochs=max_train_epochs
        )

    def train(self):
        train_loaders = {}
        val_loaders = {}
        for label in self.labels:
            dataset = self.datasets[label]["train"]
            # use 20% of training data for validation
            train_set_size = int(len(dataset) * 0.8)
            valid_set_size = len(dataset) - train_set_size
            seed = torch.Generator().manual_seed(42)
            train_set, val_set = torch.utils.data.random_split(
                dataset, [train_set_size, valid_set_size], generator=seed
            )

            train_loaders[label] = data_utils.DataLoader(
                train_set, batch_size=10, shuffle=True, num_workers=19
            )
            val_loaders[label] = data_utils.DataLoader(
                val_set, batch_size=10, num_workers=19
            )

        combined_train_loader = CombinedLoader(train_loaders)
        combined_val_loader = CombinedLoader(val_loaders)
        iter(combined_train_loader)
        iter(combined_val_loader)
        print("TRAIN SET")
        print(len(combined_train_loader))
        print("VAL SET")
        print(len(combined_val_loader))

        self.trainer.fit(self.lit_model, combined_train_loader, combined_val_loader)

    def test(self):
        test_loaders = {}
        for label in self.labels:
            dataset = self.datasets[label]["test"]
            # use 20% of testing data for validation
            test_set_size = int(len(dataset) * 0.8)
            valid_set_size = len(dataset) - test_set_size
            seed = torch.Generator().manual_seed(42)
            test_set, val_set = torch.utils.data.random_split(
                dataset, [test_set_size, valid_set_size], generator=seed
            )

            test_loaders[label] = data_utils.DataLoader(
                test_set, batch_size=10, shuffle=True, num_workers=19
            )

        combined_test_loader = CombinedLoader(test_loaders)
        iter(combined_test_loader)
        print("TEST SET")
        print(len(combined_test_loader))

        self.trainer.test(self.lit_model, combined_test_loader)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
