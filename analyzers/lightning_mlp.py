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
from lightning.pytorch.callbacks import DeviceStatsMonitor, StochasticWeightAveraging
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.utilities import CombinedLoader

from lightning.pytorch.tuner import Tuner


# define the LightningModule
class LitMlp(L.LightningModule):
    def __init__(
        self,
        labels: list,
        lr=1e-4,
        input_dim: int = None,
        hidden_size: int = 200,
        output_dim: int = 1,
        datasets=None,
    ):
        super().__init__()
        torch.set_float32_matmul_precision("medium")
        self.lr = lr

        self.datasets = datasets

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.labels: list[str] = labels

        self.hidden = nn.Sequential(
            (
                nn.Linear(input_dim, hidden_size)
                if input_dim is not None
                else nn.LazyLinear(hidden_size)
            ),
            nn.ReLU(),
            # nn.Linear(l1, l2),
            # nn.ReLU(),
        )

        self.heads = nn.ModuleList([nn.Linear(hidden_size, output_dim) for _ in labels])

    def train_dataloader(self):

        train_loaders = {}
        for label in self.labels:
            dataset = self.datasets[label]["train"]
            # use 20% of training data for validation
            train_set_size = int(len(dataset) * 0.8)
            seed = torch.Generator().manual_seed(42)
            # train_set, val_set = torch.utils.data.random_split(
            #     dataset, [train_set_size, valid_set_size], generator=seed
            # )

            train_loaders[label] = data_utils.DataLoader(
                dataset, batch_size=10, shuffle=True, num_workers=19
            )

        combined_train_loader = CombinedLoader(train_loaders, mode="max_size_cycle")

        return combined_train_loader

    def predict_step(self, batch, batch_idx):
        for label, (x, y) in batch.items():
            print(f"{label}: {x.shape}, {y.shape}")
        # return pred

    def training_step(self, batch, batch_idx):

        losses = []
        for label, (x, y) in batch.items():
            # print(label, x, y)
            x = x.view(x.size(0), -1)
            x = self.hidden(x)

            task_idx = self.labels.index(label)
            y_pred = self.heads[task_idx](x)

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
            x = self.hidden(x)

            task_idx = self.labels.index(label)
            y_pred = self.heads[task_idx](x)

            test_loss = nn.functional.mse_loss(y_pred, y)
            self.log(f"test_loss_{label}", test_loss)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # this is the validation loop
        # print(batch)

        val_losses = {}
        val_losses_list = []

        for label, (x, y) in batch.items():
            # print(label, x, y)
            x = x.view(x.size(0), -1)
            x = self.hidden(x)

            task_idx = self.labels.index(label)
            y_pred = self.heads[task_idx](x)

            val_loss = nn.functional.mse_loss(y_pred, y)
            val_losses[label] = val_loss
            val_losses_list.append(val_loss.reshape(1))
            # self.log()

        tensorboard = self.logger.experiment
        tensorboard.add_scalars(f"val_loss", val_losses, self.current_epoch)

        mean_val_loss = torch.mean(torch.cat(val_losses_list))
        self.log(f"val_loss", mean_val_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x, label) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.hidden(x)

        task_idx = self.labels.index(label)
        y_pred = self.heads[task_idx](x)

        return y_pred


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
        batch_size: int = 100,
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
        self.batch_size = batch_size

        self.lit_model = LitMlp(
            labels,
            input_dim=1051,
            hidden_size=hidden_size,
            lr=lr,
            datasets=self.datasets,
        )

        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.lit_model = LitMlp.load_from_checkpoint(
                checkpoint_path=checkpoint_path, labels=self.labels
            )

        self.trainer = L.Trainer(
            accelerator="gpu",
            # limit_val_batches=100,
            max_epochs=max_train_epochs,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=3),
                DeviceStatsMonitor(),
                # StochasticWeightAveraging(swa_lrs=1e-2),
            ],
            # accumulate_grad_batches=7,
            # gradient_clip_val=0.5,
            log_every_n_steps=20,
            # profiler="simple",
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
                train_set, batch_size=self.batch_size, shuffle=True, num_workers=19
            )
            val_loaders[label] = data_utils.DataLoader(
                val_set, batch_size=self.batch_size, num_workers=19
            )

        combined_train_loader = CombinedLoader(train_loaders, mode="max_size_cycle")
        combined_val_loader = CombinedLoader(val_loaders, mode="max_size_cycle")
        iter(combined_train_loader)
        iter(combined_val_loader)
        print("TRAIN SET")
        print(len(combined_train_loader))
        print("VAL SET")
        print(len(combined_val_loader))

        self.trainer.fit(self.lit_model, combined_train_loader, combined_val_loader)

    def hypertune(self):
        # Create a Tuner
        tuner = Tuner(self.trainer)

        # finds learning rate automatically
        # sets hparams.lr or hparams.learning_rate to that learning rate
        lr_finder = tuner.lr_find(self.lit_model, early_stop_threshold=None)

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()
        fig.savefig("lrs.png")

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        print(f"May we suggest: {new_lr}")

        # # update hparams of the model
        # self.lit_model.hparams.lr = new_lr

        # # Fit model
        # self.trainer.fit(self.lit_model)

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
                test_set, batch_size=10, num_workers=19
            )

        combined_test_loader = CombinedLoader(test_loaders)
        iter(combined_test_loader)
        print("TEST SET")
        print(len(combined_test_loader))

        self.trainer.test(self.lit_model, combined_test_loader)

    def predict(self, X: pd.DataFrame, label: str):
        return self.lit_model.forward(X, label)
