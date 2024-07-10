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
class LitPiMlp(L.LightningModule):
    def __init__(
        self,
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

        # Model layers
        self.l1 = (
            nn.Linear(input_dim, hidden_size)
            if input_dim is not None
            else nn.LazyLinear(hidden_size)
        )
        self.relu1 = nn.ReLU()

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()

        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        y_pred = self.head(x)
        return y_pred

    def augment(
        self, x, vertical_scale: float = 0.2, noise_scale: float = 0.01, seed: int = 64
    ):
        rng = torch.Generator(device="cuda")
        rng.manual_seed(seed)
        x_ = x + (x * torch.rand((1), generator=rng, device="cuda") * vertical_scale)

        # print(f"{x} -> {x_}")
        return x_

    def training_step(self, batch, batch_idx):

        x, y = batch

        # unsupervised weight ramp-up function
        T = self.current_epoch
        w = max(0.1 * T, 1.0)

        # print(x.numpy(force=True))

        y_pred = self.forward(x)
        y1 = self.forward(self.augment(x, seed=64))
        y2 = self.forward(self.augment(x, seed=65))

        # print(f"{y1}, {y2}")

        unsupervised_loss = nn.functional.mse_loss(y1, y2)

        # unsupervised loss

        # Detect if labeled
        if torch.isnan(y) == torch.tensor(True):
            # print("y is nan!")
            supervised_loss = 0
        else:
            # print(y)
            # print(type(y))
            supervised_loss = nn.functional.mse_loss(y_pred, y)

        loss = supervised_loss + w * unsupervised_loss
        # print(f"{supervised_loss:.2f} + {unsupervised_loss:.2f} = {loss:.2f}")

        self.log(f"train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        # unsupervised weight ramp-up function
        T = self.current_epoch
        w = max(0.1 * T, 1.0)

        # print(x.numpy(force=True))

        y_pred = self.forward(x)
        y1 = self.forward(self.augment(x, seed=64))
        y2 = self.forward(self.augment(x, seed=65))

        # print(f"{y1}, {y2}")

        unsupervised_loss = nn.functional.mse_loss(y1, y2)

        # unsupervised loss

        # Detect if labeled
        if torch.isnan(y) == torch.tensor(True):
            # print("y is nan!")
            supervised_loss = 0
        else:
            # print(y)
            # print(type(y))
            supervised_loss = nn.functional.mse_loss(y_pred, y)

        loss = supervised_loss + w * unsupervised_loss
        # print(f"{supervised_loss:.2f} + {unsupervised_loss:.2f} = {loss:.2f}")

        self.log(f"test_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        x, y = batch

        # unsupervised weight ramp-up function
        T = self.current_epoch
        w = max(0.1 * T, 1.0)

        # print(x.numpy(force=True))

        y_pred = self.forward(x)
        y1 = self.forward(self.augment(x, seed=64))
        y2 = self.forward(self.augment(x, seed=65))

        # print(f"{y1}, {y2}")

        unsupervised_loss = nn.functional.mse_loss(y1, y2)

        # unsupervised loss

        # Detect if labeled
        if torch.isnan(y) == torch.tensor(True):
            # print("y is nan!")
            supervised_loss = 0
        else:
            # print(y)
            # print(type(y))
            supervised_loss = nn.functional.mse_loss(y_pred, y)

        loss = supervised_loss + w * unsupervised_loss
        # print(f"{supervised_loss:.2f} + {unsupervised_loss:.2f} = {loss:.2f}")

        self.log(f"val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class PiMlpAnalyzer(Analyzer):
    def __init__(
        self,
        output_size,
        verbose: int = 0,
        lr=1e-3,
        hidden_size=200,
        batch_size: int = 100,
        input_size=None,
        checkpoint_path=None,
        max_train_epochs: int = 1000,
    ) -> None:
        super().__init__(verbose=verbose)

        self.lr = lr
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.lit_model = LitPiMlp(
            input_dim=input_size,
            hidden_size=hidden_size,
            lr=lr,
            output_dim=output_size,
        )

        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.lit_model = LitPiMlp.load_from_checkpoint(
                checkpoint_path=checkpoint_path, output_dim=output_size
            )

        self.trainer = L.Trainer(
            accelerator="gpu",
            # limit_val_batches=100,
            max_epochs=max_train_epochs,
            callbacks=[
                # EarlyStopping(monitor="val_loss", mode="min", patience=10),
                DeviceStatsMonitor(),
                # StochasticWeightAveraging(swa_lrs=1e-2),
            ],
            # accumulate_grad_batches=7,
            # gradient_clip_val=0.5,
            log_every_n_steps=20,
            # profiler="simple",
        )

    def train(self, X_train, Y_train):

        dataset = CustomDataset(X_train, Y_train)
        # use 20% of training data for validation
        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = len(dataset) - train_set_size
        seed = torch.Generator().manual_seed(64)
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_set_size, valid_set_size], generator=seed
        )

        train_loader = data_utils.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=19
        )
        val_loader = data_utils.DataLoader(
            val_set, batch_size=self.batch_size, num_workers=19
        )

        self.trainer.fit(self.lit_model, train_loader, val_loader)

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

    def test(self, X_test, Y_test):
        dataset = CustomDataset(X_test, Y_test)

        test_dataloader = data_utils.DataLoader(
            dataset, batch_size=self.batch_size, num_workers=19
        )

        self.trainer.test(self.lit_model, test_dataloader)

    def predict(self, X: pd.DataFrame):
        return self.lit_model.forward(X)
