from analyzers.analyzer import Analyzer
import numpy as np
import pandas as pd
from analyzers import utils
from analyzers.utils import CustomDataset
from scipy.stats import zscore


# from sklearn.cross_decomposition import PLSRegression
# import tensorflow as tf
# from tensorflow import keras
# from keras import callbacks, layers, Model, saving, regularizers
from tqdm import trange, tqdm

from analyzers.utils import rsquared
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

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
class LitPlainMlp(L.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        input_dim: int = None,
        hidden_size: int = 200,
        output_dim: int = 1,
        datasets=None,
        n_augmentations=0,
        p=0.5,
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
        self.dropout = nn.Dropout(p=p)

        self.head = nn.Linear(hidden_size, output_dim)

        self.n_augmentations = n_augmentations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        y_pred = self.head(x)
        return y_pred

    def training_step(self, batch, batch_idx):

        x, y = batch

        losses = []

        if self.training:
            for _ in range(self.n_augmentations):
                # Augment the spectrum

                # Augmentation step 1: Vertical scaling
                scaling_magnitude = 0.2  # TODO: Parameterize this
                scale = torch.rand(1, device="cuda") * 2 - 1 * scaling_magnitude

                noise = torch.randn_like(x) * 1e-3  # TODO: Parameterize this

                x: torch.Tensor
                x_ = x.clone()

                og_max = torch.max(x_)
                x_ += x_ * scale
                x_ += noise
                y_pred = self.forward(x_)

                loss = nn.functional.mse_loss(y_pred, y)

                print(f"The {_+1}th loss is {loss.item()}")
                print(
                    f"The {_+1}th spectra's max went from {og_max} -> {torch.max(x_)}"
                )
                losses.append(loss)

        y_pred = self.forward(x)

        losses.append(nn.functional.mse_loss(y_pred, y))

        loss = torch.mean(torch.stack(losses))
        print(f"The mean loss is {loss.item()}")

        self.log(f"train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):

        x, y = batch

        y_pred = self.forward(x)

        loss = nn.functional.mse_loss(y_pred, y)

        self.log(f"test_loss", loss)

        y_np: np.ndarray = y.numpy(force=True).reshape((-1, 1))
        y_pred_np: np.ndarray = y_pred.numpy(force=True).reshape((-1, 1))

        # Filter to only include non-nan values
        y_pred_np = y_pred_np[np.isnan(y_np) == False]
        y_np = y_np[np.isnan(y_np) == False]

        errors = np.abs(y_pred_np - y_np)
        Y_true_no_outliers = y_np[(np.abs(zscore(errors)) < 3)]
        Y_pred_no_outliers = y_pred_np[(np.abs(zscore(errors)) < 3)]

        if np.isnan(y_np).any():
            print(
                f"Labels contain {np.isnan(y_np).sum()} null values ({np.isnan(y_np).sum()/y_np.size*100:.2f}%)!"
            )
            print(
                f"Labels contain {y_np.size-np.isnan(y_np).sum()} null values ({(y_np.size-np.isnan(y_np).sum())/y_np.size*100:.2f}%)!"
            )
            r2 = -1.0
        elif np.isnan(y_pred_np).any():
            print(f"Predictions contain {np.isnan(y_pred_np).sum()} null values!")
            r2 = -1.0
        else:
            r2 = rsquared(y_np, y_pred_np)

        ax = plt.subplot()
        ax.scatter(Y_true_no_outliers, Y_pred_no_outliers)
        ax.plot([-1, 1], [-1, 1])

        plt.savefig("test_plot.png")
        plt.show()

        self.log(f"test_r2", r2)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_pred = self.forward(x)

        loss = nn.functional.mse_loss(y_pred, y).reshape(1)

        self.log(f"val_loss", loss)

        y_np: np.ndarray = y.numpy(force=True).reshape((-1, 1))
        y_pred_np: np.ndarray = y_pred.numpy(force=True).reshape((-1, 1))

        # Filter to only include non-nan values
        y_pred_np = y_pred_np[np.isnan(y_np) == False]
        y_np = y_np[np.isnan(y_np) == False]

        if np.isnan(y_np).any():
            print(
                f"Labels contain {np.isnan(y_np).sum()} null values ({np.isnan(y_np).sum()/y_np.size*100:.2f}%)!"
            )
            print(
                f"Labels contain {y_np.size-np.isnan(y_np).sum()} null values ({(y_np.size-np.isnan(y_np).sum())/y_np.size*100:.2f}%)!"
            )
            r2 = -1.0
        elif np.isnan(y_pred_np).any():
            print(f"Predictions contain {np.isnan(y_pred_np).sum()} null values!")
            r2 = -1.0
        else:
            r2 = rsquared(y_np, y_pred_np)

        # test_pd = pd.DataFrame(
        #     np.hstack((y_np, y_pred_np)), columns=["y_true", "y_pred"]
        # )
        # test_pd.to_csv(f"test_results_{batch_idx}.csv")

        ax = plt.subplot()
        ax.scatter(y_np, y_pred_np)
        # ax.xlabel("True")
        # ax.xlabel("Predicted")
        ax.plot([-1, 1], [-1, 1])

        tensorboard = self.logger.experiment
        tensorboard.add_figure(
            "val/real_vs_pred", plt.gcf(), global_step=self.current_epoch
        )

        # log the outputs!
        self.log("r2/val", r2, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MlpAnalyzer(Analyzer):
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
        n_augmentations: int = 10,
    ) -> None:
        super().__init__(verbose=verbose)

        self.lr = lr
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.lit_model = LitPlainMlp(
            input_dim=input_size,
            hidden_size=hidden_size,
            lr=lr,
            n_augmentations=n_augmentations,
            output_dim=output_size,
        )

        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.lit_model = LitPlainMlp.load_from_checkpoint(
                checkpoint_path=checkpoint_path, output_dim=output_size
            )

        self.trainer = L.Trainer(
            accelerator="gpu",
            # limit_val_batches=100,
            max_epochs=max_train_epochs,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=10),
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
            dataset, batch_size=len(X_test), num_workers=19
        )

        self.trainer.test(self.lit_model, test_dataloader)

    def predict(self, X: pd.DataFrame):
        return self.lit_model.forward(X)
