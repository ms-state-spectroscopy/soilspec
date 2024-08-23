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

# For .png -> .gif
from PIL import Image
import glob
import contextlib


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
        p=0.2,
    ):
        super().__init__()
        torch.set_float32_matmul_precision("medium")
        self.lr = lr

        self.datasets = datasets

        print(f"Input dim {input_dim}")
        print(f"Hidden dim {hidden_size}")

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p),
        )

        # self.backbone.requires_grad_(False)

        # Model layers

        self.head = nn.Linear(hidden_size, output_dim)

        self.n_augmentations = n_augmentations
        self.current_train_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x = x.view(x.size(0), -1)
        y_pred = self.head(self.backbone(x))
        # y_pred = self.head(x)
        return y_pred

    def training_step(self, batch, batch_idx):

        x, y = batch

        losses = []

        if self.training and self.n_augmentations > 0:
            preds = []
            for _ in range(self.n_augmentations):

                # should be in range (1-scaling_factor, 1+scaling factor)
                # starts in range (0,1)
                # Moves to range (-1,1)
                scaling_factor = (torch.rand(1).cuda() * 2 - 1) * 0.5 + 1
                # plt.plot(scale_line, label="Scale line")

                noise = torch.randn_like(x) * 1e-3

                print(f"Scaling factor is {scaling_factor}")

                augmented_X = x * scaling_factor
                # augmented_X = x * scaling_factor
                preds.append(self.head(self.backbone(augmented_X)))

            preds = torch.stack(preds)
            y_pred = preds.mean(dim=0)
            # print(preds.shape)
            # print(y_pred.shape)

        else:
            y_pred = self.forward(x)

        # print(f"{y.shape} vs {y_pred.shape}")

        loss = nn.functional.mse_loss(y_pred, y)

        self.log(f"train_loss", loss, prog_bar=True)
        self.current_train_loss = loss
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
        eps = 1.1  # So that min/max values are not on the edge
        ax.scatter(y_np, y_pred_np)
        ax.plot(
            [y_np.min() * eps, y_np.max() * eps],
            [y_np.min() * eps, y_np.max() * eps],
            c="red",
        )
        ax.set_title(f"Real versus predicted results, epoch {self.current_epoch}")
        ax.set_xlabel("Real")
        ax.set_ylabel("Predicted")

        # Set the plot boundaries and aspect
        ax.set_xlim(y_np.min() * eps, y_np.max() * eps)
        ax.set_ylim(y_np.min() * eps, y_np.max() * eps)
        ax.set_aspect("equal")

        plt.savefig("images/test_plot.png")
        plt.show()

        self.log(f"test_r2", r2)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_pred = self.forward(x)

        loss = nn.functional.mse_loss(y_pred, y).reshape(1)

        tensorboard = self.logger.experiment
        tensorboard.add_scalars(
            "loss",
            {"val": loss, "train": self.current_train_loss},
            global_step=self.current_epoch,
        )

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
        eps = 1.1  # So that min/max values are not on the edge
        ax.scatter(y_np, y_pred_np)
        ax.plot(
            [y_np.min() * eps, y_np.max() * eps],
            [y_np.min() * eps, y_np.max() * eps],
            c="red",
        )
        ax.set_title(f"Real versus predicted results, epoch {self.current_epoch}")
        ax.set_xlabel("Real")
        ax.set_ylabel("Predicted")

        # Set the plot boundaries and aspect
        ax.set_xlim(y_np.min() * eps, y_np.max() * eps)
        ax.set_ylim(y_np.min() * eps, y_np.max() * eps)
        ax.set_aspect("equal")

        if self.current_epoch > 0:
            plt.gcf().savefig(f"images/version_{self._version}/{self.current_epoch}.png")

        tensorboard = self.logger.experiment
        tensorboard.add_figure(
            "real_vs_pred/val", plt.gcf(), global_step=self.current_epoch
        )

        # log the outputs!
        self.log("r2/val", r2, prog_bar=True)

        return loss

    def on_train_start(self):
        path = f"images/version_{self._version}/"
        try:
            print(f"Creating {path}")
            os.mkdir(path)
        except FileExistsError:
            import shutil

            print(f"Path {path} already exists. Overwriting all contents.")
            shutil.rmtree(path)
            os.mkdir(path)

    def on_test_start(self):

        files: list = os.listdir(f"images/version_{self._version}/")
        print(f"Saving gif with {len(files)} frames!")
        images = []

        # filepaths
        fp_in = f"images/version_{self._version}/*.png"
        fp_out = f"images/version_{self._version}/full.gif"

        with contextlib.ExitStack() as stack:

            # lazily load images
            imgs = []

            i = 1
            while True:
                try:
                    imgs.append(
                        stack.enter_context(
                            Image.open(f"images/version_{self._version}/{i}.png")
                        )
                    )
                    i += 1
                except Exception as e:
                    print(e)
                    break

            # extract  first image from iterator
            imgs = iter(imgs)
            img = next(imgs)

            # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
            img.save(
                fp=fp_out,
                format="GIF",
                append_images=imgs,
                save_all=True,
                duration=100,
                loop=0,
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MlpAnalyzer(Analyzer):
    def __init__(
        self,
        output_size,
        input_size,
        verbose: int = 0,
        lr=1e-3,
        hidden_size=200,
        batch_size: int = 100,
        checkpoint_path=None,
        max_train_epochs: int = 1000,
        n_augmentations: int = 10,
        num_workers = None
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

        # Automatically set num_workers if not provided
        self.num_workers = num_workers or os.cpu_count()
        print("Number of worker (cpu cores): " , self.num_workers)


        if checkpoint_path is not None:
            print(
                f"Loading checkpoint from {checkpoint_path} with input size {input_size}"
            )
            checkpoint = torch.load(checkpoint_path)
            # for k, v in checkpoint["state_dict"].items():
            #     print(k, v)

            for i in [0, 3, 6]:
                self.lit_model.backbone[i].weight = torch.nn.Parameter(
                    checkpoint["state_dict"][f"backbone.{i}.weight"]
                )
                self.lit_model.backbone[i].bias = torch.nn.Parameter(
                    checkpoint["state_dict"][f"backbone.{i}.bias"]
                )

            # print(self.lit_model.seq.layer)
            # self.lit_model = LitPlainMlp.load_from_checkpoint(
            #     checkpoint_path=checkpoint_path,
            #     output_dim=output_size,
            #     input_dim=input_size,
            # )

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

        print("Accelerator being used: ", accelerator)

        self.trainer = L.Trainer(
            accelerator=accelerator,
            # limit_val_batches=100,
            max_epochs=max_train_epochs,
            callbacks=[
                EarlyStopping(
                    monitor="r2/val", mode="max", patience=10, min_delta=0.01
                ),
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
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True
        )
        val_loader = data_utils.DataLoader(
            val_set, batch_size=valid_set_size, num_workers=self.num_workers, persistent_workers=True
        )

        self.trainer.fit(self.lit_model, train_loader, val_loader)

    def test(self, X_test, Y_test):
        dataset = CustomDataset(X_test, Y_test)

        test_dataloader = data_utils.DataLoader(
            dataset, batch_size=len(X_test), num_workers=self.num_workers, persistent_workers=True
        )

        self.trainer.test(self.lit_model, test_dataloader)

    def predict(self, X: pd.DataFrame):
        return self.lit_model.forward(X)
