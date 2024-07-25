from tqdm import trange
from analyzers.cubist import CubistAnalyzer
from analyzers.rf import RandomForestAnalyzer
import analyzers.utils as utils
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import mississippi_db
import mississippi_db.loader
import numpy as np
import os
import ossl_db.loader
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

# For .png -> .gif
import contextlib
from PIL import Image


"""
General steps:

1. Load data with physical indicators from OSSL
2. Train an MLP on the data
3. Test on the MS dataset
"""

# PARAMS
BATCH_SIZE = 256
SEED = 64

# 0. Set the seed.
utils.seedEverything(SEED)
torch.set_float32_matmul_precision("medium")


ossl_labels = [
    # "cf_usda.c236_w.pct",
    # "oc_usda.c729_w.pct",
    # "clay.tot_usda.a334_w.pct",
    # "sand.tot_usda.c60_w.pct",
    # "silt.tot_usda.c62_w.pct",
    # "bd_usda.a4_g.cm3",
    "wr.1500kPa_usda.a417_w.pct",
    # "awc.33.1500kPa_usda.c80_w.frac",
]

((X_train, Y_train), (X_val, Y_val), original_label_min, original_label_max, pca) = (
    ossl_db.loader.load(
        labels=ossl_labels,
        normalize_Y=True,
        from_pkl=True,
        include_unlabeled=False,
        take_grad=False,
        n_components=None,
    )
)

# pca: PCA

# Build a Cubist model for WR prediction
# analyzer = CubistAnalyzer(neighbors=None)
# analyzer.train(X_train, Y_train)
# r2 = analyzer.test(X_val, Y_val)
# print(r2)

# analyzer = RandomForestAnalyzer(verbose=1)
# analyzer.train(X_train, Y_train)
# r2 = analyzer.test(X_val, Y_val)
# print(r2)
# exit()


class LitModel(L.LightningModule):
    def __init__(
        self, input_dim: int, hidden_size: int, output_dim: int, p: float = 0.2
    ):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size, output_dim),
        )
        self.current_train_loss = 0.0

    def forward(self, x):
        return self.seq(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.current_train_loss = loss

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

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
            r2 = utils.rsquared(y_np, y_pred_np)

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

        plt.savefig("test_plot.png")
        plt.show()

        self.log(f"test_r2", r2)

        return loss

    def on_test_start(self):

        files: list = os.listdir(f"version_{self._version}/")
        print(f"Saving gif with {len(files)} frames!")
        images = []

        # filepaths
        fp_in = f"version_{self._version}/*.png"
        fp_out = f"version_{self._version}/full.gif"

        with contextlib.ExitStack() as stack:

            # lazily load images
            imgs = []

            i = 1
            for i in trange(9, 2000, 5):
                try:
                    imgs.append(
                        stack.enter_context(
                            Image.open(f"version_{self._version}/{i}.png")
                        )
                    )
                    i += 1  # skip every ten images
                except Exception as e:
                    # print(e)
                    continue

            # extract  first image from iterator
            imgs = iter(imgs)
            img = next(imgs)

            # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
            img.save(
                fp=fp_out,
                format="GIF",
                append_images=imgs,
                save_all=True,
                duration=10,
                loop=0,
            )

    def validation_step(self, batch, batch_idx):
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
            r2 = utils.rsquared(y_np, y_pred_np)

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
            plt.gcf().savefig(f"version_{self._version}/{self.current_epoch}.png")

        tensorboard = self.logger.experiment
        tensorboard.add_figure(
            "real_vs_pred/val", plt.gcf(), global_step=self.current_epoch
        )

        # log the outputs!
        self.log("r2/val", r2, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return y_pred


if __name__ == "__main__":

    checkpoint_callback = ModelCheckpoint(monitor="r2/val", mode="max", verbose=True)

    model = LitModel(
        input_dim=X_train.shape[-1], hidden_size=200, output_dim=len(ossl_labels), p=0.2
    )

    # model = LitModel.load_from_checkpoint(
    #     "/home/main/soilspec/lightning_logs/version_11/checkpoints/epoch=1159-step=10440.ckpt",
    #     input_dim=X_train.shape[-1],
    #     hidden_size=400,
    #     output_dim=len(ossl_labels),
    #     p=0.2,
    # )
    trainer = L.Trainer(
        callbacks=[
            EarlyStopping(monitor="r2/val", mode="max", patience=100, min_delta=0.01),
            checkpoint_callback,
            # StochasticWeightAveraging(swa_lrs=1e-2),
        ],
        check_val_every_n_epoch=20,
        max_epochs=2000,
    )

    train_loader = data_utils.DataLoader(
        utils.CustomDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    val_loader = data_utils.DataLoader(
        utils.CustomDataset(X_val, Y_val),
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    # utils.plotSpectraFromNumpy(X_val, n=10)

    # Sanity check our data
    print(f"Training with {len(Y_train)} train labels, {len(Y_val)} val labels")

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)
