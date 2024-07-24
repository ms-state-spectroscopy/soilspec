from tqdm import trange
import analyzers.utils as utils
import lightning as L
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
BATCH_SIZE = 32
SEED = 64

# 0. Set the seed.
utils.seedEverything(SEED)
torch.set_float32_matmul_precision("medium")

mississippi_labels = ["wilting_point"]
ossl_labels = [
    # "cf_usda.c236_w.pct",
    # "oc_usda.c729_w.pct",
    "clay.tot_usda.a334_w.pct",
    "sand.tot_usda.c60_w.pct",
    "silt.tot_usda.c62_w.pct",
    # "bd_usda.a4_g.cm3",
    "wr.1500kPa_usda.a417_w.pct",
    # "awc.33.1500kPa_usda.c80_w.frac",
]

(_, _, _, _, pca) = ossl_db.loader.load(
    labels=ossl_labels,
    normalize_Y=True,
    from_pkl=True,
    include_unlabeled=False,
    take_grad=False,
    n_components=120,
)

(
    (X_train, Y_train),
    (X_val, Y_val),
    original_label_mean,
    original_label_std,
) = mississippi_db.loader.load(
    labels=mississippi_labels,
    normalize_Y=True,
    from_pkl=False,
    train_split=185 / 225,
    take_grad=False,
    n_components=None,
    include_unlabeled=False,
)

# X_train = pca.transform(X_train)
# X_val = pca.transform(X_val)


class LitModel(L.LightningModule):
    def __init__(
        self, input_dim: int, hidden_size: int, output_dim: int, pca, p: float = 0.2
    ):
        super().__init__()

        backbone_output_size = 4  # sand, silt, clay, wr

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
            nn.Linear(hidden_size, backbone_output_size),
        )

        self.head = nn.Sequential(
            nn.Linear(input_dim + backbone_output_size, hidden_size),
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
        backbone_y: torch.Tensor = self.backbone(x)  # sand, silt, clay, wr
        # print(f"X has shape {x.shape}")
        # print(backbone_y.numpy(force=True))

        if self.training:
            noise = torch.zeros_like(x)
            noise.normal_(0, std=1e-2)
            scale = torch.rand(1).cuda() + 0.5
            noisy_x = x * scale + noise.cuda()
            y_pred = self.head(torch.cat((noisy_x, backbone_y), dim=-1))
        else:
            y_pred = self.head(torch.cat((x, backbone_y), dim=-1))
        # print(torch.cat((x, backbone_y), dim=-1))
        # print(torch.cat((x, backbone_y), dim=-1).shape)
        # print("EQUALS")
        # print(y_pred)
        return y_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.current_train_loss = loss

        self.log("loss/train", loss, prog_bar=True)

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

        print(f"y_np has shape {y_np.shape}")
        print(f"y_pred_np has shape {y_pred_np.shape}")
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
    pca = PCA(n_components=120)
    pca.fit(np.vstack((X_train, X_val)))

    model = LitModel(
        input_dim=X_train.shape[-1],
        hidden_size=200,
        output_dim=len(mississippi_labels),
        pca=pca,
        p=0.5,
    )

    # CKPT_PATH = (
    #     "/home/main/soilspec/named_ckpts/ossl/checkpoints/epoch=499-step=2500.ckpt"
    # )

    CKPT_PATH = "/home/main/soilspec/named_ckpts/ossl_uncompressed/checkpoints/epoch=1159-step=10440.ckpt"

    backbone_ckpt = torch.load(CKPT_PATH)

    backbone_weights = {
        k: v for k, v in backbone_ckpt["state_dict"].items() if k.startswith("seq.")
    }
    print(backbone_weights)

    with torch.no_grad():
        model.backbone.weights = backbone_weights

    for param in model.backbone.parameters():
        param.requires_grad = False
    # exit()

    trainer = L.Trainer(
        callbacks=[
            EarlyStopping(monitor="r2/val", mode="max", patience=100, min_delta=0.01),
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
        batch_size=len(X_val),
        num_workers=4,
    )

    # print(f"X_train has shape {X_train.shape}")
    # print(f"X_val has shape {X_val.shape}")
    # utils.plotSpectraFromNumpy(X_val, n=10)
    # exit()

    # Sanity check our data
    print(f"Training with {len(Y_train)} train labels, {len(Y_val)} val labels")

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)
