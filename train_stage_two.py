from math import sin
import random
from turtle import xcor
from tqdm import trange
from analyzers.analyzer import RandomForestAnalyzer
from analyzers.cubist import CubistAnalyzer
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
from torch.nn import BatchNorm1d
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
    original_label_max,
    original_label_min,
) = mississippi_db.loader.load(
    labels=mississippi_labels,
    normalize_Y=True,
    from_pkl=False,
    train_split=100 / 225,
    take_grad=False,
    n_components=None,
    include_unlabeled=False,
)


# X_train = pca.transform(X_train)
# X_val = pca.transform(X_val)


class LitModel(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        p: float = 0.2,
        original_label_minmax=(1.0, 1.0),
        pca: PCA = None,
        add_contrastive=False,
        total_epochs=1e3,
        augment=True,
        fcn=True,
    ):
        super().__init__()

        if fcn:
            self.head = nn.Sequential(
                # Layer 1
                nn.Linear(input_dim, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(p),
                # Layer 2
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(p),
                # Layer 3
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(p),
                # Output
                nn.Linear(hidden_size, output_dim),
            )
        else:
            self.head = nn.Sequential(
                # Layer 1
                nn.Conv1d(1, 32, 3, 2),
                nn.MaxPool1d(3, 2, 1),
                nn.LeakyReLU(),
                # Layer 2
                nn.Conv1d(32, 64, 3, 2),
                nn.MaxPool1d(3, 2, 1),
                nn.LeakyReLU(),
                # Layer 3
                nn.Conv1d(64, 128, 3, 2),
                nn.MaxPool1d(3, 2, 1),
                nn.LeakyReLU(),
                # Layer 4
                # nn.Conv1d(128, 256, 3, 2),
                # nn.BatchNorm1d(256),
                # nn.LeakyReLU(),
                # Output
                nn.Flatten(),
                nn.Linear(2048, output_dim),
            )

        self.current_train_loss = 0.0
        self.current_r2 = 0.0
        self.original_minmax = original_label_minmax
        self.denorm_scale = original_label_minmax[1] - original_label_minmax[0]
        self.pca = pca
        self.add_contrastive = add_contrastive
        self.total_epochs = total_epochs
        self.augment = augment
        self.fcn = fcn

    def forward(self, x: torch.Tensor):

        # if self.pca is not None:
        #     compressed_x = self.pca.transform(x.numpy(force=True))
        #     compressed_x = torch.from_numpy(compressed_x).type(torch.float32).cuda()
        #     backbone_y: torch.Tensor = self.backbone(
        #         compressed_x
        #     )  # sand, silt, clay, wr
        # else:
        #     backbone_y = self.backbone(x)
        # print(f"X has shape {x.shape}")
        # print(backbone_y.numpy(force=True))

        if self.training:
            noise = torch.zeros_like(x)
            noise.normal_(0, std=1e-2)
            scale = torch.rand(1).cuda() + 0.5

            # Generate Perlin noise
            factor_1 = (random.random() - 0.5) * 10
            factor_pi = (random.random() - 0.5) * 10
            factor_e = (random.random() - 0.5) * 10
            scale_1 = (random.random() - 0.5) * 10
            scale_pi = (random.random() - 0.5) * 10
            scale_e = (random.random() - 0.5) * 10
            factor_total = random.random() / 100

            u = np.linspace(0, 10, x.shape[-1])
            perlin = factor_total * (
                factor_1 * np.sin(scale_1 * u)
                + factor_e * np.sin(scale_e * u)
                + factor_pi * np.sin(scale_pi * u)
            )
            # plt.plot(perlin)
            # plt.show()

            if self.augment:
                noise += torch.from_numpy(perlin).type(torch.float32).cuda()
                noise[:, :-2] = 0.0
                noisy_x = x * scale + noise.cuda()
            else:
                noisy_x = x

            if self.pca is not None:
                x = self.pca.transform(noisy_x.numpy(force=True))
                x = torch.from_numpy(x).type(torch.float32).cuda()

                if not self.fcn:
                    x = x[:, None, :]
                y_pred = self.head(x)
            else:
                if not self.fcn:
                    noisy_x = noisy_x[:, None, :]
                y_pred = self.head(noisy_x)

        else:

            if self.pca is not None:
                x = self.pca.transform(x.numpy(force=True))
                x = torch.from_numpy(x).type(torch.float32).cuda()

            if not self.fcn:
                x = x[:, None, :]

            y_pred = self.head(x)
        # print(torch.cat((x, backbone_y), dim=-1))
        # print(torch.cat((x, backbone_y), dim=-1).shape)
        # print("EQUALS")
        # print(y_pred)
        return y_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)

        nonnan_indices = torch.logical_not(torch.isnan(y)).any(dim=-1)
        y_masked = y[nonnan_indices]
        y_pred_masked = y_pred[nonnan_indices]

        loss = F.mse_loss(y_pred_masked, y_masked)

        # Contrastive loss
        y_pred_2 = self.forward(x)
        contrastive_loss = F.mse_loss(y_pred, y_pred_2)

        self.log("loss/contrastive", contrastive_loss, prog_bar=True)

        if self.add_contrastive:
            loss += (self.current_epoch / self.total_epochs) * contrastive_loss

        self.current_train_loss = loss

        self.log("loss/train", loss, prog_bar=False)
        self.log("rmse/train", torch.sqrt(loss) * self.denorm_scale, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def test_step(self, batch, batch_idx):

        x, y = batch

        y_pred = self.forward(x)

        nonnan_indices = torch.logical_not(torch.isnan(y)).any(dim=-1)
        y_masked = y[nonnan_indices]
        y_pred_masked = y_pred[nonnan_indices]

        loss = F.mse_loss(y_pred_masked, y_masked)

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

        self.log("rmse/test", torch.sqrt(loss) * self.denorm_scale)

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

        nonnan_indices = torch.logical_not(torch.isnan(y)).any(dim=-1)
        y_masked = y[nonnan_indices]
        y_pred_masked = y_pred[nonnan_indices]

        loss = F.mse_loss(y_pred_masked, y_masked)

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
        self.current_r2 = r2

        self.log(
            "rmse/val",
            torch.sqrt(loss) * self.denorm_scale,
            prog_bar=True,
        )

        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return y_pred


if __name__ == "__main__":
    pca = PCA(n_components=80)

    clay_model = LitModel(
        input_dim=1051,
        hidden_size=200,
        output_dim=len(mississippi_labels),
        p=0.5,
        original_label_minmax=(original_label_min, original_label_max),
        pca=None,
        add_contrastive=False,
    )

    checkpoint = torch.load(
        "/home/main/soilspec/named_ckpts/ossl_clay/checkpoints/epoch=1959-step=168560.ckpt"
    )
    clay_weights = {
        k.replace("seq", "head"): v for k, v in checkpoint["state_dict"].items()
    }
    clay_model.weights = clay_weights

    clay_model.eval()
    clay_pred = clay_model.forward(torch.from_numpy(X_train).type(torch.float32))
    print(clay_pred)

    bd_model = LitModel(
        input_dim=1051,
        hidden_size=200,
        output_dim=len(mississippi_labels),
        p=0.5,
        original_label_minmax=(original_label_min, original_label_max),
        pca=None,
        add_contrastive=False,
    )

    checkpoint = torch.load(
        "/home/main/soilspec/named_ckpts/ossl_bd/checkpoints/epoch=1999-step=10000.ckpt"
    )
    bd_weights = {
        k.replace("seq", "head"): v for k, v in checkpoint["state_dict"].items()
    }
    bd_model.weights = bd_weights

    bd_model.eval()
    bd_pred = bd_model.forward(torch.from_numpy(X_train).type(torch.float32))
    print(bd_pred)

    bd_pred_test = bd_model.forward(torch.from_numpy(X_val).type(torch.float32))
    print(bd_pred_test)
    clay_pred_test = clay_model.forward(torch.from_numpy(X_val).type(torch.float32))
    print(clay_pred_test)

    print(X_train.shape)
    X_train = np.hstack(
        (X_train, bd_pred.numpy(force=True), clay_pred.numpy(force=True))
    )

    X_val = np.hstack(
        (X_val, bd_pred_test.numpy(force=True), clay_pred_test.numpy(force=True))
    )
    print(X_train.shape)

    pca.fit(X_train)

    model = LitModel(
        input_dim=1053,
        hidden_size=200,
        output_dim=len(mississippi_labels),
        p=0.5,
        original_label_minmax=(original_label_min, original_label_max),
        pca=pca,
        add_contrastive=False,
        augment=True,
        fcn=False,
    )
    # CKPT_PATH = (
    #     "/home/main/soilspec/named_ckpts/ossl/checkpoints/epoch=499-step=2500.ckpt"
    # )

    # CKPT_PATH = "/home/main/soilspec/named_ckpts/ossl_uncompressed/checkpoints/epoch=1159-step=10440.ckpt"

    # backbone_ckpt = torch.load(CKPT_PATH)

    # backbone_weights = {
    #     k: v for k, v in backbone_ckpt["state_dict"].items() if k.startswith("seq.")
    # }
    # print(backbone_weights)

    # with torch.no_grad():
    #     model.backbone.weights = backbone_weights

    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    # exit()

    # X_train = pca.transform(X_train)
    # X_val = pca.transform(X_val)
    # analyzer = CubistAnalyzer()
    # # analyzer = RandomForestAnalyzer()
    # analyzer.train(X_train, Y_train)
    # r2 = analyzer.test(X_val, Y_val)
    # print(r2)

    # exit()
    checkpoint_callback = ModelCheckpoint(monitor="rmse/val", mode="min", verbose=True)

    trainer = L.Trainer(
        callbacks=[
            EarlyStopping(monitor="rmse/val", mode="min", patience=100, min_delta=0.01),
            checkpoint_callback,
            # StochasticWeightAveraging(swa_lrs=1e-2),
        ],
        check_val_every_n_epoch=20,
        max_epochs=1000,
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
