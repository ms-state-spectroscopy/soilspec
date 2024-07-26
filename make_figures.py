from math import sin
import random
from turtle import xcor
from cubist import Cubist
from tqdm import trange
from analyzers.rf import RandomForestAnalyzer
from analyzers.cubist import CubistAnalyzer
from analyzers.two_part_mlp import LitMlp
import analyzers.utils as utils
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
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

SEED = 64
utils.seedEverything(SEED)

light_purple = "#C5A9C7"
dark_purple = "#543856"
purple = "#B38CB4"
gambodge_500 = "#EC9A29"
green = "#79A48A"
dark_green = "#065143"
light_green = "#AAC5B5"

sky_200 = "#bae6fd"
sky_700 = "#0369a1"
violet_500 = "#B38CB4"
blue_500 = "#3b82f6"
yellow_500 = "#eab308"
teal_200 = "#99f6e4"
teal_800 = "#115e59"
cambridge_200 = "#C2D6CA"
cambridge_800 = "#42614E"
pink_700 = "#be185d"
pink_600 = "#db2777"


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

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4 if self.fcn else 1e-5)
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

        ax = plt.subplot()
        eps = 1.1  # So that min/max values are not on the edge
        ax.scatter(y_np, y_pred_np, c="#B38CB4")
        ax.plot(
            [y_np.min() * eps, y_np.max() * eps],
            [y_np.min() * eps, y_np.max() * eps],
            c="#79A48A",
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


def hex_to_RGB(hex_str):
    """#FFFFFF -> [255,255,255]"""
    # Pass 16 to the integer function for change of base
    return [int(hex_str[i : i + 2], 16) for i in range(1, 6, 2)]


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1)) / 255
    c2_rgb = np.array(hex_to_RGB(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
    return [
        "#" + "".join([format(int(round(val * 255)), "02x") for val in item])
        for item in rgb_colors
    ]


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

# (
#     (X_train, Y_train),
#     (X_test, Y_test),
#     original_label_mean,
#     original_label_std,
# ) = ossl_db.loader.load(
#     labels=ossl_labels,
#     normalize_Y=True,
#     from_pkl=True,
#     include_unlabeled=False,
#     take_grad=False,
#     n_components=60,
# )

mississippi_labels = ["wilting_point"]

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

pca = PCA(n_components=80)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_val_pca = pca.transform(X_val)

# Remove a single pesky outlier
X_val_pca = X_val_pca[X_val_pca[:, 0] < 7]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1: Axes
fig.set_dpi(150.0)
fig.set_size_inches(12.0, 4.0)

ax1.grid(True)


ax1.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=violet_500, label="Train")
ax1.scatter(X_val_pca[:, 0], X_val_pca[:, 1], c=gambodge_500, label="Test")
ax1.set_xlabel("Principal Component 1")
ax1.set_ylabel("Principal Component 2")
ax1.legend()

denorm_scale = original_label_max - original_label_min
Y_train_denorm = Y_train * denorm_scale + original_label_min

sort_idx = np.argsort(Y_train_denorm.flatten())
print(sort_idx)
X_train_sorted = X_train[sort_idx]
Y_train_sorted = Y_train_denorm[sort_idx]


bottom_ten_X = X_train_sorted[:10]
bottom_ten_Y = Y_train_sorted[:10]

top_ten_X = X_train_sorted[-10:]
top_ten_Y = Y_train_sorted[-10:]
print(Y_train_sorted)


wavelengths = np.linspace(400, 2500, X_train.shape[-1])
# for spectrum in top_ten_X[:]:

#     ax2.plot(wavelengths, spectrum, c="red")

# for spectrum in bottom_ten_X[:]:

#     ax2.plot(wavelengths, spectrum, c="blue")


colors = get_color_gradient(cambridge_200, teal_800, len(Y_train_sorted))

ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel("Reflectance (%)")

for i in range(0, len(Y_train_sorted), 4):
    spectrum = X_train_sorted[i]
    ax2.plot(wavelengths, spectrum, color=colors[i])


final_model = LitModel.load_from_checkpoint(
    "/home/main/soilspec/named_ckpts/trial_9_cnn_cbd/checkpoints/epoch=879-step=3520.ckpt",
    input_dim=1053,
    hidden_size=200,
    output_dim=len(mississippi_labels),
    p=0.5,
    original_label_minmax=(original_label_min, original_label_max),
    pca=None,
    add_contrastive=False,
    augment=False,
    fcn=False,  # Use an FCN or a CNN
)

Y_pred = final_model.forward(torch.from_numpy(X_val).type(torch.float32).cuda())

print(Y_pred)

cubist = CubistAnalyzer()
cubist.train(X_train, Y_train)
Y_pred_cubist = cubist.predict(X_val)

print(Y_pred_cubist)

denorm_scale = original_label_max - original_label_min
Y_val_denorm = Y_val * denorm_scale + original_label_min
Y_pred_denorm = Y_pred * denorm_scale + original_label_min
Y_cubist_denorm = Y_pred_cubist * denorm_scale + original_label_min

eps = 1.1
ax3.grid(True)
ax3.scatter(
    Y_val_denorm,
    Y_pred_denorm.numpy(force=True),
    color=teal_800,
    label="CNN (best model)",
)
ax3.plot(
    [Y_val_denorm.min() * eps, Y_val_denorm.max() * eps],
    [Y_val_denorm.min() * eps, Y_val_denorm.max() * eps],
    c="#79A48A",
    linewidth=3,
)
ax3.scatter(Y_val_denorm, Y_cubist_denorm, color=pink_600, label="Cubist (baseline)")

ax3.legend()

# ax3.set_ylim([0.1, 0.55])
ax3.set_xlabel("True value (% water)")
ax3.set_ylabel("Predicted value (% water)")


# ax1.set_xlim(-6, 6)
# ax1.set_ylim(-1.2, 1.5)
# ax1.set_aspect("equal")
plt.show()
