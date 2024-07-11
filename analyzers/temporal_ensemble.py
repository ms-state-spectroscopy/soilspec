from analyzers.analyzer import Analyzer
import numpy as np
import pandas as pd
from keras import callbacks, layers, regularizers
from tqdm import trange, tqdm
from torch import optim, nn, utils, Tensor

from analyzers.utils import rsquared
import torch
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from analyzers.utils import CustomDataset
import torch.utils.data as data_utils
from torchmetrics.regression import R2Score


import lightning as L
from lightning.pytorch.callbacks import DeviceStatsMonitor, StochasticWeightAveraging
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.utilities import CombinedLoader

from lightning.pytorch.tuner import Tuner


class LitTemporalEnsemble(L.LightningModule):

    def __init__(
        self,
        max_train_epochs: int,
        p=0.5,
        max_val=30.0,
        ramp_up_mult=-5.0,
        batch_size: int = 100,
        n_labeled=100,
        n_samples=60000,
        # n_classes=10,
        lr=0.002,
        std=0.15,
        alpha=0.6,
        input_dim: int = None,
        hidden_size: int = 200,
        output_dim: int = 1,
        datasets=None,
    ):
        super(LitTemporalEnsemble, self).__init__()

        self.lr = lr
        self.max_epochs = max_train_epochs
        self.max_val = max_val
        self.n_classes = output_dim
        self.n_samples = n_samples
        self.n_labeled = n_labeled  # TODO: Set this automatically
        self.std = std
        self.alpha = alpha
        self.batch_size = batch_size
        self.ramp_up_mult = ramp_up_mult

        self.Z = None

        # Model layers
        self.l1 = (
            nn.Linear(input_dim, hidden_size)
            if input_dim is not None
            else nn.LazyLinear(hidden_size)
        )
        self.relu1 = nn.LeakyReLU()

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU()

        self.drop = nn.Dropout(p)
        self.head = nn.Linear(hidden_size, output_dim)

    def augment(
        self,
        x,
        vertical_scale: float = 0.2,
        noise_scale: float = 0.00005,
        seed: int = None,
    ):
        rng = torch.Generator(device="cuda")
        if seed is None:
            rng.seed()
        else:
            rng.manual_seed(seed)

        random_scale = torch.rand((1), generator=rng, device="cuda") * 2 - 1
        x_ = x + (x * random_scale * vertical_scale)

        # Gaussian noise
        noise = torch.randn_like(x, device="cuda") * 2 - 1
        x_ += noise * noise_scale

        # print(f"{x} -> {x_}")
        return x_

    def forward(self, x: torch.Tensor):

        if self.training:
            # x_np = x.numpy(force=True).flatten()
            # t = range(len(x_np))
            # print(x_np)
            # plt.plot(t, x_np)
            x = self.augment(x)
            # plt.plot(t, x.numpy(force=True).flatten())
            # plt.show()

        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)

        x = self.drop(x)
        y_pred = self.head(x)
        return y_pred

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)

        # print(x)
        # print(y)
        # print(y_pred)

        # loss, supervised_loss, unsupervised_loss, nbsup = self.temporal_loss(
        #     y_pred, zcomp, self.w, y
        # )

        # metric = R2Score().cuda()

        y: torch.Tensor

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
        self.log("val/r2", r2)

    def ramp_up(self, epoch, max_epochs, max_val, mult):
        if epoch == 0:
            return 0.0
        elif epoch >= max_epochs:
            return max_val
        return max_val * np.exp(mult * (1.0 - float(epoch) / max_epochs) ** 2)

    def weight_schedule(self, max_epochs, max_val, mult, n_labeled, n_samples):

        epoch = float(self.current_epoch)

        max_val = max_val * (float(n_labeled) / n_samples)
        # return self.ramp_up(epoch, max_epochs, max_val, mult)

        # simple linear ramp-up!
        print(f"({epoch}/{max_epochs}) -> {epoch / max_epochs}")
        return epoch / max_epochs

    def on_train_epoch_start(self):
        w = self.weight_schedule(
            self.max_epochs,
            self.max_val,
            self.ramp_up_mult,
            self.n_labeled,
            self.n_samples,
        )

        self.log("unsupervised_loss_weight", w)

        # turn it into a usable pytorch object
        self.w = torch.autograd.Variable(
            torch.tensor([w], device="cuda"), requires_grad=False
        )

    def on_train_epoch_end(self):
        self.Z = self.alpha * self.Z + (1.0 - self.alpha) * self.outputs
        self.z = self.Z * (1.0 / (1.0 - self.alpha ** (self.current_epoch + 1)))

    def training_step(self, batch, batch_idx):

        # Check if temporal variables need to be initialized
        if self.Z is None:
            self.Z = (
                torch.zeros(self.ntrain, self.n_classes).float().cuda()
            )  # intermediate values
            self.z = (
                torch.zeros(self.ntrain, self.n_classes).float().cuda()
            )  # temporal outputs
            self.outputs = (
                torch.zeros(self.ntrain, self.n_classes).float().cuda()
            )  # current outputs

        x, y = batch
        y_pred = self.forward(x)

        zcomp = torch.autograd.Variable(
            self.z[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size],
            requires_grad=False,
        )
        loss, supervised_loss, unsupervised_loss, nbsup = self.temporal_loss(
            y_pred, zcomp, self.w, y
        )

        # log the outputs!
        self.log_dict(
            {
                "train_loss": loss,
                "sup_loss": nbsup * supervised_loss,
                "unsup_loss": unsupervised_loss,
            },
            prog_bar=True,
        )

        return loss

    def temporal_loss(self, out1, out2, w, labels):

        # MSE between current and temporal outputs
        def mse_loss(out1: torch.Tensor, out2):
            quad_diff = torch.sum(
                (F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2
            )
            return quad_diff / out1.numel()

        def masked_crossentropy(out: torch.Tensor, labels):
            labels = labels.flatten()
            cond = labels >= 0
            nnz = torch.nonzero(cond)
            nbsup = len(nnz)
            # check if labeled samples in batch, return 0 if none
            if nbsup > 0:
                # print(out.shape)
                # print(nnz.shape)
                masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
                masked_labels = labels[cond]
                # loss = F.cross_entropy(masked_outputs, masked_labels)
                loss = torch.nn.functional.mse_loss(
                    masked_outputs.flatten(), masked_labels
                )
                # print(f"Labels: {labels}")
                return loss, nbsup
            return (
                torch.autograd.Variable(
                    torch.FloatTensor([0.0]).cuda(), requires_grad=False
                ),
                0,
            )

        sup_loss, nbsup = masked_crossentropy(out1, labels)
        # unsup_loss = mse_loss(out1, out2)
        unsup_loss = torch.nn.functional.mse_loss(out1, out2, reduction="mean")
        unsup_loss *= 0.0
        return sup_loss + w * unsup_loss, sup_loss, unsup_loss, nbsup

    def configure_optimizers(self):
        # TODO: Investigate betas
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99))

        # Setting gamma to 1.0 basically turns the lr_scheduler off
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=1.0
        )

        return [optimizer], [lr_scheduler]

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)

        print(x)
        print(y)
        print(y_pred)

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

        y: torch.Tensor

        y_np = y.numpy(force=True).reshape((-1, 1))
        y_pred_np = y_pred.numpy(force=True).reshape((-1, 1))

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
            "test/real_vs_pred", plt.gcf(), global_step=self.current_epoch
        )

        # log the outputs!
        self.log("test/r2", r2)


class TemporalEnsembleAnalyzer(Analyzer):
    def __init__(
        self,
        output_size,
        verbose: int = 0,
        lr=1e-4,
        hidden_size=200,
        batch_size: int = 100,
        input_size=None,
        checkpoint_path=None,
        max_train_epochs: int = 1000,
    ) -> None:
        super().__init__(verbose=verbose)

        torch.set_float32_matmul_precision("medium")

        self.lr = lr
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_train_epochs = max_train_epochs

        self.lit_model = LitTemporalEnsemble(
            input_dim=input_size,
            hidden_size=hidden_size,
            lr=lr,
            output_dim=output_size,
            batch_size=batch_size,
            max_train_epochs=max_train_epochs,
        )

        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.lit_model = LitTemporalEnsemble.load_from_checkpoint(
                checkpoint_path=checkpoint_path, output_dim=output_size
            )

        self.trainer = L.Trainer(
            # accelerator="gpu",
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

    def resetTrainer(self):
        self.trainer = L.Trainer(
            # accelerator="gpu",
            # limit_val_batches=100,
            max_epochs=self.max_train_epochs,
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

    def train(self, X_train: pd.DataFrame, Y_train: pd.DataFrame):

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
            val_set, batch_size=len(val_set), num_workers=19
        )

        self.lit_model.ntrain = len(train_set)

        print(Y_train.notna())
        n_labeled = Y_train.notna().sum().values[0]
        self.lit_model.n_labeled = n_labeled
        print(f"n_labeled is: {n_labeled}")

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

    def predict(self, X: torch.Tensor):
        # print(X)
        return self.lit_model.forward(X)
