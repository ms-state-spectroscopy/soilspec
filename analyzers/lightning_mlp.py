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


# define the LightningModule
class LitMlp(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        torch.set_float32_matmul_precision("medium")
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.model(x)
        loss = nn.functional.mse_loss(y_pred, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.model(x)
        test_loss = nn.functional.mse_loss(y_pred, y)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.model(x)
        val_loss = nn.functional.mse_loss(y_pred, y)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class LightningMlpAnalyzer(Analyzer):
    def __init__(
        self,
        verbose: int = 0,
        lr=1e-3,
        hidden_size=200,
        activation="relu",
        n_logits=3,
        input_size=None,
        checkpoint_path=None,
    ) -> None:
        super().__init__(verbose=verbose)

        self.activation = activation
        self.lr = lr
        self.hidden_size = hidden_size

        # self.normalizer = tf.keras.layers.Normalization(axis=-1)

        self.model = nn.Sequential(
            (
                nn.Linear(input_size, hidden_size)
                if input_size is not None
                else nn.LazyLinear(hidden_size)
            ),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # 1 logit, 1 feature predicted at a time
            nn.Linear(hidden_size, 1),
        )

        self.lit_model = LitMlp(self.model)

        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.lit_model = LitMlp.load_from_checkpoint(
                checkpoint_path=checkpoint_path, model=self.model
            )

        # self.model = keras.Sequential(
        #     [
        #         self.normalizer,
        #         layers.Dense(
        #             hidden_size,
        #             activation=activation,
        #             activity_regularizer=regularizers.l2(0.01),
        #         ),
        #         layers.Dense(
        #             hidden_size,
        #             activation=activation,
        #             activity_regularizer=regularizers.l2(0.01),
        #         ),
        #         # layers.Dense(
        #         #     hidden_size,
        #         #     activation=activation,
        #         #     activity_regularizer=regularizers.l2(0.01),
        #         # ),
        #         layers.Dense(n_logits),
        #     ]
        # )

        # self.optimizer = tf.keras.optimizers.Adam(lr)

        # self.loss_fn = nn.functional.mse_loss

        # self.model.compile(
        #     loss="mean_absolute_error",
        #     optimizer=tf.keras.optimizers.Adam(lr),
        # )

    def setHeadWeights(self, new_weights):
        self.model.get_layer(index=-2).set_weights(new_weights)

    def getHeadWeights(self):
        return self.model.get_layer(index=-2).get_weights()

    def train(
        self,
        dataset: CustomDataset,
        epochs=500,
        val_split=0.2,
        early_stop_patience=50,
        batch_size=32,
    ):
        # use 20% of training data for validation
        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = len(dataset) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        train_set, valid_set = torch.utils.data.random_split(
            dataset, [train_set_size, valid_set_size], generator=seed
        )
        # train = data_utils.TensorDataset(X.values, Y.values)
        train_loader = data_utils.DataLoader(
            train_set, batch_size=10, shuffle=True, num_workers=19
        )
        valid_loader = data_utils.DataLoader(valid_set, batch_size=10, num_workers=19)
        trainer = L.Trainer(max_epochs=epochs, accelerator="gpu", limit_val_batches=100)
        trainer.fit(self.lit_model, train_loader, valid_loader)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
