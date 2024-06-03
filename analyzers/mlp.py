from analyzers.analyzer import Analyzer
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import tensorflow as tf
from tensorflow import keras
from keras import callbacks, layers, Model, saving, regularizers


class MlpAnalyzer(Analyzer):
    def __init__(
        self,
        verbose: int = 0,
        lr=1e-3,
        hidden_size=320,
        activation="relu",
        n_logits=3,
    ) -> None:
        super().__init__(verbose=verbose)

        self.activation = activation
        self.lr = lr
        self.hidden_size = hidden_size

        self.normalizer = tf.keras.layers.Normalization(axis=-1)

        self.model = keras.Sequential(
            [
                self.normalizer,
                layers.Dense(
                    hidden_size,
                    activation=activation,
                    activity_regularizer=regularizers.l2(0.01),
                ),
                layers.Dense(
                    hidden_size,
                    activation=activation,
                    activity_regularizer=regularizers.l2(0.01),
                ),
                layers.Dense(
                    hidden_size,
                    activation=activation,
                    activity_regularizer=regularizers.l2(0.01),
                ),
                layers.Dense(n_logits),
            ]
        )

        self.model.compile(
            loss="mean_absolute_error",
            optimizer=tf.keras.optimizers.Adam(lr),
        )

    def train(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        epochs=500,
        val_split=0.2,
        early_stop_patience=50,
        batch_size=32
    ):
        self.normalizer.adapt(np.array(X))
        return self.model.fit(
            X,
            Y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=val_split,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor="val_loss", patience=early_stop_patience
                )
            ],
        )

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
