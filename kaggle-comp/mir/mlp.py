from analyzer import Analyzer
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras import saving


class MlpAnalyzer(Analyzer):
    def __init__(
        self,
        verbose: int = 0,
        lr=1e-3,
        hidden_size=320,
        epochs=500,
        train_split=0.8,
        batch_size=32,
        activation="relu",
        n_logits=3,
    ) -> None:
        super().__init__(verbose=verbose)

        self.activation = activation
        self.lr = lr
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.train_split = train_split
        self.batch_size = batch_size

        self.normalizer = tf.keras.layers.Normalization(axis=-1)

        self.model = keras.Sequential(
            [
                self.normalizer,
                layers.Dense(hidden_size, activation=activation),
                layers.Dense(hidden_size, activation=activation),
                layers.Dense(1),
            ]
        )

        self.model.compile(
            loss="mean_absolute_error",
            optimizer=tf.keras.optimizers.Adam(lr),
        )

    def train(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.normalizer.adapt(np.array(X))
        self.model.fit(X, Y, batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
