from analyzers.analyzer import Analyzer
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import tensorflow as tf
from tensorflow import keras
from keras import callbacks, layers, Model, saving, regularizers
from tqdm import trange, tqdm


class MlpAnalyzer(Analyzer):
    def __init__(
        self,
        verbose: int = 0,
        lr=1e-3,
        hidden_size=200,
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

        self.optimizer = tf.keras.optimizers.Adam(lr)

        self.loss_fn = tf.keras.losses.MAE

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
        batch_size=32,
    ):
        self.normalizer.adapt(np.array(X))

        split_idx = int(len(X) * val_split)
        X_val = X.iloc[:split_idx, :]
        Y_val = Y.iloc[:split_idx, :]
        X_train = X.iloc[split_idx:, :]
        Y_train = Y.iloc[split_idx:, :]

        batch_indices = []
        idx = 0
        while idx < len(X_train):
            batch_indices.append(idx)
            idx += batch_size

        val_batch_indices = []
        idx = 0
        while idx < len(X_train):
            val_batch_indices.append(idx)
            idx += batch_size

        val_metric = keras.metrics.MeanAbsoluteError()
        best_val_loss = 99999.9
        epochs_until_stop = early_stop_patience

        history = {"loss": [], "val_loss": []}

        with tqdm(total=epochs, colour="blue") as pbar:
            for epoch in range(epochs):
                pbar.update()

                for step, batch_idx in enumerate(batch_indices):
                    x_batch_train = X_train.iloc[batch_idx : batch_idx + batch_size, :]
                    y_batch_train = Y_train.iloc[batch_idx : batch_idx + batch_size, :]

                    with tf.GradientTape() as tape:
                        logits = self.model(x_batch_train, training=True)

                        loss_value = tf.math.reduce_mean(
                            self.loss_fn(y_batch_train, logits)
                        )

                    grads = tape.gradient(loss_value, self.model.trainable_weights)

                    self.optimizer.apply_gradients(
                        zip(grads, self.model.trainable_weights)
                    )

                    # if step % 200 == 0:
                    #     # print(
                    #     #     f"Training loss (for one batch) at step {step}:{float(loss_value):.4f}"
                    #     # )
                    #     # print("Seen so far: %s samples" % ((step + 1) * batch_size))

                for step, batch_idx in enumerate(val_batch_indices):
                    x_batch_val = X_val.iloc[batch_idx : batch_idx + batch_size, :]
                    y_batch_val = Y_val.iloc[batch_idx : batch_idx + batch_size, :]
                    val_logits = self.model(x_batch_val, training=False)
                    # Update val metrics
                    # tf.math.reduce_mean(self.loss_fn(y_batch_val, val_logits))
                    val_metric.update_state(y_batch_val, val_logits)
                val_loss = val_metric.result()
                val_metric.reset_state()

                # For early stopping
                if epochs_until_stop == 0:
                    break
                elif val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_until_stop = early_stop_patience
                else:
                    epochs_until_stop -= 1

                history["loss"].append(float(loss_value))
                history["val_loss"].append(float(val_loss))

                pbar.set_description(
                    f"Val loss:{float(val_loss):.3f}. Best: {best_val_loss:.3f}. Waiting {epochs_until_stop} epochs."
                )

            # return self.model.fit(
            #     X,
            #     Y,
            #     batch_size=batch_size,
            #     epochs=epochs,
            #     validation_split=val_split,
            #     callbacks=[
            #         callbacks.EarlyStopping(
            #             monitor="val_loss", patience=early_stop_patience
            #         )
            #     ],
            # )

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
