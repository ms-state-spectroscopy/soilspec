import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress annoying tensorflow warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from tensorflow import keras
from keras import layers, Model
from keras import saving
import keras_tuner as kt

# Weights and Biases related imports
import wandb

wandb.login()
from wandb.integration.keras import WandbMetricsLogger


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

"""
General steps:

1. Load spectra and labels
2. Preprocess spectra:
    a. Evenly interpolate to 5nm scale from 1350-2550 (length 241)
3. Build the MLP: 2 dense layers
4. Evaluate
"""


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 3])
    plt.xlabel("Epoch")
    plt.ylabel("Error [Estimated organic Carbon, % by weight]")
    plt.legend()
    plt.grid(True)


def build_and_compile_model(config: dict) -> Model:

    model = keras.Sequential(
        [
            normalizer,
            layers.Dense(config["hidden_size"], activation="relu"),
            layers.Dense(config["hidden_size"], activation="relu"),
            layers.Dense(3),
        ]
    )

    model.compile(
        loss="mean_absolute_error",
        optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
    )
    return model


SPECTRUM_START_COLUMN = 55
INCLUDE_EC = True

config = dict(
    learning_rate=1e-3, hidden_size=320, epochs=500, train_split=0.8, batch_size=32
)

dataset = pd.read_csv("neospectra_db/Neospectra_WoodwellKSSL_avg_soil+site+NIR.csv")

# Drop NaNs for labels
dataset = dataset.dropna(
    axis="index", subset=["c_tot_ncs", "n_tot_ncs", "s_tot_ncs", "ec_12pre"]
)

# Split into train and test
train_dataset = dataset.sample(frac=config["train_split"], random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_conductivity = train_dataset["ec_12pre"]
test_conductivity = test_dataset["ec_12pre"]

train_spectrum = train_dataset.copy().iloc[:, SPECTRUM_START_COLUMN:]
test_spectrum = test_dataset.copy().iloc[:, SPECTRUM_START_COLUMN:]


if INCLUDE_EC:
    train_features = train_spectrum.join(train_conductivity)
    test_features = test_spectrum.join(test_conductivity)
else:
    train_features = train_spectrum
    test_features = test_spectrum

# print(train_features.shape)
# print(train_features["ec_12pre"].isna().sum())
# print(train_features["ec_12pre"].describe())

train_labels = train_dataset.copy()[["c_tot_ncs", "n_tot_ncs", "s_tot_ncs"]]
test_labels = test_dataset.copy()[["c_tot_ncs", "n_tot_ncs", "s_tot_ncs"]]

# Save for unnormalization later
original_test_label_std = test_labels.std()
original_test_label_mean = test_labels.mean()

# Normalize
train_labels = (train_labels - train_labels.mean()) / train_labels.std()
test_labels = (test_labels - test_labels.mean()) / test_labels.std()

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))


# history = model.fit(
#     train_features,
#     train_labels,
#     validation_split=0.2,
#     verbose=2,
#     epochs=config["epochs"],
#     callbacks=[WandbMetricsLogger(log_freq=10)],
# )


def get_optimizer(lr=1e-3, optimizer="adam"):
    "Select optmizer between adam and sgd with momentum"
    if optimizer.lower() == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if optimizer.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.1)


def train(model, batch_size=64, epochs=10, lr=1e-3, optimizer="adam", log_freq=10):

    # # Compile model like you usually do.
    # keras.backend.clear_session()
    # model.compile(
    #     loss="categorical_crossentropy",
    #     optimizer=get_optimizer(lr, optimizer),
    #     metrics=["accuracy"],
    # )

    # callback setup
    wandb_callbacks = [WandbMetricsLogger(log_freq=log_freq)]

    history = model.fit(
        train_features,
        train_labels,
        batch_size=batch_size,
        validation_data=(test_features, test_labels),
        verbose=2,
        epochs=epochs,
        callbacks=wandb_callbacks,
    )


with wandb.init(project="neospectra-mlp", name="With EC"):
    load = True
    if load:
        if INCLUDE_EC:
            model = saving.load_model("dnn_with_ec.keras")
        else:
            model = saving.load_model("dnn.keras")
    else:
        model = build_and_compile_model(config=config)
        model.build((None, train_features.shape[1]))
    model.summary()
    train(
        model,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        lr=config["learning_rate"],
    )

    test_labels_np = test_labels.to_numpy()
    mean_np = original_test_label_mean.to_numpy()
    std_np = original_test_label_std.to_numpy()

    test_predictions = model.predict(test_features)

    print("PREDICTIONS")
    print(test_predictions)
    print("TEST_LABELS")
    print(test_labels_np)

    print("MEAN")
    print(original_test_label_mean.to_numpy())
    print("STD")
    print(original_test_label_std.to_numpy())

    # Unnormalize
    test_predictions = test_predictions * mean_np + std_np
    print("PREDICTIONS UNNORMED")
    print(test_predictions)

    test_labels_unnormed = test_labels_np * mean_np + std_np

    error = np.subtract(test_predictions, test_labels_unnormed)
    mean_error = np.mean(error, axis=1)
    mean_error_no_outliers = np.copy(mean_error)
    mean_error_no_outliers[np.abs(mean_error_no_outliers) > 1.0] = (
        mean_error_no_outliers.mean()
    )

    preds = pd.DataFrame(
        data=test_predictions, columns=["carbon_pred", "nitrogen_pred", "sulfur_pred"]
    )
    labels = pd.DataFrame(
        data=test_labels_unnormed,
        columns=["carbon_true", "nitrogen_true", "sulfur_true"],
    )

    # hist = np.histogram(error, bins=25, range=(-2, 2))
    # wandb.log(wandb.Histogram(np_histogram=hist))

    df = pd.DataFrame(
        data={
            "mean_errors": mean_error,
            "mean_errors_no_outliers": mean_error_no_outliers,
        }
    )
    df = pd.concat([df, preds, labels], axis=1)

    print(df.head(n=10))

    table = wandb.Table(dataframe=df)
    wandb.log(
        {"error_hist": wandb.plot.histogram(table, "mean_errors", title="Mean errors")}
    )
    wandb.log(
        {
            "error_hist_no_outliers": wandb.plot.histogram(
                table, "mean_errors_no_outliers", title="Mean errors, no outliers"
            )
        }
    )

    wandb.log(
        {
            "label_prediction_plot_c": wandb.plot.scatter(
                table,
                "carbon_true",
                "carbon_pred",
                title="Labels vs Predictions, Carbon",
            )
        }
    )

    wandb.log(
        {
            "label_prediction_plot_n": wandb.plot.scatter(
                table,
                "nitrogen_true",
                "nitrogen_pred",
                title="Labels vs Predictions, Nitrogen",
            )
        }
    )

    wandb.log(
        {
            "label_prediction_plot_s": wandb.plot.scatter(
                table,
                "sulfur_true",
                "sulfur_pred",
                title="Labels vs Predictions, sulfur",
            )
        }
    )

    wandb.log({"eval_data": table})

    if INCLUDE_EC:
        model.save("dnn_with_ec.keras")
    else:
        model.save("dnn.keras")

    metric = keras.metrics.R2Score()
    metric.update_state(
        test_labels_unnormed.reshape((-1, 1)), test_predictions.reshape((-1, 1))
    )
    result = metric.result()
    print(f"R2: {result}")

    # Add a diagnoal 1/1 ground truth line
    # data = [[0, 0], [1, 1]]
    # table = wandb.Table(columns=["test_labels", "test_predictions"], data=data)
    # wandb.log(
    #     {
    #         "label_prediction_plot": wandb.plot.line(
    #             table,
    #             "test_labels",
    #             "test_predictions",
    #             title="Labels vs Predictions, Carbon",
    #         )
    #     }
    # )
