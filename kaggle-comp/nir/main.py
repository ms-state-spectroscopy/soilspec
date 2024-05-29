import csv

# Suppress annoying tensorflow warnings
import os
import pickle

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from analyzer import Analyzer
from keras import Model, layers, regularizers, saving
from matplotlib import pyplot as plt
from rf import RandomForestAnalyzer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from tensorflow import keras
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def root_mean_squared_error(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))


class RandomForestAnalyzer(Analyzer):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.model = RandomForestRegressor(n_jobs=10, verbose=verbose)

    def train(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.model.fit(X, Y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def getFeatureImportance(self, X, Y, n_repeats=10):

        result = permutation_importance(self.model, X, Y, n_repeats=10, n_jobs=10)

        print(result.importances_mean.shape)

        forest_importances = (
            pd.DataFrame(
                np.hstack(
                    (
                        result.importances_mean.reshape(-1, 1),
                        result.importances_std.reshape(-1, 1),
                    )
                ).reshape(-1, 2),
                index=list(X),
                columns=["mean", "std"],
            )
            .sort_values(by="mean")
            .tail(n=10)
        )
        print(forest_importances)
        plt.bar(
            forest_importances.index.to_list(),
            forest_importances["mean"],
            yerr=forest_importances["std"],
        )
        # forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
        # ax.set_title("Feature importances using permutation on full model")
        # ax.set_ylabel("Mean accuracy decrease")
        # fig.tight_layout()
        plt.title("Permutation importance in RF model, highest ten features")
        plt.show()


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
        l1_size=120,
        l2_size=120,
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
                layers.Dense(
                    l1_size,
                    activation=activation,
                ),
                layers.Dropout(0.2),
                layers.Dense(
                    l2_size,
                    activation=activation,
                ),
                layers.Dropout(0.2),
                layers.Dense(1),
            ]
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss="mean_squared_error",
            metrics=["mean_squared_error"],
        )

    def train(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.normalizer.adapt(np.array(X))
        history = self.model.fit(
            X,
            Y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=100)],
        )
        return history

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def hypertune(self, X_train, Y_train):

        self.X_train = X_train
        tuner = kt.Hyperband(
            self.buildModel,
            objective="val_loss",
            max_epochs=50,
            factor=3,
            directory="keras-tuner",
            project_name="neospectra",
        )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

        tuner.search(
            X_train, Y_train, epochs=50, validation_split=0.2, callbacks=[stop_early]
        )

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(
            f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('units_1')}/{best_hps.get('units_2')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}.
        """
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        self.model = tuner.hypermodel.build(best_hps)

    def buildModel(self, hp):

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        hp_units_1 = hp.Int("units_1", min_value=32, max_value=512, step=32)
        hp_units_2 = hp.Int("units_2", min_value=32, max_value=512, step=32)
        # hp_units_3 = hp.Int("units_3", min_value=32, max_value=512, step=32)
        activation = "relu"
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(self.X_train))

        model = keras.Sequential(
            [
                normalizer,
                layers.Dropout(0.3),
                layers.Dense(
                    hp_units_1,
                    activation=activation,
                    activity_regularizer=regularizers.L2(0.001),
                ),
                layers.Dropout(0.3),
                layers.Dense(
                    hp_units_2,
                    activation=activation,
                    activity_regularizer=regularizers.L2(0.001),
                ),
                layers.Dense(1),
            ]
        )

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss="mean_absolute_error",
            metrics=["mean_absolute_error"],
        )

        return model


def loadNeospectraData(train_split=0.75, pca_components=None):

    df = pd.read_csv(
        "neospectra_db/Neospectra_WoodwellKSSL_avg_soil+site+NIR.csv"
    ).set_index("kssl_id")

    if pca_components is not None:
        print(df.shape)
        pca = PCA(n_components=pca_components)
        X_new = pca.fit_transform(df.iloc[:, 54:])
        print(pca.explained_variance_ratio_)
        df = pd.DataFrame(X_new, index=df.index).join(df["eoc_tot_c"])
        print(df.shape)
    print(df)

    # Drop non-numeric columns
    dts = pd.Series(data=df.dtypes, name="dtypes")
    non_num = dts[dts == "object"].index.to_list()

    df = df.drop(non_num, axis=1)

    df = df.fillna(0)

    train_df = df.sample(frac=train_split, axis=0)
    test_df = df.drop(train_df.index)

    Y_train = train_df["eoc_tot_c"]
    Y_test = test_df["eoc_tot_c"]
    X_train = train_df.drop(["eoc_tot_c"], axis=1)
    # X_train = pd.DataFrame(data=X_new.loc[list(X_train.index)])
    X_test = test_df.drop(["eoc_tot_c"], axis=1)
    # X_test = pd.DataFrame(data=X_new.loc[list(X_test.index)])

    # train_df_cov = train_df_cov.reindex(sorted(train_df_cov.columns), axis=1)
    # train_df = (
    #     pd.read_csv("kaggle-comp/nir/train.csv")
    #     .set_index("sample_id")
    #     .join(train_df_cov)
    # )

    # train_dataset = train_df.sample(frac=train_split)

    # X_train = train_dataset.drop("soc_perc_log1p", axis=1)
    # Y_train = train_dataset["soc_perc_log1p"]

    # X_val_cov = pd.read_csv("kaggle-comp/nir/test_geocovariates.csv").set_index(
    #     "sample_id"
    # )

    # X_val_cov = X_val_cov.reindex(sorted(X_val_cov.columns), axis=1)

    # X_test = (
    #     pd.read_csv("kaggle-comp/nir/test.csv").set_index("sample_id").join(X_val_cov)
    # )

    print(X_train.shape, Y_train.shape)

    return X_train, Y_train, X_test, Y_test


def loadKaggleDataWithPCA(train_split=0.75, pca_components=None):
    X_train_cov = pd.read_csv("kaggle-comp/nir/train_geocovariates.csv").set_index(
        "sample_id"
    )
    train_df = (
        pd.read_csv("kaggle-comp/nir/train.csv")
        .set_index("sample_id")
        .join(X_train_cov)
    )
    X_val_cov = pd.read_csv("kaggle-comp/nir/test_geocovariates.csv").set_index(
        "sample_id"
    )
    val_df = (
        pd.read_csv("kaggle-comp/nir/test.csv").set_index("sample_id").join(X_val_cov)
    )

    X = pd.concat([train_df.iloc[:, 1:], val_df.iloc[:, 0:]], axis=0)

    if pca_components is not None:
        pca = PCA(n_components=pca_components)
        X_new = pca.fit_transform(X)
        print(pca.explained_variance_ratio_)
        X = pd.DataFrame(X_new, index=X.index)

    print(X.head())
    print(train_df.head())

    X_train = X.loc[train_df.index]
    Y_train = train_df["soc_perc_log1p"]
    X_val = X.loc[val_df.index]

    return X_train, Y_train, X_val


def loadKaggleData(train_split=0.75):

    train_df_cov = pd.read_csv("kaggle-comp/nir/train_geocovariates.csv").set_index(
        "sample_id"
    )

    train_df_cov = train_df_cov.reindex(sorted(train_df_cov.columns), axis=1)
    train_df = (
        pd.read_csv("kaggle-comp/nir/train.csv")
        .set_index("sample_id")
        .join(train_df_cov)
    )

    train_dataset = train_df.sample(frac=train_split)

    X_train = train_dataset.drop("soc_perc_log1p", axis=1)
    Y_train = train_dataset["soc_perc_log1p"]

    X_val_cov = pd.read_csv("kaggle-comp/nir/test_geocovariates.csv").set_index(
        "sample_id"
    )

    X_val_cov = X_val_cov.reindex(sorted(X_val_cov.columns), axis=1)

    X_test = (
        pd.read_csv("kaggle-comp/nir/test.csv").set_index("sample_id").join(X_val_cov)
    )

    print(X_train)
    print(Y_train)
    print(X_test)

    return X_train, Y_train, X_test


def loadCleanKaggleData(train_split=0.75):

    train_df_cov = pd.read_csv("kaggle-comp/nir/train_geocovariates.csv").set_index(
        "sample_id"
    )

    train_df_cov = train_df_cov.reindex(sorted(train_df_cov.columns), axis=1)
    train_df = (
        pd.read_csv("kaggle-comp/nir/train.csv")
        .set_index("sample_id")
        .join(train_df_cov)
    )

    train_dataset = train_df

    X_train = train_dataset.drop("soc_perc_log1p", axis=1)
    Y_train = train_dataset["soc_perc_log1p"]

    spectra: np.ndarray = X_train.to_numpy()[:, :601].copy()

    spectra -= spectra.min(axis=1)[:, np.newaxis]
    spectra /= spectra.max(axis=1)[:, np.newaxis]

    print(spectra.shape)
    X_train.iloc[:, :601] = spectra

    X_val_cov = pd.read_csv("kaggle-comp/nir/test_geocovariates.csv").set_index(
        "sample_id"
    )

    X_val_cov = X_val_cov.reindex(sorted(X_val_cov.columns), axis=1)

    X_val = (
        pd.read_csv("kaggle-comp/nir/test.csv").set_index("sample_id").join(X_val_cov)
    )

    spectra: np.ndarray = X_val.to_numpy()[:, :601].copy()
    spectra -= spectra.min(axis=1)[:, np.newaxis]
    spectra /= spectra.max(axis=1)[:, np.newaxis]
    X_val.iloc[:, :601] = spectra

    print(X_train.dtypes.tail(50))

    return X_train, Y_train, X_val


def loadAugmentedKaggleData(train_split=0.75):

    train_df_cov = pd.read_csv("kaggle-comp/nir/train_geocovariates.csv").set_index(
        "sample_id"
    )

    train_df_cov = train_df_cov.reindex(sorted(train_df_cov.columns), axis=1)
    train_df = (
        pd.read_csv("kaggle-comp/nir/train.csv")
        .set_index("sample_id")
        .join(train_df_cov)
    )

    train_df = train_df.reset_index()

    print(train_df.dtypes[1:602])

    train_dataset = train_df.sample(frac=train_split)

    X_train = train_dataset.drop("soc_perc_log1p", axis=1)
    Y_train = train_dataset["soc_perc_log1p"]

    X_val_cov = pd.read_csv("kaggle-comp/nir/test_geocovariates.csv").set_index(
        "sample_id"
    )

    X_val_cov = X_val_cov.reindex(sorted(X_val_cov.columns), axis=1)

    X_test = (
        pd.read_csv("kaggle-comp/nir/test.csv").set_index("sample_id").join(X_val_cov)
    )

    return X_train, Y_train, X_test


def plotLoss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 3])
    plt.xlabel("Epoch")
    plt.ylabel("Error [Estimated organic Carbon, % by weight]")
    plt.legend()
    plt.grid(True)
    plt.show()


def compareSpectra(X: pd.DataFrame, Y: pd.DataFrame):

    fig, (axLeft, axRight) = plt.subplots(1, 2)

    df = X.join(Y)
    leftGroup = df[(df["soc_perc_log1p"] >= 1.0) & (df["soc_perc_log1p"] <= 1.1)]
    rightGroup = df[(df["soc_perc_log1p"] >= 0.2) & (df["soc_perc_log1p"] <= 0.25)]

    X = np.array(list(leftGroup)[:601]).astype(float)
    Y = leftGroup.iloc[:, :601]
    axLeft.plot(X, Y.values[:5].T, "b-", label="1.0-1.1 nm")

    X = np.array(list(rightGroup)[:601]).astype(float)
    Y = rightGroup.iloc[:, :601]
    axLeft.plot(X, Y.values[:5].T, "g-", label="0.2-0.25 nm")

    axLeft.legend(loc="upper right")
    axLeft.set_title("Spectra within two ranges of organic carbon")

    # Normalize each spectrum from 0 to 1

    X = np.array(list(leftGroup)[:601]).astype(float)
    Y = leftGroup.iloc[:, :601].values.T
    Y -= Y.min(axis=0)
    Y /= Y.max(axis=0)
    print(Y)
    axRight.plot(X, Y[:, :5], "b-", linewidth=0.5)
    axRight.plot(X, Y.mean(axis=1), "b-", label="1.0-1.1 nm, Averaged", linewidth=5.0)

    X = np.array(list(rightGroup)[:601]).astype(float)
    Y = rightGroup.iloc[:, :601].values.T
    Y -= Y.min(axis=0)
    Y /= Y.max(axis=0)
    axRight.plot(X, Y[:, :5], "g-", linewidth=0.5)
    axRight.plot(X, Y.mean(axis=1), "g-", label="0.2-0.25 nm, Averaged", linewidth=5.0)

    axRight.legend(loc="upper right")
    axRight.set_title("Spectra after normalization")

    plt.show()


if __name__ == "__main__":
    LOAD_MLP = True
    # X_train, Y_train, X_test = loadAugmentedKaggleData()
    # X_train, Y_train, X_test = loadKaggleDataWithPCA(pca_components=30)
    # X_train, Y_train, X_test = loadCleanKaggleData()
    X_train, Y_train, X_test = loadKaggleData()

    # print(X_train.isnull().sum().head(50))
    # exit()
    # compareSpectra(X_train, Y_train)

    print(f"Training on {X_train.shape[0]} samples.")

    if LOAD_MLP:
        with open("mlp.pickle", "rb") as f:
            mlp_analyzer: MlpAnalyzer = pickle.load(f)
            # mlp_analyzer: RandomForestAnalyzer = pickle.load(f)
    else:
        mlp_analyzer = MlpAnalyzer(
            verbose=0, epochs=800, n_logits=1, lr=1e-4, l1_size=200, l2_size=200
        )
        print(mlp_analyzer.model.summary())

        # mlp_analyzer.hypertune(X_train, Y_train)
        history = mlp_analyzer.train(X_train, Y_train)

        # rf_analyzer.getFeatureImportance(X_train, Y_train)

        with open("mlp.pickle", "wb") as f:
            pickle.dump(mlp_analyzer, f)

    Y_pred_mlp = pd.DataFrame(
        mlp_analyzer.predict(X_test), columns=["soc_perc_log1p"], index=X_test.index
    )
    print(Y_pred_mlp)
    with open("submission.csv", "w") as f:
        Y_pred_mlp.to_csv(f, quoting=csv.QUOTE_NONNUMERIC)

    plotLoss(history)
