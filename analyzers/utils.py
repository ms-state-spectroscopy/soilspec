from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
import seaborn as sns
from matplotlib.axes import Axes
from torchmetrics.regression import R2Score
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import lightning as L
import random
from sklearn.metrics import r2_score


def rsquared(true, predicted):
    """Return R^2 where x and y are array-like."""

    r2 = r2_score(true, predicted)
    return r2


def describeAccuracy(Y_true: pd.DataFrame, Y_pred: pd.DataFrame, silent: bool = False):
    errors = pd.DataFrame(
        data=np.abs(Y_pred.values - Y_true.values), columns=list(Y_true)
    )

    print("=== Absolute Errors ===")
    print(errors.describe())

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(14.0, 4.0)
    axs: list[Axes]

    sns.histplot(data=errors, x=list(Y_true)[0], bins=25, ax=axs[0])
    axs[0].set_title("Absolute errors")

    errors_no_outliers = errors[(np.abs(zscore(errors)) < 3).all(axis=1)]

    Y_true_no_outliers = pd.DataFrame(
        Y_true.values[(np.abs(zscore(errors)) < 3).all(axis=1)], columns=Y_true.columns
    )
    Y_pred_no_outliers = pd.DataFrame(
        Y_pred.values[(np.abs(zscore(errors)) < 3).all(axis=1)], columns=Y_pred.columns
    )

    sns.histplot(data=errors_no_outliers, x=list(Y_true)[0], bins=25, ax=axs[1])
    axs[1].set_title("Absolute errors, no outliers")

    print(Y_true.tail())
    print(Y_pred.tail())

    pred_vs_true_df = Y_true_no_outliers.reset_index(drop=True).join(
        Y_pred_no_outliers.reset_index(drop=True),
        rsuffix="_pred",
        how="outer",
    )
    # pred_vs_true_df = pred_vs_true_df.join(errors, rsuffix="_err")
    print(pred_vs_true_df.isna().sum())
    print(pred_vs_true_df)

    for label_name in list(Y_true):
        g = sns.scatterplot(
            pred_vs_true_df,
            x=label_name,
            y=label_name + "_pred",
            ax=axs[2],
        )
    # g = sns.scatterplot(
    #     pred_vs_true_df,
    #     x="c_tot_ncs",
    #     y="c_tot_ncs_pred",
    #     ax=axs[2],
    # )
    # g = sns.scatterplot(
    #     pred_vs_true_df,
    #     x="n_tot_ncs",
    #     y="n_tot_ncs_pred",
    #     ax=axs[2],
    # )
    # g = sns.scatterplot(
    #     pred_vs_true_df,
    #     x="s_tot_ncs",
    #     y="s_tot_ncs_pred",
    #     ax=axs[2],
    # )
    x0, x1 = g.get_xlim()
    y0, y1 = g.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.plot(lims, lims, "-r")
    plt.legend(list(Y_true))

    if not silent:
        plt.show()

    # Calculate R2

    feature_names = list(Y_true)
    r2s = []

    for i in range(0, Y_true.shape[1]):
        # https://lightning.ai/docs/torchmetrics/stable/regression/r2_score.html
        metric = R2Score()
        result = metric(
            Y_true_no_outliers.values[:, i].reshape((-1, 1)),
            Y_pred_no_outliers.values[:, i].reshape((-1, 1)),
        )
        print(f"R2 for {feature_names[i]}: {result}")
        r2s.append(result.numpy())

    return r2s[0]  # return the final R2 value


def describeAccuracies(
    model_results: list[tuple[pd.DataFrame, pd.DataFrame]], model_labels: list[str]
):

    subplot_size = 4.0
    title_size = 12
    sns.set_palette("twilight")

    fig, axs = plt.subplots(
        len(model_results),
        3,
        layout="constrained",
        figsize=(subplot_size * 3, subplot_size * len(model_results)),
    )
    axs: np.ndarray[Axes]

    print(axs.shape)

    r2s = []

    for idx, result_pair in enumerate(model_results):

        Y_pred, Y_true = result_pair

        errors = pd.DataFrame(
            data=np.hstack(
                [
                    np.abs(Y_pred.values - Y_true.values),
                    np.mean(np.abs(Y_pred.values - Y_true.values), axis=1).reshape(
                        -1, 1
                    ),
                ]
            ),
            columns=list(Y_true) + ["mean_error"],
        )

        metric = R2Score()
        r2 = metric(Y_true.values.reshape((-1, 1)), Y_pred.values.reshape((-1, 1)))

        print("=== Absolute Errors ===")
        print(errors.describe())

        sns.histplot(data=errors, x="mean_error", bins=25, ax=axs[idx, 0])
        axs[idx, 0].set_ylabel(model_labels[idx], fontsize=title_size)

        errors_no_outliers = errors[(np.abs(zscore(errors)) < 3).all(axis=1)]

        sns.histplot(data=errors_no_outliers, x="mean_error", bins=25, ax=axs[idx, 1])

        pred_vs_true_df = Y_true.reset_index(drop=True).join(
            Y_pred.reset_index(drop=True), rsuffix="_pred", how="outer"
        )

        # Scatterplots
        # Three, one for each predicted feature
        g = sns.scatterplot(
            pred_vs_true_df,
            x="c_tot_ncs",
            y="c_tot_ncs_pred",
            ax=axs[idx, 2],
        )
        g = sns.scatterplot(
            pred_vs_true_df,
            x="n_tot_ncs",
            y="n_tot_ncs_pred",
            ax=axs[idx, 2],
        )
        g = sns.scatterplot(
            pred_vs_true_df,
            x="s_tot_ncs",
            y="s_tot_ncs_pred",
            ax=axs[idx, 2],
        )

        x0, x1 = g.get_xlim()
        y0, y1 = g.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        g.plot(lims, lims, "-r")
        ax: Axes = axs[idx, 2]
        ax.text(
            0.95,
            0.95,
            f"$R^2$={r2:.2f}",
            ha="right",
            va="top",
            backgroundcolor="white",
            transform=ax.transAxes,
        )
        plt.legend(["Carbon", "Nitrogen", "Sulphur"])

    axs[0, 0].set_title("Absolute errors", fontsize=title_size)
    axs[0, 1].set_title("Absolute errors, no outliers", fontsize=title_size)
    axs[0, 2].set_title("True versus predicted values", fontsize=title_size)

    plt.show()


def plotLoss(history):
    plt.plot(history["loss"], label="loss")
    plt.plot(history["val_loss"], label="val_loss")
    # plt.ylim([0, 3])
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()


def augmentSpectra(
    X: pd.DataFrame, Y: pd.DataFrame, reps: int, noise_std=1e-3, scale=1.5, plot=False
):

    augmented_Xs = []
    augmented_Ys = []

    for _ in range(reps):
        # Scale spectrum by some value between 0.8-1.2

        assert scale >= 1.0 and scale <= 2.0
        scaling_factor = random.uniform((2 - scale), 1.2)

        noise = np.random.normal(1, noise_std, X.shape)

        augmented_X = X * scaling_factor * noise

        # plotSpectraFromSet(X, indices=0, show=False)
        if plot:
            plotSpectraFromSet(augmented_X, indices=0, show=False)
        augmented_Xs.append(augmented_X)
        augmented_Ys.append(Y)

    X_augemented = np.asarray(augmented_Xs).reshape((-1, X.shape[1]))
    Y_augmented = np.asarray(augmented_Ys).reshape((-1, Y.shape[1]))

    if plot:
        plotSpectraFromSet(X_augemented, indices=0, show=True)

    return X_augemented, Y_augmented


def plotSpectraFromSet(df: pd.DataFrame, n=1, indices=None, show=True):
    """
    Plot spectra from a dataframe.
    NOTE: We assume that a column contains spectral data IFF the column name is a float,
    and that this column name is a wavelength in nm.

    Args:
        df (pd.DataFrame): The dataframe containing spectral values.
        n (int, optional): Number of spectra to plot. Defaults to 1.
    """

    rng = np.random.default_rng()

    # choose n spectra at random
    if indices is None:
        indices = rng.integers(0, df.shape[0], n)

    spectra_column_names = []
    wavelengths = []

    for col_name in list(df):
        try:
            col_name: str

            # For compatibility with the OSSL dataset,
            # where column names end with "_ref"
            if col_name.endswith("_ref"):
                wavelength = col_name.rsplit("_", 1)[0]
                wavelength = wavelength.split(".")[-1]
            else:
                wavelength = col_name

            # Attempt to convert the column name to a float.
            # If this works, then our column must represent a wavelength.
            wavelength = float(wavelength)
        except ValueError:
            # Otherwise, move on to the next column.
            print(f"{col_name} is not a wavelength.")
            continue
        spectra_column_names.append(col_name)
        wavelengths.append(wavelength)

    # NOTE: We assume that a column contains spectral data IFF the column name is a float,
    # and that this column name is a wavelength in nm.
    spectra = df.loc[:, spectra_column_names]
    X = np.array(wavelengths).astype(float)

    plt.plot(X, spectra.iloc[indices].T)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance (fraction)")
    plt.title(f"Reflectance curves for {n} randomly sampled spectra")
    # plt.ylim([0, 100])

    if show:
        plt.show()


def seedEverything():
    """Seeds pseudorandomness in PyTorch, Numpy, and Python's `random` module."""
    L.seed_everything(64)


class CustomDataset(Dataset):
    def __init__(self, X, Y):

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        self.X: pd.DataFrame = X
        self.Y: pd.DataFrame = Y

        assert isinstance(Y, np.ndarray)

    def __getitem__(self, index):
        # row = self.dataframe.iloc[index].to_numpy()
        features = self.X[index, :].astype(np.float32)
        label = self.Y[index, :].astype(np.float32)
        return features, label

    def __len__(self):
        return len(self.X)
