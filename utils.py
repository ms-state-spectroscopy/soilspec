from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
import seaborn as sns
from matplotlib.axes import Axes
from tensorflow import keras


def describeAccuracy(Y_true: pd.DataFrame, Y_pred: pd.DataFrame):
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

    sns.histplot(data=errors_no_outliers, x=list(Y_true)[0], bins=25, ax=axs[1])
    axs[1].set_title("Absolute errors, no outliers")

    print(Y_true.tail())
    print(Y_pred.tail())

    pred_vs_true_df = Y_true.reset_index(drop=True).join(
        Y_pred.reset_index(drop=True), rsuffix="_pred", how="outer"
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

    plt.show()

    # Calculate R2

    feature_names = list(Y_true)
    for i in range(0, Y_true.shape[1]):
        metric = keras.metrics.R2Score()
        metric.update_state(
            Y_true.values[:, i].reshape((-1, 1)), Y_pred.values[:, i].reshape((-1, 1))
        )
        result = metric.result()
        print(f"R2 for {feature_names[i]}: {result}")


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

        metric = keras.metrics.R2Score()
        metric.update_state(
            Y_true.values.reshape((-1, 1)), Y_pred.values.reshape((-1, 1))
        )
        r2 = metric.result()

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


def plotSpectraFromSet(df: pd.DataFrame, n=1):
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
    indices = rng.integers(0, df.shape[0], n)

    spectra_column_names = []

    for col_name in list(df):
        try:
            float(col_name)
        except ValueError:
            continue
        spectra_column_names.append(col_name)

    # NOTE: We assume that a column contains spectral data IFF the column name is a float,
    # and that this column name is a wavelength in nm.
    spectra = df.loc[:, spectra_column_names]
    X = np.array(spectra_column_names).astype(float)

    plt.plot(X, spectra.iloc[indices].T)
    plt.ylim([0, 100])
    plt.show()
