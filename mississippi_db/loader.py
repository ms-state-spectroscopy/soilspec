import pandas as pd
import numpy as np
from halo import Halo
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import random


SPECTRUM_START_COLUMN = 16


def augmentSpectra(
    X: pd.DataFrame, Y: pd.DataFrame, reps: int, noise_std=1e-3, scale=0.5, plot=False
):

    augmented_Xs = []
    augmented_Ys = []

    augmented_Xs.append(X.values)
    augmented_Ys.append(Y)

    X = X.values

    for _ in range(reps):
        # Scale spectrum by some value between 0.8-1.2

        assert scale > 0.0 and scale < 1.0
        scaling_factor = random.uniform(1 - scale, 1 + scale)
        # plt.plot(scale_line, label="Scale line")

        noise = np.random.normal(1, noise_std, X.shape)

        augmented_X = (X * scaling_factor) * noise

        # augmented_X = X.values

        print(
            f"Augmenting by {scaling_factor}. Max went from {X.max()}->{augmented_X.max()}"
        )

        if plot:
            plt.plot(augmented_X[0, :], label="X*")
            plt.plot(X[0, :], label="X")
            plt.legend()
            plt.show()
        # plotSpectraFromSet(X, indices=0, show=False)
        augmented_Xs.append(augmented_X)
        augmented_Ys.append(Y)

    X_augemented = np.asarray(augmented_Xs).reshape((-1, X.shape[1]))
    Y_augmented = np.asarray(augmented_Ys).reshape((-1, Y.shape[1]))

    return X_augemented, Y_augmented


def getPicklePath(labels):
    base = "mississippi_db/ms_"

    for label in labels:
        assert isinstance(label, str)
        base += f"{label}_"

    base += ".pkl"
    return base


def load(
    labels: list[str],
    train_split=0.75,
    normalize_Y=False,
    match_ossl_spectra=True,
    from_pkl=False,
    take_grad=True,
    n_components=None,
    n_augmentations=0,
    seed=64,
) -> tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:

    if from_pkl:
        try:
            dataset = pd.read_pickle(getPicklePath(labels))
        except FileNotFoundError as e:
            print(f"{getPicklePath(labels)} not found. Falling back to csv.")
            return load(
                labels=labels,
                train_split=train_split,
                normalize_Y=normalize_Y,
                match_ossl_spectra=match_ossl_spectra,
                from_pkl=False,
            )

    else:
        dataset = pd.read_csv("mississippi_db/mississippi_db.csv").set_index(
            ["sample_id", "trial"]
        )

        # Drop samples without spectra
        dataset = dataset.dropna(axis="index", subset=["path"])

        # print(f"MISSISSIPPI DATASET INCLUDES THE FOLLOWING:")
        # print(list(dataset))

        # Drop NaNs for labels
        original_len = len(dataset)

        print(f"Dataset length went from {original_len} to {len(dataset)}")

        # if include_ec:
        #     dataset = dataset.dropna(axis="index", subset=["ec_12pre"])
        with Halo(text="Saving to " + getPicklePath(labels)):
            dataset.to_pickle(getPicklePath(labels))

    spectra_column_names = []
    for col_name in list(dataset):
        try:
            wavelength = int(col_name)
            if match_ossl_spectra:
                if wavelength >= 400 and wavelength % 2 == 0:
                    spectra_column_names.append(col_name)
            else:
                spectra_column_names.append(col_name)
        except ValueError:
            continue

    # dataset.to_csv(f"{labels[0]}.csv")

    before_len = len(dataset)
    dataset = dataset.dropna(axis="index", subset=spectra_column_names)
    print(
        f"Dataset length went from {before_len} to {len(dataset)} after dropping nans from spectra"
    )

    # Save for unnormalization later
    if normalize_Y:
        Y = dataset.loc[:, labels]
        original_label_std = Y.std()
        original_label_mean = Y.mean()

        # Normalize
        dataset.loc[:, labels] = (Y - Y.mean()) / Y.std()

        dataset = dataset.dropna(axis="index", subset=labels)

    # Split into train and test

    X = dataset.loc[:, spectra_column_names]
    Y = dataset.loc[:, labels]

    X, Y = augmentSpectra(X, Y, reps=n_augmentations)

    # Random indices for splitting into train & test
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    print(indices)

    stop_idx = int(X.shape[0] * train_split)

    X_train = X[indices[:stop_idx], :]
    Y_train = Y[indices[:stop_idx], :]
    X_test = X[indices[stop_idx:], :]
    Y_test = Y[indices[stop_idx:], :]

    # test_dataset = dataset.sample(frac=1 - train_split, random_state=seed)

    # train_dataset = dataset.drop(test_dataset.index)

    # # NOTE: We assume that a column contains spectral data IFF the column name is a float,
    # # and that this column name is a wavelength in nm.
    # X_train = train_dataset.loc[:, spectra_column_names]
    # X_test = test_dataset.loc[:, spectra_column_names]

    if take_grad == True:
        # Convert the data to use the gradient
        X_train = np.gradient(X_train, axis=1)

        X_train = (X_train - X_train.min(axis=1).reshape(-1, 1)) / X_train.max(
            axis=1
        ).reshape(-1, 1)

    # Y_train = train_dataset.loc[:, labels]
    # Y_test = test_dataset.loc[:, labels]
    if n_components is not None:
        pca = PCA(n_components=n_components)
        pca.fit(np.vstack((X_train, X_test)))

        # X_train, Y_train = augmentSpectra(X_train, Y_train, reps=n_augmentations)
        # X_test, Y_test = augmentSpectra(X_test, Y_test, reps=n_augmentations)

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    if normalize_Y:
        return (
            (X_train, Y_train),
            (X_test, Y_test),
            original_label_mean,
            original_label_std,
        )
    else:
        return (X_train, Y_train), (X_test, Y_test)
