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

    augmented_Xs.append(X)
    augmented_Ys.append(Y)

    X = X

    for _ in range(reps):
        # Scale spectrum by some value between 0.8-1.2

        assert scale > 0.0 and scale < 1.0
        scaling_factor = random.uniform(1 - scale, 1 + scale)
        # plt.plot(scale_line, label="Scale line")

        noise = np.random.normal(1, noise_std, X.shape)

        augmented_X = (X * scaling_factor) * noise

        # augmented_X = X.values

        # print(
        #     f"Augmenting by {scaling_factor}. Max went from {X.max()}->{augmented_X.max()}"
        # )

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
    include_unlabeled=True,
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
    if not include_unlabeled:
        dataset = dataset.dropna(axis="index", subset=labels)
    print(
        f"Dataset length went from {before_len} to {len(dataset)} after dropping nans from spectra"
    )

    # print(dataset.head(10))

    # # 1. Get list of unique sample IDs

    # sample_ids = dataset.reset_index()["sample_id"].unique()

    # for i in range(10):

    #     # 2. Pick a sample ID at random
    #     sample_id = np.random.choice(sample_ids)
    #     # 3. Return all matching rows (should be three)
    #     spectra = dataset.loc[sample_id, spectra_column_names].iloc[:2]
    #     if len(spectra) < 2:
    #         continue
    #     diff = spectra.values[0] - spectra.values[1]
    #     # print(spectra)
    #     # 4. Plot all three.
    #     spectra.T.plot()
    #     plt.plot(diff, label="Trial 1 - Trial 2")

    #     # # Generate a similar augmentation
    #     ampl = 0.04 * random.random()

    #     # normal period is 2pi ~= 6
    #     # should be 2000
    #     # x /= 2000?
    #     freq = random.random()
    #     shift = random.random() * 2 * np.pi

    #     first_sin = (
    #         np.sin(freq * np.linspace(0 + shift, 2 * np.pi + shift, spectra.shape[1]))
    #         * ampl
    #     )
    #     plt.plot(first_sin)

    #     total = first_sin

    #     for i in range(100):
    #         ampl = 0.0004 * random.random()

    #         # normal period is 2pi ~= 6
    #         # should be 2000
    #         # x /= 2000?
    #         freq = random.random() * 10
    #         shift = random.random() * 2 * np.pi
    #         second_sin = (
    #             np.sin(
    #                 freq * np.linspace(0 + shift, 2 * np.pi + shift, spectra.shape[1])
    #             )
    #             * ampl
    #         )
    #         # plt.plot(second_sin)

    #         total += second_sin

    #     # Finally, add Gaussian noise
    #     augmented_spectrum = spectra.values[0]
    #     noise = np.random.randn(augmented_spectrum.shape[0]) * 1e-4

    #     augmented_spectrum += total + noise

    #     plt.plot(augmented_spectrum)

    #     plt.plot(total, label="Augmentation")

    #     print(second_sin)

    #     plt.legend()
    #     plt.show()
    # exit()

    # Take average across three trials/scans
    # X = dataset.loc[:, spectra_column_names].groupby("sample_id").mean()
    # Y = dataset.loc[X.index, labels].groupby("sample_id").mean()

    X = dataset.loc[:, spectra_column_names]
    Y = dataset.loc[X.index, labels]
    # X = pd.concat([X, X_avg], axis="index")
    # Y = pd.concat([Y, Y_avg], axis="index")

    print(X)
    print(Y)

    # Save for unnormalization later
    original_label_min = Y.min()
    original_label_max = Y.max()

    # Y = Y.values
    if normalize_Y:
        # Normalize
        Y -= Y.min()
        Y /= Y.max()

    print(f"Y has {len(np.unique(Y))} unique vals / {len(Y)} total")
    print(f"X has {len(np.unique(X.iloc[:,100]))} unique vals / {len(X)} total")
    print(f"X has {len(np.unique(X.iloc[:,1]))} unique vals / {len(X)} total")
    print(f"X has {len(np.unique(X.iloc[:,1000]))} unique vals / {len(X)} total")

    # Random indices for splitting into train & test
    sample_ids = dataset.reset_index()["sample_id"].unique()
    print(sample_ids)
    print(sample_ids.shape)
    np.random.shuffle(sample_ids)

    stop_idx = int(sample_ids.shape[0] * train_split)

    print(Y)
    Y_train = Y.loc[sample_ids[:stop_idx], :].values
    X_train = X.loc[sample_ids[:stop_idx], :].values

    Y_test = Y.loc[sample_ids[stop_idx:], :].values
    X_test = X.loc[sample_ids[stop_idx:], :].values

    # print(indices)
    print(stop_idx)
    print(len(sample_ids))

    # print(
    #     np.hstack(
    #         (np.unique(Y_train).reshape((-1, 1)), np.unique(Y_test).reshape((-1, 1)))
    #     )
    # )

    print(
        f"There are {len(np.unique(np.concatenate((Y_train, Y_test))))} unique values across Y_train, Y_test"
    )
    print(
        f"Y_train {len(np.intersect1d(Y_train, Y_test))} values common values with Y_test"
    )

    print(
        f"There are about {len(np.unique(np.concatenate((X_train[:,1000], X_test[:,1000]))))} unique values across X_train, X_test"
    )
    print(
        f"X_train has about {len(np.intersect1d(X_train[:,1000], X_test[:,1000]))} common values with X_test"
    )

    # TODO: Split file using JSON format

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

    return (
        (X_train, Y_train),
        (X_test, Y_test),
        original_label_max,
        original_label_min,
    )
