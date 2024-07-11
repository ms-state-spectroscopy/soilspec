import pandas as pd
from halo import Halo
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import numpy as np


SPECTRUM_START_COLUMN = 16


def getPicklePath(labels):
    base = "/home/main/ossl/ossl_"

    for label in labels:
        assert isinstance(label, str)
        base += f"{label}_"

    base += ".pkl"
    return base


def load(
    labels: list[str],
    include_ec=True,
    include_depth=True,
    train_split=0.75,
    normalize_Y=False,
    from_pkl=False,
    include_unlabeled=True,
    n_components=None,
) -> tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    """Load the averaged NIR samples from the Neospectra dataset

    Args:
        labels (list[str]): The labels that we wish to include in Y_train and Y_test
        include_ec (bool, optional): Include soil electroconductivity as a feature. Defaults to True.
        train_split (float, optional): Split between training and test set. Defaults to 0.75.
        normalize_Y (bool, optional): Whether to normalize the labels. Defaults to False.

    Returns:
        tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]: _description_
    """

    if from_pkl:
        dataset = pd.read_pickle(getPicklePath(labels))

    else:
        with Halo(text="Reading scan CSV"):
            visnir_df = pd.read_csv(
                "/home/main/ossl/ossl_visnir_L0_v1.2.csv"
            ).set_index("id.layer_uuid_txt")

            print(f"visnir_df has shape {visnir_df.shape}")

        with Halo(text="Reading lab0 CSV"):
            lab0_df = pd.read_csv("/home/main/ossl/ossl_soillab_L0_v1.2.csv").set_index(
                "id.layer_uuid_txt"
            )

        with Halo(text="Reading lab1 CSV"):
            lab1_df = pd.read_csv("/home/main/ossl/ossl_soillab_L1_v1.2.csv").set_index(
                "id.layer_uuid_txt"
            )

        lab_df = pd.concat([lab0_df, lab1_df])

        # lab_df.rename(
        #     {
        #         "clay.tot_usda.a334_w.pct": "clay_tot_psa",
        #         "sand.tot_usda.c60_w.pct": "sand_tot_psa",
        #         "silt.tot_usda.c62_w.pct": "silt_tot_psa",
        #         "ph.h2o_usda.a268_index": "ph_h2o",
        #         "ec_usda.a364_ds.m": "ec_12pre",
        #         "bd_usda.a4_g.cm3": "db_13b",
        #     },
        #     inplace=True,
        #     axis="columns",
        # )

        visnir_df = visnir_df.dropna(subset="scan_visnir.2492_ref", axis="index")

        # visnir_df.to_csv("/home/main/ossl/ossl_visnir_L0_v1.2_no_nan.csv")

        print("Joining visnir and lab dfs.")
        dataset = visnir_df.join(lab_df, rsuffix="_lab")

        dataset.columns.to_series().to_csv("ossl_db/colum_names.csv")

        # Drop NaNs for labels
        for label in labels:

            print(dataset.loc[:, label])
            dataset.dropna(axis="index", subset=[label], inplace=True)

            print(f"DATASET HAS {len(dataset)} SAMPLES after dropping {label}")

        non_nans = len(dataset) - dataset.isna().sum()
        non_nans.to_csv("ossl_db/non-nan_counts.csv")

        print("Saving to " + getPicklePath(labels))
        with Halo(text="Saving to " + getPicklePath(labels)):
            dataset.to_pickle(getPicklePath(labels))

    spectra_column_names = []
    for col_name in list(dataset):
        col_name: str
        if (
            col_name.endswith("_ref")
            and int(col_name.split(".")[-1].split("_")[0]) >= 400
        ):
            spectra_column_names.append(col_name)

    dataset.loc[:, spectra_column_names].isna().sum().to_csv(
        "ossl_db/spectrum_nan_counts.csv"
    )
    dataset = dataset.dropna(axis="index", subset=spectra_column_names)

    Y = dataset.loc[:, labels]

    original_label_max = Y.max()
    original_label_min = Y.min()

    # Save for unnormalization later
    if normalize_Y:
        Y = dataset.loc[:, labels]
        original_label_std = Y.std()
        original_label_mean = Y.mean()

        # Normalize
        dataset.loc[:, labels] = (Y - Y.mean()) / Y.std()

        if include_unlabeled:
            labeled_dataset = dataset.dropna(axis="index", subset=labels)
        else:
            dataset = dataset.dropna(axis="index", subset=labels)

    # If we're including unlabeled data, we only want labeled data in the test set
    if include_unlabeled:
        test_dataset = labeled_dataset.sample(frac=1 - train_split, random_state=64)
    else:
        test_dataset = dataset.sample(frac=1 - train_split, random_state=64)
    train_dataset = dataset.drop(test_dataset.index)

    # NOTE: We assume that a column contains spectral data IFF the column name is a float,
    # and that this column name is a wavelength in nm.
    X_train = train_dataset.loc[:, spectra_column_names]
    X_test = test_dataset.loc[:, spectra_column_names]

    if n_components is not None:
        pca = PCA(n_components=n_components)
        pca.fit(np.vstack((X_train.values, X_test.values)))

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
    # if include_depth:

    #     X_train = X_train.join(
    #         train_dataset.loc[:, ["lay.depth.to.bottom", "lay.depth.to.top"]]
    #     )
    #     X_test = X_test.join(
    #         test_dataset.loc[:, ["lay.depth.to.bottom", "lay.depth.to.top"]]
    #     )

    Y_train = train_dataset.loc[:, labels]
    Y_test = test_dataset.loc[:, labels]

    return (
        (X_train, Y_train),
        (X_test, Y_test),
        original_label_min,
        original_label_max,
    )
