import pandas as pd
import numpy as np

SPECTRUM_START_COLUMN = 16


def load(
    labels: list[str],
    include_ec=True,
    include_depth=True,
    train_split=0.75,
    normalize_Y=False,
    match_ossl_spectra=True,
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
    dataset = pd.read_csv("mississippi_db/mississippi_db.csv").set_index(
        ["sample_id", "trial"]
    )

    # print(f"MISSISSIPPI DATASET INCLUDES THE FOLLOWING:")
    # print(list(dataset))

    # Drop NaNs for labels
    dataset = dataset.dropna(axis="index", subset=labels)

    # if include_ec:
    #     dataset = dataset.dropna(axis="index", subset=["ec_12pre"])
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

    # Split into train and test
    train_dataset = dataset.sample(frac=train_split, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # NOTE: We assume that a column contains spectral data IFF the column name is a float,
    # and that this column name is a wavelength in nm.
    X_train = train_dataset.loc[:, spectra_column_names]
    X_test = test_dataset.loc[:, spectra_column_names]

    if include_depth:

        X_train = X_train.join(
            train_dataset.loc[:, ["lay.depth.to.bottom", "lay.depth.to.top"]]
        )
        X_test = X_test.join(
            test_dataset.loc[:, ["lay.depth.to.bottom", "lay.depth.to.top"]]
        )

    Y_train = train_dataset.loc[:, labels]
    Y_test = test_dataset.loc[:, labels]

    if normalize_Y:
        return (
            (X_train, Y_train),
            (X_test, Y_test),
            original_label_mean,
            original_label_std,
        )
    else:
        return (X_train, Y_train), (X_test, Y_test)
