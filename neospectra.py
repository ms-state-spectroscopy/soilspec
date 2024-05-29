import pandas as pd

SPECTRUM_START_COLUMN = 55


def load(
    include_ec=True, train_split=0.75
) -> tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    dataset = pd.read_csv("neospectra_db/Neospectra_WoodwellKSSL_avg_soil+site+NIR.csv")

    # Drop NaNs for labels
    dataset = dataset.dropna(
        axis="index", subset=["c_tot_ncs", "n_tot_ncs", "s_tot_ncs", "ec_12pre"]
    )

    # Split into train and test
    train_dataset = dataset.sample(frac=train_split, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_conductivity = train_dataset["ec_12pre"]
    test_conductivity = test_dataset["ec_12pre"]

    train_spectrum = train_dataset.copy().iloc[:, SPECTRUM_START_COLUMN:]
    test_spectrum = test_dataset.copy().iloc[:, SPECTRUM_START_COLUMN:]

    if include_ec:
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

    return (train_features, train_labels), (test_features, test_labels)
