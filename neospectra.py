import pandas as pd

SPECTRUM_START_COLUMN = 55


def load(
    include_ec=True, train_split=0.75, normalize_Y=False
) -> tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    dataset = pd.read_csv("neospectra_db/Neospectra_WoodwellKSSL_avg_soil+site+NIR.csv")

    # Drop NaNs for labels
    dataset = dataset.dropna(
        axis="index", subset=["c_tot_ncs", "n_tot_ncs", "s_tot_ncs"]
    )

    if include_ec:
        dataset = dataset.dropna(axis="index", subset=["ec_12pre"])

    # Split into train and test
    train_dataset = dataset.sample(frac=train_split, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_conductivity = train_dataset["ec_12pre"]
    test_conductivity = test_dataset["ec_12pre"]

    train_spectrum = train_dataset.copy().iloc[:, SPECTRUM_START_COLUMN:]

    print(f"The training data has {train_spectrum.shape[1]} cols")

    test_spectrum = test_dataset.copy().iloc[:, SPECTRUM_START_COLUMN:]

    if include_ec:
        X_train = train_spectrum.join(train_conductivity)
        X_test = test_spectrum.join(test_conductivity)
    else:
        X_train = train_spectrum
        X_test = test_spectrum
        

    Y_train = train_dataset.copy()[["c_tot_ncs", "n_tot_ncs", "s_tot_ncs"]]
    Y_test = test_dataset.copy()[["c_tot_ncs", "n_tot_ncs", "s_tot_ncs"]]

    # Save for unnormalization later
    original_test_label_std = Y_test.std()
    original_test_label_mean = Y_test.mean()

    # Normalize
    if normalize_Y:
        Y_train = (Y_train - Y_train.mean()) / Y_train.std()
        Y_test = (Y_test - Y_test.mean()) / Y_test.std()

    return (X_train, Y_train), (X_test, Y_test)
