"""
Steps:

1. Load the data. This is any set of numerical features (X) and labels (Y), structured as pandas DataFrames and divided into a training and test set.

2. Instantiate an Analyzer.

3. Train the Analyzer on the training data.

4. Evaluate the Analyzer using the test set.
"""

import argparse

import pandas as pd
from sklearn.decomposition import PCA
#from analyzers.cubist import CubistAnalyzer

from analyzers.rf import RandomForestAnalyzer
import mississippi_db
import mississippi_db.loader
import ossl_db.loader

# import neospectra
# import pickle
import analyzers.utils as utils

from analyzers.mlp import MlpAnalyzer
from tqdm import tqdm
import torch
import time
import numpy as np

from matplotlib import pyplot as plt

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme(style="ticks", palette="pastel")

parser = argparse.ArgumentParser()

# -db DATABASE -u USERNAME -p PASSWORD -size 20
parser.add_argument(
    "-s", "--save_analyzer", help="Save the Analyzer as a .pkl", action="store_true"
)
parser.add_argument("-l", "--load_analyzer", help="Load Analyzer from a .ckpt file")
parser.add_argument("-k", "--skip_training", help="Skip training", action="store_true")
parser.add_argument(
    "-p", "--pre_train", help="Pre-train on the OSSL dataset", action="store_true"
)
parser.add_argument("-r", "--seed", help="Seed to use", default=64)
args = parser.parse_args()


if __name__ == "__main__":

    utils.seedEverything(args.seed)

    # 1. Load the data. This is any set of numerical features (X) and labels (Y),
    # structured as pandas DataFrames and divided into a training and test set.

    # These are the labels that we should train/predict on
    # They should match the column names in the CSV file
    # mississippi_labels = ["field_capacity"]
    mississippi_labels = ["wilting_point"]

    ossl_labels = [
        # "cf_usda.c236_w.pct",
        # "oc_usda.c729_w.pct",
        "clay.tot_usda.a334_w.pct",
        "sand.tot_usda.c60_w.pct",
        "silt.tot_usda.c62_w.pct",
        # "bd_usda.a4_g.cm3",
        "wr.1500kPa_usda.a417_w.pct",
        # "awc.33.1500kPa_usda.c80_w.frac",
    ]

    # Each sample in the dataset will be augmented this many times
    n_dataset_augmentations = 0

    (
        (X_train, Y_train),
        (X_test, Y_test),
        original_label_mean,
        original_label_std,
    ) = mississippi_db.loader.load(
        labels=mississippi_labels,
        normalize_Y=True,
        from_pkl=False,
        train_split=50 / 75,
        take_grad=False,
        n_components=60,
        include_unlabeled=False,
    )

    # exit()

    # Select only original, non-augmented test values
    print(f"Y_test has {len(np.unique(Y_test))} unique vals")

    # X_train = (X_train - X_train.min(axis=1).reshape(-1, 1)) / X_train.max(
    #     axis=1
    # ).reshape(-1, 1)

    print(X_train[0].shape)
    # print(X_train)
    # for spectrum in X_train[:100]:
    #     # print(spectrum)
    #     norm = (spectrum - spectrum.min()) / spectrum.max()
    #     plt.plot(range(len(spectrum)), norm, c="blue")
    # plt.show()

    # exit()

    if args.pre_train:
        (
            (X_train, Y_train),
            (X_test, Y_test),
            original_label_mean,
            original_label_std,
        ) = ossl_db.loader.load(
            labels=ossl_labels,
            normalize_Y=True,
            from_pkl=False,
            include_unlabeled=False,
            take_grad=False,
            n_components=60,
        )

    print(
        f"Y_train has {len(Y_train)-np.isnan(Y_train).sum().sum()} non-null values ({(len(Y_train)-np.isnan(Y_train).sum().sum())/len(Y_train)*100})"
    )
    print(
        f"Y_test has {len(Y_test)-np.isnan(Y_test).sum().sum()} non-null values, {len(Y_test)} total values"
    )

    print(
        original_label_mean,
        original_label_std,
    )

    # TODO: K-fold cross val
    analyzer = MlpAnalyzer(
        output_size=len(ossl_labels if args.pre_train else mississippi_labels),
        lr=1e-4,
        hidden_size=200,
        batch_size=128,
        input_size=X_train.shape[1],
        checkpoint_path=args.load_analyzer,
        n_augmentations=0,
    )

    # for i, spectrum in enumerate(X_train[:5]):
    #     plt.plot(range(len(spectrum)), spectrum, label=str(Y_train[i]))

    # for i, spectrum in enumerate(X_test[:5]):
    #     plt.plot(range(len(spectrum)), spectrum, label=str(Y_train[i]))

    plt.ylabel("PCA component magnitude")
    plt.ylabel("PCA component index")

    plt.legend()
    plt.title("Subsample of train and test features")
    plt.show()

    # analyzer = CubistAnalyzer()
    # analyzer = RandomForestAnalyzer()''

    print(f"X_train has shape {X_train.shape}")
    print(f"Y_test has shape {Y_test.shape}")

    # print(len(np.intersect1d(X_test[:, 1], X_train[:, 1])))
    # print(len(np.intersect1d(Y_test, Y_train)))
    # print(len(np.unique(np.concatenate((Y_train, Y_test)))))

    # assert (
    #     len(np.intersect1d(Y_train, Y_test)) == 0
    # ) or args.pre_train, f"Y_train {len(np.intersect1d(Y_train, Y_test))} values common values with Y_test"

    if not args.pre_train:
        ax = plt.subplot()
        ax.hist(Y_train, color="blue", alpha=0.5, label="Training data")
        ax.set_ylabel("Train label counts")

        ax2 = ax.twinx()
        ax2.hist(Y_test, alpha=0.5, color="red")
        ax2.set_ylabel("Test label counts")
        ax.set_title("Distribution of training and test labels")
        plt.show()

    if not args.skip_training:
        analyzer.train(X_train, Y_train)

    analyzer.test(X_test, Y_test)
