"""
Steps:

1. Load the data. This is any set of numerical features (X) and labels (Y), structured as pandas DataFrames and divided into a training and test set.

2. Instantiate an Analyzer.

3. Train the Analyzer on the training data.

4. Evaluate the Analyzer using the test set.
"""

import argparse

import pandas as pd
import mississippi_db
import mississippi_db.loader
import ossl_db.loader
import neospectra
import pickle
import analyzers.utils as utils

# from analyzers.mlp import MlpAnalyzer
from analyzers.lightning_mlp import LightningMlpAnalyzer
from analyzers.mlp import MlpAnalyzer
from analyzers.temporal_ensemble import TemporalEnsembleAnalyzer
from analyzers.pi_mlp import PiMlpAnalyzer
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
args = parser.parse_args()


if __name__ == "__main__":

    utils.seedEverything()

    # 1. Load the data. This is any set of numerical features (X) and labels (Y),
    # structured as pandas DataFrames and divided into a training and test set.

    ossl_labels = [
        # "cf_usda.c236_w.pct",
        "oc_usda.c729_w.pct",
        # "clay.tot_usda.a334_w.pct",
        # "sand.tot_usda.c60_w.pct",
        # "silt.tot_usda.c62_w.pct",
        # "bd_usda.a4_g.cm3",
        # "wr.1500kPa_usda.a417_w.pct",
        # "awc.33.1500kPa_usda.c80_w.frac",
    ]

    # mississippi_labels = ["wilting_point", "field_capacity"]
    mississippi_labels = ["wilting_point"]
    # mississippi_labels = []

    (
        (X_train, Y_train),
        (X_test, Y_test),
        original_label_mean,
        original_label_std,
    ) = mississippi_db.loader.load(
        labels=mississippi_labels,
        normalize_Y=True,
        from_pkl=False,
        include_unlabeled=False,
        train_split=125 / 225,
        include_depth=False,
    )

    # (
    #     (X_train, Y_train),
    #     (X_test, Y_test),
    #     original_label_mean,
    #     original_label_std,
    # ) = ossl_db.loader.load(
    #     labels=ossl_labels, normalize_Y=True, from_pkl=True, include_unlabeled=True
    # )

    print(
        f"Y_train has {len(Y_train)-Y_train.isna().sum().sum()} non-null values ({(len(Y_train)-Y_train.isna().sum().sum())/len(Y_train)*100})"
    )
    print(
        f"Y_test has {len(Y_test)-Y_test.isna().sum().sum()} non-null values, {len(Y_test)} total values"
    )

    print(Y_train.describe())
    print(Y_test.describe())

    # Flippity floppity
    # X_train, X_test = X_test, X_train
    # Y_train, Y_test = Y_test, Y_train

    # X_train, Y_train = utils.augmentSpectra(X_train, Y_train, reps=500)
    # X_test, Y_test = utils.augmentSpectra(X_test, Y_test, reps=50)

    # utils.plotSpectraFromSet(X_train, n=30)


    analyzer = MlpAnalyzer(
        output_size=1,
        lr=1e-4,
        hidden_size=200,
        batch_size=128,
        input_size=X_train.shape[1],
        checkpoint_path=args.load_analyzer,
        n_augmentations=0,
    )
    if not args.skip_training:
        analyzer.train(X_train, Y_train)

    analyzer.test(X_test, Y_test)
