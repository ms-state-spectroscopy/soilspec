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
from analyzers.mlp import MlpAnalyzer
from analyzers.lightning_plain_mlp import LightningPlainMlpAnalyzer
from analyzers.rf import RandomForestAnalyzer
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()

# -db DATABASE -u USERNAME -p PASSWORD -size 20
parser.add_argument(
    "-s", "--save_analyzer", help="Save the Analyzer as a .pkl", action="store_true"
)
parser.add_argument("-l", "--load_analyzer", help="Load Analyzer from a .ckpt file")
parser.add_argument("-k", "--skip_training", help="Skip training", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":

    # 1. Load the data. This is any set of numerical features (X) and labels (Y),
    # structured as pandas DataFrames and divided into a training and test set.
    # (X_train, Y_train), (X_test, Y_test) = mississippi_db.loader.load(
    #     include_ec=True,
    #     labels=["sand_tot_psa", "clay_tot_psa", "silt_tot_psa"],
    # )

    physical_indicators = [
        # "cf_usda.c236_w.pct",
        "clay.tot_usda.a334_w.pct",
        "sand.tot_usda.c60_w.pct",
        "silt.tot_usda.c62_w.pct",
        "bd_usda.a4_g.cm3",
        "wr.1500kPa_usda.a417_w.pct",
        # "awc.33.1500kPa_usda.c80_w.frac",
    ]

    separate_dsets = {}
    head_weights = {}

    for label in physical_indicators:
        (
            (X_train, Y_train),
            (X_test, Y_test),
            original_label_mean,
            original_label_std,
        ) = ossl_db.loader.load(
            include_ec=True,
            labels=[label],
            from_pkl=True,
            normalize_Y=True,
        )

        separate_dsets[label] = (X_train, Y_train), (X_test, Y_test)
        head_weights[label] = None

    mississippi_labels = ["wilting_point", "field_capacity"]
    # mississippi_labels = ["sand_tot_psa", "clay_tot_psa"]
    # mississippi_labels = ["wilting_point"]
    # mississippi_labels = []

    (
        (X_train, Y_train),
        (X_test, Y_test),
        original_label_mean,
        original_label_std,
    ) = mississippi_db.loader.load(
        include_ec=False,
        include_depth=False,
        labels=mississippi_labels,
        normalize_Y=True,
        match_ossl_spectra=True,
    )

    dsets = {
        "train": utils.CustomDataset(X_train, Y_train),
        "test": utils.CustomDataset(X_test, Y_test),
    }

    # print(X_train)
    # print(X_train.isna().sum())

    # print(Y_train)
    # print(Y_train.isna().sum())

    # utils.plotSpectraFromSet(X_train, n=10)

    # 2. Instantiate an Analyzer.
    if args.load_analyzer:
        analyzer = LightningPlainMlpAnalyzer(
            checkpoint_path=args.load_analyzer,
            datasets=separate_dsets,
            labels=physical_indicators,
        )
    else:
        # 1 logit-- only one feature at a time
        # analyzer = LightningPlainMlpAnalyzer(
        #     dsets,
        #     mississippi_labels,
        #     lr=1e-4,
        #     n_logits=len(mississippi_labels),
        #     hidden_size=200,
        # )
        # analyzer = MlpAnalyzer(n_logits=2, lr=1e-4)
        analyzer = RandomForestAnalyzer()

    # 3. Train the Analyzer on the training data.

    if not args.skip_training:
        # analyzer.train()
        X_train = dsets["train"].X
        Y_train = dsets["train"].Y
        analyzer.train(X_train, Y_train)

    for label in mississippi_labels:
        X_test = dsets["test"].X
        Y_test = dsets["test"].Y
        # 4. Evaluate the Analyzer using the test set.
        Y_pred = analyzer.predict(X_test)

        if isinstance(Y_pred, np.ndarray):
            Y_pred = pd.DataFrame(
                data=Y_pred, index=X_test.index, columns=Y_train.columns
            )
        else:
            Y_pred = pd.DataFrame(
                data=Y_pred.detach().cpu(), index=X_test.index, columns=Y_train.columns
            )

        utils.describeAccuracy(Y_test, Y_pred)
