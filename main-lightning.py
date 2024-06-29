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
from analyzers.lightning_mlp import LightningMlpAnalyzer
from tqdm import tqdm
import torch
import time

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

        separate_dsets[label] = {
            "train": utils.CustomDataset(X_train, Y_train),
            "test": utils.CustomDataset(X_test, Y_test),
        }

    # 2. Instantiate an Analyzer.
    if args.load_analyzer:
        analyzer = LightningMlpAnalyzer(
            checkpoint_path=args.load_analyzer,
            datasets=separate_dsets,
            labels=physical_indicators,
        )
    else:
        # 1 logit-- only one feature at a time
        analyzer = LightningMlpAnalyzer(
            datasets=separate_dsets,
            labels=physical_indicators,
            n_logits=1,
            hidden_size=200,
            lr=1e-4,
            input_size=X_train.shape[1],
            # logdir=time.strftime("%Y_%m_%d-%H_%M"),
        )

    # 3. Train the Analyzer on the training data.

    if not args.skip_training:
        # num_loops = 100
        # total_iters = num_loops * len(physical_indicators)
        # pbar = tqdm(total=total_iters)
        # for i in range(num_loops):
        #     print(f"ITER {i+1}/{num_loops}")
        #     for label in physical_indicators:

        #         print(f"Training on {label}")

        analyzer.train()

    for label in physical_indicators:
        (X_train, Y_train), (X_test, Y_test) = separate_dsets[label]
        if head_weights[label] is not None:
            analyzer.setHeadWeights(head_weights[label])
        # 4. Evaluate the Analyzer using the test set.
        Y_pred = analyzer.predict(X_test)
        Y_pred = pd.DataFrame(data=Y_pred, index=X_test.index, columns=Y_train.columns)

        utils.describeAccuracy(Y_test, Y_pred)
