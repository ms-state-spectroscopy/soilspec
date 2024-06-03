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
import neospectra
import pickle
import utils
from analyzers.mlp import MlpAnalyzer

parser = argparse.ArgumentParser()

#-db DATABASE -u USERNAME -p PASSWORD -size 20
parser.add_argument("-s", "--save_analyzer", help="Save the Analyzer as a .pkl", action="store_true")
parser.add_argument("-l", "--load_analyzer", help="Load Analyzer from a .pkl", action="store_true")
parser.add_argument("-k", "--skip_training", help="Skip training", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":


    # 1. Load the data. This is any set of numerical features (X) and labels (Y),
    # structured as pandas DataFrames and divided into a training and test set.
    (X_train, Y_train), (X_test, Y_test) = mississippi_db.loader.load(
        include_ec=True,
        labels=[
            "CH_Ksat", "FH_Ksat"
        ],
    )

    

    print(Y_train)

    # utils.plotSpectraFromSet(X_train, n=10)

    # 2. Instantiate an Analyzer.
    if args.load_analyzer:
        with open('analyzer.pkl', 'rb') as f:
            analyzer = pickle.load(f)
    else:
        analyzer = MlpAnalyzer(n_logits=Y_train.shape[1], hidden_size=200, lr=1e-4)

    # 3. Train the Analyzer on the training data.
    if not args.skip_training:
        history = analyzer.train(X_train, Y_train, epochs=10000, early_stop_patience=500, batch_size=8)
        utils.plotLoss(history)

    # Save the model.
    if args.save_analyzer:
        with open('analyzer.pkl', 'wb') as f:
            pickle.dump(analyzer, f)

    # 4. Evaluate the Analyzer using the test set.
    Y_pred = analyzer.predict(X_test)
    Y_pred = pd.DataFrame(data=Y_pred, index=X_test.index, columns=Y_train.columns)

    utils.describeAccuracy(Y_test, Y_pred)
