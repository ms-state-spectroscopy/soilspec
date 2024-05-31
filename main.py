"""
Steps:

1. Load the data. This is any set of numerical features (X) and labels (Y), structured as pandas DataFrames and divided into a training and test set.

2. Instantiate an Analyzer.

3. Train the Analyzer on the training data.

4. Evaluate the Analyzer using the test set.
"""

import pandas as pd
import mississippi_db
import mississippi_db.loader
import neospectra
import utils
from analyzers.mlp import MlpAnalyzer

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
    analyzer = MlpAnalyzer(n_logits=Y_train.shape[1], hidden_size=200, lr=1e-4)

    # 3. Train the Analyzer on the training data.
    history = analyzer.train(X_train, Y_train, epochs=10000, early_stop_patience=500)
    utils.plotLoss(history)

    # 4. Evaluate the Analyzer using the test set.
    Y_pred = analyzer.predict(X_test)
    Y_pred = pd.DataFrame(data=Y_pred, index=X_test.index, columns=Y_train.columns)

    utils.describeAccuracy(Y_test, Y_pred)
