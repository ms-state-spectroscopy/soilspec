"""
Steps:

1. Load the data. This is any set of numerical features (X) and labels (Y), structured as pandas DataFrames and divided into a training and test set.

2. Instantiate an Analyzer.

3. Train the Analyzer on the training data.

4. Evaluate the Analyzer using the test set.
"""

import neospectra
import utils

if __name__ == "__main__":

    # 1. Load the data. This is any set of numerical features (X) and labels (Y),
    # structured as pandas DataFrames and divided into a training and test set.
    (X_train, Y_train), (Y_test, Y_test) = neospectra.load(include_ec=True)

    print(f"The training data has {X_train.shape[1]} cols")

    utils.plotSpectraFromSet(X_train, n=10)

    # 4. Evaluate the Analyzer using the test set.
    utils.describeAccuracy(Y_test, Y_pred)
