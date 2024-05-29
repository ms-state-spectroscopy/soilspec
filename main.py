from analyzers.rf import RandomForestAnalyzer
from analyzers.plsr import PlsrAnalyzer
from analyzers.mlp import MlpAnalyzer
from matplotlib import pyplot as plt
import neospectra
import pandas as pd
import pickle
from scipy.stats import zscore
import utils

LOAD_RF = True
LOAD_MLP = True

(X_train, Y_train), (X_test, Y_test) = neospectra.load()

# analyzer = RandomForestAnalyzer(verbose=2)
plsr_analyzer = PlsrAnalyzer(verbose=2)
plsr_analyzer.train(X_train, Y_train)
Y_pred_plsr = pd.DataFrame(plsr_analyzer.predict(X_test), columns=list(Y_test))

if LOAD_RF:
    with open("rf.pickle", "rb") as f:
        rf_analyzer: RandomForestAnalyzer = pickle.load(f)
else:
    rf_analyzer = RandomForestAnalyzer(verbose=2)
    rf_analyzer.train(X_train, Y_train)

    with open("rf.pickle", "wb") as f:
        pickle.dump(rf_analyzer, f)

Y_pred_rf = pd.DataFrame(rf_analyzer.predict(X_test), columns=list(Y_test))
rf_analyzer.getFeatureImportance(X_test, Y_test)
exit()

if LOAD_MLP:
    with open("mlp.pickle", "rb") as f:
        mlp_analyzer: MlpAnalyzer = pickle.load(f)
else:
    mlp_analyzer = MlpAnalyzer(2, epochs=500)
    mlp_analyzer.train(X_train, Y_train)

    with open("mlp.pickle", "wb") as f:
        pickle.dump(mlp_analyzer, f)

Y_pred_mlp = pd.DataFrame(mlp_analyzer.predict(X_test), columns=list(Y_test))

# utils.describeAccuracies([(Y_test, Y_pred_plsr)])
utils.describeAccuracies(
    [(Y_test, Y_pred_plsr), (Y_test, Y_pred_rf), (Y_test, Y_pred_mlp)],
    ["PLSR", "Random Forest", "MLP"],
)
