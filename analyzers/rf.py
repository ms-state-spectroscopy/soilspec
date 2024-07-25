from analyzers.analyzer import Analyzer
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


class RandomForestAnalyzer(Analyzer):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.model = RandomForestRegressor(n_jobs=10, verbose=verbose)

    def train(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.model.fit(X, Y)

    def test(self, X, Y):
        r2 = self.model.score(X, Y)

        y_pred = self.model.predict(X)
        mse = np.mean((y_pred - Y) ** 2)

        rmse = np.sqrt(mse)
        # print(f"R2 is {r2}")
        return r2, rmse

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def getFeatureImportance(self, X, Y, n_repeats=10):

        result = permutation_importance(self.model, X, Y, n_repeats=10, n_jobs=10)

        print(result.importances_mean.shape)

        forest_importances = (
            pd.DataFrame(
                np.hstack(
                    (
                        result.importances_mean.reshape(-1, 1),
                        result.importances_std.reshape(-1, 1),
                    )
                ).reshape(-1, 2),
                index=list(X),
                columns=["mean", "std"],
            )
            .sort_values(by="mean")
            .tail(n=10)
        )
        print(forest_importances)
        plt.bar(
            forest_importances.index.to_list(),
            forest_importances["mean"],
            yerr=forest_importances["std"],
        )
        # forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
        # ax.set_title("Feature importances using permutation on full model")
        # ax.set_ylabel("Mean accuracy decrease")
        # fig.tight_layout()
        plt.title("Permutation importance in RF model, highest ten features")
        plt.show()
