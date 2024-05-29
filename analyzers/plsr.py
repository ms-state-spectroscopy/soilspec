from analyzers.analyzer import Analyzer
import pandas as pd
from sklearn.cross_decomposition import PLSRegression


class PlsrAnalyzer(Analyzer):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.model = PLSRegression(n_components=25)

    def train(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.model.fit(X, Y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
