from analyzers.analyzer import Analyzer
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from cubist import Cubist
from halo import Halo


class CubistAnalyzer(Analyzer):
    def __init__(
        self, verbose: int = 0, n_components: int = 120, n_committees=10, neighbors=9
    ) -> None:
        super().__init__(verbose=verbose)
        self.model = Cubist(
            verbose=verbose,
            n_committees=n_committees,
            neighbors=neighbors,
            unbiased=False,
        )
        # self.pca = PCA(n_components=n_components)

    def train(self, X: pd.DataFrame, Y: pd.DataFrame):
        with Halo(f"Fitting Cubist model to {len(Y)} samples"):
            # self.pca.fit(X)
            # print(
            #     f"[Cubist] PCA has the following EVR:\n {self.pca.explained_variance_ratio_} SVs."
            # )
            # X_reduced = self.pca.transform(X)
            self.model.fit(X, Y)

    def predict(self, X: pd.DataFrame):
        # X_reduced = self.pca.transform(X)
        return self.model.predict(X)

    def test(self, X: pd.DataFrame, Y: pd.DataFrame) -> float:
        with Halo(f"Scoring Cubist model on {len(Y)} samples"):
            # X_reduced = self.pca.transform(X)
            r2 = self.model.score(X, Y)

        print(f"The R2 of the Cubist model is {r2:.3f}")
        return r2
