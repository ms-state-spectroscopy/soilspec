from sklearn.ensemble import RandomForestRegressor


class Analyzer:
    def __init__(self, verbose: int = 0) -> None:
        pass

    def train(self, X, Y):
        raise NotImplementedError


class RandomForestAnalyzer(Analyzer):
    def __init__(self) -> None:
        super().__init__()
        self.regressor = RandomForestRegressor(verbose=2, n_jobs=10)

    def train(self, X, Y):
        self.regressor.fit(X, Y)

    def test(self, X, Y):
        return self.regressor.score(X, Y)

    def predict(self, X):
        return self.regressor.predict(X)


class PlsrAnalyzer(Analyzer):
    def __init__(self) -> None:
        super().__init__()
