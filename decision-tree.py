import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import neospectra
from sklearn.metrics import mean_absolute_error


def getGini(group: pd.DataFrame) -> float:
    """Calculate the Gini coefficient, which represents the likelihood that a randomly picked datum would be erroneously classified.

    Lower is better.

    Args:
        group (pd.DataFrame)

    Returns:
        float: the Gini coefficient
    """
    sum = 0

    classes = group["class"].unique()
    total_size = group.shape[0]

    for class_id in classes:
        prob = len(group[group["class"] == class_id]) / total_size
        sum += prob**2

    return 1 - sum


def plot(df: pd.DataFrame):
    class_0 = df[df["class"] == 0]
    class_1 = df[df["class"] == 1]
    plt.scatter(class_0["x1"], class_0["x2"], c="red")
    plt.scatter(class_1["x1"], class_1["x2"], c="blue")
    plt.title("Dataset")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.gca().set_aspect("equal", "box")
    plt.show()


def findBestSplit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    best_gini = 1.0
    best_x1_split = None
    best_left = None

    for feature in ["x1", "x2"]:
        for index, row in df.iterrows():
            x1_split = row[feature]
            left = df[df[feature] < x1_split]
            if left.shape[0] == 0:
                continue
            gini = getGini(left)
            if (
                gini < best_gini
                or gini == best_gini
                and left.shape[0] > best_left.shape[0]
            ):
                best_gini = gini
                best_x1_split = x1_split
                best_left = left

        for index, row in df.iterrows():
            x1_split = row[feature]
            left = df[df[feature] > x1_split]
            if left.shape[0] == 0:
                continue
            gini = getGini(left)
            if (
                gini < best_gini
                or gini == best_gini
                and left.shape[0] > best_left.shape[0]
            ):
                best_gini = gini
                best_x1_split = x1_split
                best_left = left

    print(f"Best Gini: {best_gini} @ x1={best_x1_split}")
    plt.vlines([best_x1_split], -2, 2)

    return df[df["x1"] < best_x1_split], df[df["x1"] >= best_x1_split]


# Generate data in a circle
a_radius = 1.5
bounds = 2

rng = np.random.default_rng()
data = rng.uniform(-bounds, bounds, size=(100, 3))
dataset = pd.DataFrame(
    data,
    columns=["x1", "x2", "class"],
)


dataset["class"] = np.linalg.norm(data[:, :2], axis=1) < a_radius
dataset["class"] = dataset["class"].astype(int)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

(X_train, Y_train), (X_test, Y_test) = neospectra.load()
regr = RandomForestClassifier(max_depth=2)
regr.fit(X_train,Y_train)

Y_pred = regr.fit(X_test)

error = mean_absolute_error(Y_test, Y_pred)

print(f"Mean abs. error: {error}")
plot(dataset)

print(getGini(dataset))
