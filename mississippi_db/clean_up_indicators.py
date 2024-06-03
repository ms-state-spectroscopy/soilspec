from pathlib import Path
import glob
from tqdm import tqdm
import pandas as pd
import specdal
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

messy = pd.read_csv("/home/main/soilspec/physical_indicators_dirty.csv")

# messy_names = (
#     pd.read_csv("/home/main/soilspec/messy_ids.csv").to_numpy().flatten().tolist()
# )
# print(messy_names)

# db_names = pd.read_csv("/home/main/soilspec/treatment_names.csv")["treatment"].unique()

# clean_names = []

for name in messy_names:
    name: str
    name = name.lower()
    name = name.replace(" #", "_")
    name = name.replace("#", "_")
    name = name.strip()
    name = name.replace(" ", "_")

    clean_names.append(name)

comparison = np.asarray([messy_names, clean_names]).T

comparison = pd.DataFrame(comparison, columns=["messy", "clean"]).sort_values("clean")
print(comparison)
print(db_names)
