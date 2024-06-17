from pathlib import Path
import glob
from tqdm import tqdm
import pandas as pd
import specdal
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import re  # regex
import json

dataset_root = "/home/main/ms_dataset/processed_xlsx"
show_warnings = True

num_errors = 0
total_files = 0
rows = []

with open("/home/main/soilspec/human_to_permanent_treatments.json", "r") as f:
    human_to_permanent_treatments = json.load(f)

print(human_to_permanent_treatments)

frames = []

for path_str in tqdm(glob.glob(dataset_root + "/**/*.xlsx", recursive=True)):
    total_files += 1

    # print(path_str)

    relative_path = path_str.split("/", 4)[-1]
    try:
        xlsx_df = pd.read_excel(path_str)

        xlsx_df = xlsx_df.replace(human_to_permanent_treatments)

        frames.append(xlsx_df)

    except Exception as e:
        if show_warnings:
            print(f"{e}")
        num_errors += 1
        continue

asd_csv = pd.read_csv("ms_database.csv")
# frames.append(asd_csv)

df = pd.concat(frames)
df = df.replace("", np.nan)
cols = df.columns.tolist()
df = df.groupby(
    ["treatment", "lay.depth.to.top", "lay.depth.to.bottom"], as_index=False
)[cols].first()

df["sample_id"] = (
    df["treatment"].astype(str)
    + "_"
    + df["lay.depth.to.top"].astype(int).astype(str)
    + "-"
    + df["lay.depth.to.bottom"].astype(int).astype(str)
    + "cm"
)

asd_csv["sample_id"] = (
    asd_csv["treatment"].astype(str)
    + "_"
    + asd_csv["lay.depth.to.top"].astype(int).astype(str)
    + "-"
    + asd_csv["lay.depth.to.bottom"].astype(int).astype(str)
    + "cm"
)

df.set_index("sample_id", inplace=True)
asd_csv.set_index("sample_id", inplace=True)

print(df)
print(asd_csv)

df3 = (
    pd.merge(
        asd_csv,
        df,
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("", "_y"),
    )
    .reset_index()
    .drop_duplicates(subset=["sample_id", "trial"])
    .set_index("sample_id")
)

df3.drop(df3.filter(regex="_y$").columns, axis=1, inplace=True)
df3.drop(df3.filter(regex="_x$").columns, axis=1, inplace=True)

print(asd_csv)
print(df)
print(df3)
df3.to_csv("merged_xlsx.csv")
