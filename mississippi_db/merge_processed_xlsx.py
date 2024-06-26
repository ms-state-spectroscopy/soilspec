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

for path_str in tqdm(glob.glob(dataset_root + "/**/**.xlsx", recursive=True)):
    total_files += 1

    # print(path_str)

    relative_path = path_str.split("/", 4)[-1]
    try:
        xlsx_df = pd.read_excel(path_str)

        xlsx_df = xlsx_df.replace(human_to_permanent_treatments)

        xlsx_df["sample_id"] = (
            xlsx_df["treatment"].astype(str)
            + "_"
            + xlsx_df["lay.depth.to.top"].astype(int).astype(str)
            + "-"
            + xlsx_df["lay.depth.to.bottom"].astype(int).astype(str)
            + "cm"
        )

        if not "permanent" in path_str:
            xlsx_df.set_index("sample_id").sort_index().to_excel(
                path_str.rsplit(".")[-2] + "_permanent_ids.xlsx"
            )

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

print(df)
print(asd_csv)

df.set_index("sample_id", inplace=True)
asd_csv.set_index("sample_id", inplace=True)
merged_df = (
    pd.merge(
        asd_csv,
        df,
        how="outer",
        left_index=True,
        right_index=True,
        suffixes=("_x", None),
    )
    .reset_index()
    .drop_duplicates(subset=["sample_id", "trial"])
    .set_index("sample_id")
)

# merged_df = (
#     pd.concat([asd_csv, df], join="outer", ignore_index=True)
#     .groupby("sample_id")
#     .first()
# )

# merged_df = merged_df.dropna(axis=0, subset=["clay_tot_psa"])

nonnan = merged_df.shape[0] - merged_df.isna().sum()

print(nonnan)
nonnan.to_excel("nonnan.xlsx")

# nonnan


# print(merged_df.groupby("group").count())

# print("Writing to CSV")
# merged_df.to_csv("mississippi_db.csv")

# print("Writing to Excel")
# merged_df.to_excel("merged.xlsx")
