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

        xlsx_df = (
            xlsx_df.replace(human_to_permanent_treatments)
            .set_index(["treatment", "lay.depth.to.top", "lay.depth.to.bottom"])
            .sort_index()
        )

        print(xlsx_df)

        frames.append(xlsx_df)

    except Exception as e:
        if show_warnings:
            print(f"{e}")
        num_errors += 1
        continue

df = pd.concat(frames).sort_index().reset_index()
df = df.replace("", np.nan)
cols = df.columns.tolist()
df = (
    df.groupby(
        ["treatment", "lay.depth.to.top", "lay.depth.to.bottom"], as_index=False
    )[cols]
    .first()
    .set_index(["treatment", "lay.depth.to.top", "lay.depth.to.bottom"])
)

print(df)

df.to_csv("merged_xslx.csv")
