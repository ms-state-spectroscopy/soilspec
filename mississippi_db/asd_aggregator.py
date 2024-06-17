from pathlib import Path
import glob
from tqdm import tqdm
import pandas as pd
import specdal
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import re  # regex


dataset_root = "/home/main/ms_dataset/"

show_warnings = True

num_errors = 0
total_files = 0
rows = []


def rsplitNonAlphanumeric(input: str):
    parts = re.split(r"[^a-zA-Z0-9\s]", input)

    end = parts[-1]

    front = input[: -len(end) - 1]

    # print(f"Input: {input}, front: {front}, end: {end}")

    return front, end


def parseFileName(file_name: str) -> tuple[str, int, int, int]:
    # 1. Get the repition number (each sample is scanned three times)
    try:
        remaining, trial = file_name.rsplit("_", 1)
        trial = int(trial)
    except Exception as e:
        raise Exception(f"Could not get trial from end of file '{file_name}': {e}")

    # 2. Get the sample depth, top and bottom
    try:
        remaining, depth_to_bottom = remaining.rsplit("-", 1)
        depth_to_bottom = int(depth_to_bottom.replace("cm", ""))
        remaining, depth_to_top = rsplitNonAlphanumeric(remaining)
        depth_to_top = int(depth_to_top)
    except Exception as e:
        raise Exception(
            f"{file_name}: Could not get depth data from '{file_name}': {e}"
        )

    sample_id = remaining

    if depth_to_top > depth_to_bottom or depth_to_top > 100 or depth_to_bottom > 100:
        raise Exception(f"{file_name}: Could not get depth data from '{file_name}'")

    return sample_id, trial, depth_to_top, depth_to_bottom


for path_str in tqdm(glob.glob(dataset_root + "/**/*.asd", recursive=True)):
    total_files += 1

    print(path_str)

    relative_path = path_str.split("/", 4)[-1]
    try:
        file_name = path_str.split("/")[-1].split(".")[-2]
        treatment, trial, depth_to_top, depth_to_bottom = parseFileName(file_name)
        # sample_id, depth, trial = file_name.rsplit("_", 2)
    except Exception as e:
        if show_warnings:
            print(f"{e}")
        num_errors += 1
        continue

    treatment = treatment.replace("nature3", "nature_3")
    treatment = treatment.replace("native5", "native_5")
    treatment = treatment.replace("natie5", "native_5")
    treatment = treatment.replace("row", "r")

    sample_id = path_str[:-6]

    rows.append(
        [
            relative_path,
            file_name,
            treatment,
            sample_id,
            trial,
            depth_to_top,
            depth_to_bottom,
        ]
        # + reflectance.to_list()
    )

df = pd.DataFrame(
    data=rows,
    columns=[
        "path",
        "file",
        "treatment",
        "sample_id",
        "trial",
        "lay.depth.to.top",
        "lay.depth.to.bottom",
    ],
    # + reflectance.index.astype(int).to_list(),
).sort_values(by=["sample_id", "lay.depth.to.top", "lay.depth.to.bottom", "trial"])

# DROP ALL SAMPLES WITH MISSING VALUES
len_before = len(df)
df.replace("", np.nan, inplace=True)
df.dropna(inplace=True)
len_after = len(df)
print(
    f"Dropped {len_before-len_after} samples ({100-(len_after/len_before)*100:.2f}%) for having missing values"
)

df.loc[:, ["file", "lay.depth.to.top", "lay.depth.to.bottom", "trial"]].to_csv(
    "partial_db.csv"
)

# DROP ALL SAMPLES WITHOUT EXACTLY THREE SCANS


len_before = len(df)
sample_id_counts = df["sample_id"].value_counts()
sample_ids_without_three_reps = sample_id_counts[sample_id_counts != 3].index

print(sample_ids_without_three_reps)

df = df.set_index("sample_id").drop(sample_ids_without_three_reps, axis="index")
len_after = len(df)
print(
    f"Dropped {len_before-len_after} samples ({100-(len_after/len_before)*100:.2f}%) for not having three trials"
)
df.set_index("path", inplace=True)
df.index.rename("sample_id", inplace=True)

print(df)
print(df.describe())
print("Writing to CSV...")
df.to_csv("ms_database.csv")
df.index.to_series().to_csv("sample_ids.csv")

print(f"Error rate: {num_errors/total_files}")
