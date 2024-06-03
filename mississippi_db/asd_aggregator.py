from pathlib import Path
import glob
from tqdm import tqdm
import pandas as pd
import specdal
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


dataset_root = "/home/main/"

show_warnings = False

num_errors = 0
total_files = 0
rows = []
for path_str in tqdm(glob.glob(dataset_root + "/**/*.asd", recursive=True)):
    total_files += 1

    relative_path = path_str.split("/", 4)[-1]
    try:
        file_name = path_str.split("/")[-1].split(".")[-2]
        treatment, depth, repitition = file_name.rsplit("_", 2)
    except Exception as e:
        if show_warnings:
            print(f"Invalid: {file_name}. {e}")
        num_errors += 1
        continue

    try:
        depth_to_top, depth_to_bottom = depth.split("-")
    except Exception as e:
        if show_warnings:
            print(f"Invalid: {depth}. {e}")
        num_errors += 1
        continue

    spectrum_data, metadata = specdal.reader.read_asd(path_str)

    reflectance = pd.Series(
        spectrum_data["tgt_count"] / spectrum_data["ref_count"], name="reflectance"
    ).astype(np.float16)

    treatment = treatment.replace("nature3", "nature_3")
    treatment = treatment.replace("native5", "native_5")
    treatment = treatment.replace("natie5", "native_5")
    treatment = treatment.replace("row", "r")

    rows.append(
        [relative_path, treatment, repitition, depth_to_top, depth_to_bottom[:-2]]
        + reflectance.to_list()
    )

df = (
    pd.DataFrame(
        data=rows,
        columns=[
            "file",
            "treatment",
            "repitition",
            "lay.depth.to.top",
            "lay.depth.to.bottom",
        ]
        + reflectance.index.astype(int).to_list(),
    )
    .set_index("file")
    .sort_values(
        by=["treatment", "lay.depth.to.top", "lay.depth.to.bottom", "repitition"]
    )
)

df.loc[
    :, ["treatment", "lay.depth.to.top", "lay.depth.to.bottom", "repitition"]
].to_csv("partial_db.csv")

print(df)
print("Writing to Excel...")
df.to_excel("ms_database.xlsx")

print(f"Error rate: {num_errors/total_files}")
