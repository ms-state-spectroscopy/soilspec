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

human_to_permanent_treatments = {}

for path_str in tqdm(glob.glob(dataset_root + "/**/*.xlsx", recursive=True)):
    total_files += 1

    # print(path_str)

    xlsx_name = path_str.rsplit("/")[-1].split(".")[0]
    try:
        df = pd.read_excel(path_str).set_index("treatment")

        treatment_names = df.index.astype(str).to_list()

        for name in treatment_names:
            human_name = name
            print(name)
            name: str
            name = name.strip()
            name = name.replace(" row", "-r")
            name = name.replace(" ", "-")
            name = name.replace(" #", "-")
            name = name.replace("#", "-")
            name = name.replace("--", "-")
            name = name.replace("gh-", "gh")
            name = name.replace("field-", "field")
            name = name.replace("native-", "native")
            name = name.lower()

            human_to_permanent_treatments[human_name] = name

        print(xlsx_name)

    except Exception as e:
        if show_warnings:
            print(f"{e}")
        num_errors += 1
        continue

with open("human_to_permanent_treatments.json", "w") as f:
    json.dump(human_to_permanent_treatments, f)
