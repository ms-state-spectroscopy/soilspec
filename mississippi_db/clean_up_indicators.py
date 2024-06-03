from pathlib import Path
import glob
from tqdm import tqdm
import pandas as pd
import specdal
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import difflib

messy = pd.read_csv("/home/main/soilspec/physical_indicators_dirty.csv")
spectra = pd.read_csv("/home/main/soilspec/partial_db.csv")

# Clean up the messy treatment IDs
messy["treatment"] = messy["treatment"].str.lower()
messy["treatment"] = messy["treatment"].str.strip()
messy["treatment"] = messy["treatment"].str.replace(" #", "_")
messy["treatment"] = messy["treatment"].str.replace("-", "_")
messy["treatment"] = messy["treatment"].str.replace(" ", "_")
messy["treatment"] = messy["treatment"].str.replace("#", "_")
messy["treatment"] = messy["treatment"].str.replace("__", "_")
messy["treatment"] = messy["treatment"].str.replace("row", "r")
messy["treatment"] = messy["treatment"].str.replace("field_", "field")


messy_treatments = messy["treatment"]
spectra_treatments = spectra["treatment"]

all_treatments = pd.concat([messy_treatments, spectra_treatments])

print(all_treatments.value_counts().sort_index().tail(n=40))

messy = messy.set_index("treatment")

print(spectra)
print(messy)

messy_names = messy.index.unique().to_list()
print(messy_names)

bd = []
fc = []
pwp = []
awc = []
total_porosity = []

for i in range(spectra.shape[0]):
    # for i in range(10, 100, 10):
    row = spectra.iloc[i]
    treatment = row["treatment"]
    try:
        print("=====================")
        print(row)
        physical_indicators = messy.loc[treatment]

        # Match to correct depth reading
        physical_indicators = physical_indicators[
            (physical_indicators["lay.depth.to.top"] == row["lay.depth.to.top"])
            & (
                physical_indicators["lay.depth.to.bottom"].astype(int)
                == row["lay.depth.to.bottom"].astype(int)
            )
        ]

        if physical_indicators.shape[0] > 0:
            physical_indicators = physical_indicators.iloc[0, 0:5]
            print(physical_indicators)
            bd.append(physical_indicators["BD"])
            fc.append(physical_indicators["FC"])
            pwp.append(physical_indicators["FC"])
            awc.append(physical_indicators["AWC"])
            total_porosity.append(physical_indicators["Total porosity"])
        else:
            bd.append(None)
            fc.append(None)
            pwp.append(None)
            awc.append(None)
            total_porosity.append(None)

    except KeyError as e:
        closest = difflib.get_close_matches(treatment, messy_names, n=1)
        print(f"{e} not found. Closest: {closest}")
        bd.append(None)
        fc.append(None)
        pwp.append(None)
        awc.append(None)
        total_porosity.append(None)


bd = pd.Series(bd, name="bd", index=spectra.index)
fc = pd.Series(fc, name="fc", index=spectra.index)
pwp = pd.Series(pwp, name="pwp", index=spectra.index)
awc = pd.Series(awc, name="awc", index=spectra.index)
total_porosity = pd.Series(total_porosity, name="total_porosity", index=spectra.index)

spectra = spectra.join(bd)
spectra = spectra.join(fc)
spectra = spectra.join(pwp)
spectra = spectra.join(awc)
spectra = spectra.join(total_porosity)

print(spectra)
spectra.to_csv("spectra_with_physical_indicators.csv")
