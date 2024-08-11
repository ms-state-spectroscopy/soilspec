from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from scipy.interpolate import interp1d

SPECTRA_START_COL = 55
rng = np.random.default_rng()

cols = pd.read_csv("neospectra_db/Neospectra_database_column_names.csv")

df = pd.read_csv("neospectra_db/Neospectra_WoodwellKSSL_avg_soil+site+NIR.csv")

ids = df.iloc[:, 0].astype(int)
spectra = pd.concat(
    [ids, df.iloc[:, SPECTRA_START_COL:].astype(float)], axis=1
).set_index("kssl_id")
labels = df.iloc[:, :SPECTRA_START_COL]


def plot_spectrum(idx: int, ax: Axes, interp=False):
    spectrum = spectra.iloc[idx]
    sample_labels = labels.iloc[idx]
    # print(sample_labels)
    print(f"Estimated organic carbon, weight percent: {sample_labels['eoc_tot_c']}")
    print(f"Carbon, total, weight percent: {sample_labels['c_tot_ncs']}")
    print(f"Nitrogen, total, weight percent: {sample_labels['n_tot_ncs']}")
    print(f"Sulphur, total, weight percent: {sample_labels['s_tot_ncs']}")
    print(f"Clay, weight percent: {sample_labels['clay_tot_psa']}")
    print(f"Silt, weight percent: {sample_labels['silt_tot_psa']}")
    print(f"Sand, weight percent: {sample_labels['sand_tot_psa']}")
    X = spectrum.index.astype(float)
    Y = spectrum.values

    if interp:
        f = interp1d(X, Y)
        X = np.linspace(X.min(),X.max(),512)
        Y = f(X)

    ax.scatter(X, Y, label=spectrum.name)


# Pick a random spectrum


fig, ax = plt.subplots()
ax: Axes  # type annotation
ax.set_ylim([0, 100])
ax.set_xlim([1350, 2550])
ax.set_title(f"Sample spectra")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance (%)")

for i in range(10):
    random_idx = rng.integers(0, spectra.shape[0])
    plot_spectrum(random_idx, ax, interp=True)

plt.legend()

plt.show()
print(labels['c_tot_ncs'].min())
print(labels['c_tot_ncs'].max())

# Now try with interpolation

