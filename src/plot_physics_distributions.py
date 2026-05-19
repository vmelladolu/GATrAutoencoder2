import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================================
# PATHS
# ==========================================================

PRED_PATH = Path(
    "/media/FQM378/vmellado/GATrEnv/GATrAutoencoder/testbeam_predictions.csv"
)

H5_PATH = Path(
    "/media/FQM378/vmellado/GATrEnv/data/clasificador/50k_e-_Emin1GeV_Emax120GeV_continuous_fixed_position_5-5--20_sigmaMomentum_0.1_train.h5"
)

OUTDIR = Path(
    "/media/FQM378/vmellado/GATrEnv/GATrAutoencoder/physics_plots"
)

OUTDIR.mkdir(exist_ok=True)


# ==========================================================
# LOAD PREDICTIONS
# ==========================================================

preds = pd.read_csv(PRED_PATH)


# ==========================================================
# LOAD H5
# ==========================================================

f = h5py.File(H5_PATH, "r")

all_event_ids = f["event_id"][:]
all_i = f["i"][:]
all_j = f["j"][:]
all_k = f["k"][:]
all_energy = f["energy"][:]


# ==========================================================
# STORAGE
# ==========================================================

n_hits = []
longitudinal_width = []
lateral_width = []
total_energy = []
pred_labels = []

# ==========================================================
# LOOP EVENTS
# ==========================================================

for _, row in preds.iterrows():

    event_id = int(row["event_id"])

    pred = row["prediction"]

    mask = (all_event_ids == event_id)

    i_hits = all_i[mask]
    j_hits = all_j[mask]
    k_hits = all_k[mask]
    e_hits = all_energy[mask]

    if len(e_hits) == 0:
        continue

 # ======================================================
    # N HITS
    # ======================================================

    n_hits.append(len(e_hits))


    # ======================================================
    # TOTAL ENERGY
    # ======================================================

    total_energy.append(np.sum(e_hits))


    # ======================================================
    # LONGITUDINAL PROFILE
    # ======================================================

    z_mean = np.average(k_hits, weights=e_hits)

    z_var = np.average((k_hits - z_mean) ** 2, weights=e_hits)

    longitudinal_width.append(np.sqrt(z_var))


    # ======================================================
    # LATERAL PROFILE
    # ======================================================

    x_mean = np.average(i_hits, weights=e_hits)
    y_mean = np.average(j_hits, weights=e_hits)

    r2 = (i_hits - x_mean) ** 2 + (j_hits - y_mean) ** 2

    r_var = np.average(r2, weights=e_hits)

    lateral_width.append(np.sqrt(r_var))


    pred_labels.append(pred)


# ==========================================================
# FINAL DF
# ==========================================================

obs = pd.DataFrame({
    "prediction": pred_labels,
    "n_hits": n_hits,
    "longitudinal": longitudinal_width,
    "lateral": lateral_width,
    "energy": total_energy,
})


# ==========================================================
# PLOTS
# ==========================================================

variables = [
    "n_hits",
    "longitudinal",
    "lateral",
    "energy",
]

for var in variables:

    plt.figure(figsize=(7,5))

    for label in sorted(obs["prediction"].unique()):

        vals = obs[obs["prediction"] == label][var]

        plt.hist(
            vals,
            bins=40,
            alpha=0.5,
            density=True,
            label=str(label)
        )

    plt.xlabel(var)

    plt.ylabel("Density")

    plt.legend()

    plt.tight_layout()

    plt.savefig(OUTDIR / f"{var}.png", dpi=200)

    plt.close()


print("Done")
