import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

# ==========================================================
# OUTPUT DIR
# ==========================================================

OUTDIR = "nhits_plots_events"

os.makedirs(
    OUTDIR,
    exist_ok=True
)

# ==========================================================
# SIMULATION FILES
# ==========================================================

SIM_H5_FILES = {

    "electron":
        "/home/vmellado/FQM378/vmellado/GATrEnv/data/clasificador/50k_e-_Emin1GeV_Emax120GeV_continuous_fixed_position_5-5--20_sigmaMomentum_0.1_test.h5",

    "muon":
        "/home/vmellado/FQM378/vmellado/GATrEnv/data/clasificador/50k_mu-_Emin1GeV_Emax120GeV_continuous_fixed_position_5-5--20_sigmaMomentum_0.1_test.h5",

    "pion":
        "/home/vmellado/FQM378/vmellado/GATrEnv/data/clasificador/50k_pi-_Emin1GeV_Emax120GeV_continuous_fixed_position_5-5--20_sigmaMomentum_0.1_test.h5",
}

SIM_CLASSIFIED_CSV = (
    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/simulaciones_classified/classified_events.csv"
)

# ==========================================================
# REAL TESTBEAM FILES
# ==========================================================

REAL_H5_FILES = {

    "electron": [

        "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/electrones_20_test.h5",
        "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/electrones_50_test.h5",
        "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/electrones_80_test.h5",
    ],

    "pion": [

        "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/piones_20_test.h5",
        "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/piones_50_test.h5",
        "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/piones_80_test.h5",
    ]
}

REAL_ELECTRON_CSV = (
    "classified_testbeam_electrones.csv"
)

REAL_PION_CSV = (
    "classified_testbeam_piones.csv"
)

# ==========================================================
# LOAD NHITS FROM H5
# ==========================================================

def load_nhits_from_h5(h5file):

    print(f"Loading H5: {h5file}")

    with h5py.File(h5file, "r") as f:

        offsets = f["offsets"][:]

    nhits = np.diff(offsets)

    return nhits

# ==========================================================
# PLOT FUNCTION
# ==========================================================

def plot_distribution(
    original,
    classified,
    title,
    outfile
):

    plt.figure(figsize=(10,7))

    plt.hist(
        original,
        bins=100,
        density=False,
        histtype="step",
        linewidth=3,
        label="Original"
    )

    for particle, values in classified.items():

        if len(values) == 0:
            continue

        plt.hist(
            values,
            bins=100,
            density=False,
            histtype="step",
            linewidth=2,
            label=f"Predicted {particle}"
        )

    plt.xlabel("N hits")
    plt.ylabel("Density")

    plt.title(title)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTDIR, outfile),
        dpi=300
    )

    plt.close()

# ==========================================================
# SIMULATION NHITS
# ==========================================================

print("\n===================================")
print("SIMULATION NHITS")
print("===================================")

sim_original = {}

for particle, h5file in SIM_H5_FILES.items():

    nhits = load_nhits_from_h5(h5file)

    sim_original[particle] = nhits

# ----------------------------------------------------------
# LOAD CLASSIFIED SIMULATION CSV
# ----------------------------------------------------------

print("\nLoading classified simulation CSV")

sim_df = pd.read_csv(
    SIM_CLASSIFIED_CSV
)

print(sim_df.columns)

# ----------------------------------------------------------
# BUILD CLASSIFIED NHITS
# ----------------------------------------------------------

classified_sim_nhits = defaultdict(list)

for particle, h5file in SIM_H5_FILES.items():

    nhits = load_nhits_from_h5(h5file)

    mask = sim_df["label"] == particle

    subset = sim_df[mask]

    for _, row in subset.iterrows():

        evt = int(row["event_id"])

        pred = row["prediction"]

        if evt >= len(nhits):
            continue

        classified_sim_nhits[pred].append(
            nhits[evt]
        )

# ----------------------------------------------------------
# PLOTS SIMULATION
# ----------------------------------------------------------

for particle in ["electron", "muon", "pion"]:

    plot_distribution(

        original=sim_original[particle],

        classified=classified_sim_nhits,

        title=f"Simulation NHits - {particle}",

        outfile=f"simulation_nhits_{particle}.png"
    )

# ==========================================================
# REAL DATA NHITS
# ==========================================================

print("\n===================================")
print("REAL DATA NHITS")
print("===================================")

real_original = {}
real_classified = defaultdict(list)

# ----------------------------------------------------------
# ORIGINAL NHITS
# ----------------------------------------------------------

for label, h5list in REAL_H5_FILES.items():

    all_nhits = []

    for h5file in h5list:

        nhits = load_nhits_from_h5(h5file)

        all_nhits.extend(nhits)

    real_original[label] = all_nhits

# ----------------------------------------------------------
# LOAD CLASSIFIED CSVs
# ----------------------------------------------------------

print("\nLoading classified_testbeam_electrones.csv")

df_e = pd.read_csv(
    REAL_ELECTRON_CSV
)

print("\nLoading classified_testbeam_piones.csv")

df_p = pd.read_csv(
    REAL_PION_CSV
)

# ==========================================================
# ELECTRON TESTBEAM
# ==========================================================

electron_offsets = {}

offset = 0

for energy, h5file in zip(

    [20, 50, 80],

    REAL_H5_FILES["electron"]
):

    nhits = load_nhits_from_h5(h5file)

    electron_offsets[energy] = {

        "offset": offset,
        "nhits": nhits
    }

    offset += len(nhits)

for _, row in df_e.iterrows():

    evt = int(row["event_id"])

    pred = row["prediction"]

    for energy in [20, 50, 80]:

        info = electron_offsets[energy]

        start = info["offset"]

        stop = start + len(info["nhits"])

        if start <= evt < stop:

            local_evt = evt - start

            real_classified[pred].append(
                info["nhits"][local_evt]
            )

            break

# ==========================================================
# PION TESTBEAM
# ==========================================================

pion_offsets = {}

offset = 0

for energy, h5file in zip(

    [20, 50, 80],

    REAL_H5_FILES["pion"]
):

    nhits = load_nhits_from_h5(h5file)

    pion_offsets[energy] = {

        "offset": offset,
        "nhits": nhits
    }

    offset += len(nhits)

for _, row in df_p.iterrows():

    evt = int(row["event_id"])

    pred = row["prediction"]

    for energy in [20, 50, 80]:

        info = pion_offsets[energy]

        start = info["offset"]

        stop = start + len(info["nhits"])

        if start <= evt < stop:

            local_evt = evt - start

            real_classified[pred].append(
                info["nhits"][local_evt]
            )

            break

# ==========================================================
# PLOTS REAL DATA
# ==========================================================

for particle in ["electron", "pion"]:

    plot_distribution(

        original=real_original[particle],

        classified=real_classified,

        title=f"Real Testbeam NHits - {particle}",

        outfile=f"real_nhits_{particle}.png"
    )

print("\n===================================")
print("ALL NHITS PLOTS GENERATED")
print("===================================")
print(f"Saved in: {OUTDIR}")
