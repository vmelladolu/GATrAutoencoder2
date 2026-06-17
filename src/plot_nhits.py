import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

# ==========================================================
# OUTPUT DIR
# ==========================================================

OUTDIR = "nhits_plots_events_softmax-cortes_hough"

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

        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_20_test_hough.h5",
        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_50_test_hough.h5",
        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_80_test_hough.h5",
    ],

    "pion": [

        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_20_test_hough.h5",
        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_50_test_hough.h5",
        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_80_test_hough.h5",
    ]
}

REAL_ELECTRON_CSV = (
    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/classified_electrones_hough.csv"
)

REAL_PION_CSV = (
    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/classified_piones_hough.csv"
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

    plt.xlabel("Nhits")
    plt.ylabel("Events")

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

classified_sim_nhits = defaultdict(lambda: defaultdict(list))

for particle, h5file in SIM_H5_FILES.items():

    nhits = load_nhits_from_h5(h5file)

    mask = sim_df["label"] == particle

    subset = sim_df[mask]

    for _, row in subset.iterrows():

        evt = int(row["event_id"])

        pred = row["prediction"]

        if evt >= len(nhits):
            continue

        classified_sim_nhits[particle][pred].append(
            nhits[evt]
        )

# ----------------------------------------------------------
# PLOTS SIMULATION
# ----------------------------------------------------------

for particle in ["electron", "muon", "pion"]:

    plot_distribution(

        original=sim_original[particle],

        classified=classified_sim_nhits[particle],

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
real_classified = defaultdict(lambda: defaultdict(list))

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

print(df_e.columns)
print(df_p.columns)
print(df_e["source_file"].value_counts())
print(df_p["source_file"].value_counts())

def print_class_summary(name, df):
    total = len(df)
    counts = df["prediction"].value_counts().reindex(["muon", "pion", "electron"], fill_value=0)
    perc = counts / total * 100

    print(f"\n=== {name} ===")
    print("Total:", total)
    for cls in ["muon", "pion", "electron"]:
        print(f"{cls:8s} {counts[cls]:10d}  ({perc[cls]:6.2f}%)")

print_class_summary("ELECTRON TESTBEAM", df_e)
print_class_summary("PION TESTBEAM", df_p)

#----------------------- PLOTS 80 GeV----------------------------------

df_e_80 = df_e[df_e["source_file"] == "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_80_test_testbeam2_hough/electrones_80_test2_features.csv"].copy()
df_p_80 = df_p[df_p["source_file"] == "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_80_test_testbeam2_hough/piones_80_test2_features.csv"].copy()

print_class_summary("ELECTRON 80 TESTBEAM", df_e_80)
print_class_summary("PION 80 TESTBEAM", df_p_80)

electron_nhits_80 = {
    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_80_test_testbeam2_hough/electrones_80_test2_features.csv": load_nhits_from_h5(
        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_80_test_hough.h5"
    )
}

pion_nhits_80 = {
    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_80_test_testbeam2_hough/piones_80_test2_features.csv": load_nhits_from_h5(
        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_80_test_hough.h5"
    )
}
# ==========================================================
# REAL DATA NHITS - 80 GeV ONLY
# ==========================================================

real_original_80 = {}
real_classified_80 = defaultdict(lambda: defaultdict(list))

real_original_80["electron"] = electron_nhits_80["/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_80_test_testbeam2_hough/electrones_80_test2_features.csv"]
real_original_80["pion"] = pion_nhits_80["/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_80_test_testbeam2_hough/piones_80_test2_features.csv"]

for _, row in df_e_80.iterrows():

    evt = int(row["event_id"])
    pred = row["prediction"]
    source = row["source_file"]

    nhits = electron_nhits_80[source]

    if evt >= len(nhits):
        continue

    real_classified_80["electron"][pred].append(
        nhits[evt]
    )

for _, row in df_p_80.iterrows():

    evt = int(row["event_id"])
    pred = row["prediction"]
    source = row["source_file"]

    nhits = pion_nhits_80[source]

    if evt >= len(nhits):
        continue

    real_classified_80["pion"][pred].append(
        nhits[evt]
    )

def plot_distribution(original, classified, title, outfile, xlim=None):
    plt.figure(figsize=(10, 7))

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
            alpha=0.9,
            label=f"Predicted {particle}"
        )

    plt.xlabel("Nhits")
    plt.ylabel("Events")
    plt.title(title)
    plt.legend()
    if xlim is not None:
        plt.xlim(*xlim)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, outfile), dpi=300)
    plt.close()

for particle in ["electron", "pion"]:
    plot_distribution(
        original=real_original_80[particle],
        classified=real_classified_80[particle],
        title=f"Real Testbeam NHits 80 GeV - {particle} (linear)",
        outfile=f"real_nhits_{particle}_80_linear.png",
        xlim=(0, 1700)
    )

# ==========================================================
# ELECTRON TESTBEAM
# ==========================================================

electron_nhits = {

    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_20_test_testbeam2_hough/electrones_20_test2_features.csv":
        load_nhits_from_h5(
            "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_20_test_hough.h5"
        ),

    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_50_test_testbeam2_hough/electrones_50_test2_features.csv":
        load_nhits_from_h5(
            "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_50_test_hough.h5"
        ),

    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_80_test_testbeam2_hough/electrones_80_test2_features.csv":
        load_nhits_from_h5(
            "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_80_test_hough.h5"
        ),
}

for _, row in df_e.iterrows():

    evt = int(row["event_id"])
    pred = row["prediction"]
    source = row["source_file"]

    nhits = electron_nhits[source]

    if evt >= len(nhits):
        continue

    real_classified["electron"][pred].append(
        nhits[evt]
    )

# ==========================================================
# PION TESTBEAM
# ==========================================================
pion_nhits = {

    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_20_test_testbeam2_hough/piones_20_test2_features.csv":
        load_nhits_from_h5(
            "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_20_test_hough.h5"
        ),

    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_50_test_testbeam2_hough/piones_50_test2_features.csv":
        load_nhits_from_h5(
            "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_50_test_hough.h5"
        ),

    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_80_test_testbeam2_hough/piones_80_test2_features.csv":
        load_nhits_from_h5(
            "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_80_test_hough.h5"
        ),
}

for _, row in df_p.iterrows():

    evt = int(row["event_id"])
    pred = row["prediction"]
    source = row["source_file"]

    nhits = pion_nhits[source]

    if evt >= len(nhits):
        continue

    real_classified["pion"][pred].append(
        nhits[evt]
    )

# ----------------------------------------------------------
# CHECK COUNTS HERE
# ----------------------------------------------------------
print("\n=== ELECTRON CHECK ===")
print("Original total:", len(real_original["electron"]))
for cls, vals in real_classified["electron"].items():
    print(cls, len(vals))

print("\n=== PION CHECK ===")
print("Original total:", len(real_original["pion"]))
for cls, vals in real_classified["pion"].items():
    print(cls, len(vals))

print(df_e["prediction"].value_counts())
print(df_p["prediction"].value_counts())

# ==========================================================
# PLOTS REAL DATA
# ==========================================================

for particle in ["electron", "pion"]:

    plot_distribution(

        original=real_original[particle],

        classified=real_classified[particle],

        title=f"Real Testbeam NHits - {particle}",

        outfile=f"real_nhits_{particle}.png"
    )

# ==========================================================
# COMPARACIÓN DIRECTA SIM vs TESTBEAM  <-- AÑADIR AQUÍ
# ==========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# --- Piones ---
ax = axes[0]
ax.hist(sim_original["pion"], bins=100, density=True, alpha=0.5, label="Sim pion")
ax.hist(sim_original["muon"], bins=100, density=True, alpha=0.5, label="Sim muon")
ax.hist(real_original["pion"], bins=100, density=True, histtype="step", lw=2, label="TB 'pion'")
ax.set_xlabel("NHits")
ax.set_ylabel("Density")
ax.set_title("Pion testbeam vs Simulación")
ax.legend()
ax.set_xlim(0, 500)
# --- Electrones ---
ax = axes[1]
ax.hist(sim_original["electron"], bins=100, density=True, alpha=0.5, label="Sim electron")
ax.hist(sim_original["muon"], bins=100, density=True, alpha=0.5, label="Sim muon")
ax.hist(real_original["electron"], bins=100, density=True, histtype="step", lw=2, label="TB 'electron'")
ax.set_xlabel("NHits")
ax.set_title("Electron testbeam vs Simulación")
ax.legend()
ax.set_xlim(0, 500)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "sim_vs_tb_comparison.png"), dpi=300)
plt.close()
print("Guardado: sim_vs_tb_comparison.png")
# ==========================================================
# FRACCIÓN CON NHITS > 200
# ==========================================================
print("\nFracción de eventos con NHits > 200:")
for label in ["electron", "pion"]:
    arr = np.array(real_original[label])
    frac = (arr > 200).mean()
    print(f"  TB {label}: {frac:.1%}")
# ==========================================================
# ALL NHITS PLOTS GENERATED
# ==========================================================

print("\n===================================")
print("ALL NHITS PLOTS GENERATED")
print("===================================")
print(f"Saved in: {OUTDIR}")
