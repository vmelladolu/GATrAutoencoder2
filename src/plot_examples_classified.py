import os
import random
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

# ==========================================================
# SIMULATED DATA
# ==========================================================

SIM_H5_FILES = {

    "electron": "/home/vmellado/FQM378/vmellado/GATrEnv/data/clasificador/50k_e-_Emin1GeV_Emax120GeV_continuous_fixed_position_5-5--20_sigmaMomentum_0.1_test.h5",
    "pion":     "/home/vmellado/FQM378/vmellado/GATrEnv/data/clasificador/50k_pi-_Emin1GeV_Emax120GeV_continuous_fixed_position_5-5--20_sigmaMomentum_0.1_test.h5",
    "muon":     "/home/vmellado/FQM378/vmellado/GATrEnv/data/clasificador/50k_mu-_Emin1GeV_Emax120GeV_continuous_fixed_position_5-5--20_sigmaMomentum_0.1_test.h5",
}

SIM_CLASSIFIED_CSV = "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/simulaciones_classified/classified_events.csv"


# ==========================================================
# REAL TESTBEAM DATA
# ==========================================================

REAL_H5_FILES = {

    "electron_20": "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_20_test_hough.h5",
    "electron_50": "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_50_test_hough.h5",
    "electron_80": "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_80_test_hough.h5",

    "pion_20": "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_20_test_hough.h5",
    "pion_50": "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_50_test_hough.h5",
    "pion_80": "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_80_test_hough.h5",
}

# CSVs clasificados
REAL_ELECTRON_CSV = "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/classified_electrones_hough.csv"
REAL_PION_CSV     = "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/classified_piones_hough.csv"


# ==========================================================
# OUTPUT
# ==========================================================

OUTDIR = "event_display_gallery_softmax"

os.makedirs(
    OUTDIR,
    exist_ok=True
)

# ==========================================================
# PARAMETERS
# ==========================================================

N_EVENTS_PER_CLASS = 10

POINT_SIZE = 3

ALPHA = 0.7


# ==========================================================
# LOAD H5 EVENT
# ==========================================================

def load_event(h5file, event_id):
    with h5py.File(h5file, "r") as f:
        x = f["x"][:]
        y = f["y"][:]
        z = f["z"][:]
        offsets = f["offsets"][:]

        if event_id >= len(offsets) - 1:
            raise ValueError(f"event_id={event_id} outside file range ({len(offsets)-1} events)")

        start = offsets[event_id]
        end = offsets[event_id + 1]

        return x[start:end], y[start:end], z[start:end]


# ==========================================================
# DETERMINE REAL H5 FROM CSV ORIGIN
# ==========================================================

def determine_real_h5(row):

    source = row["source_file"]

    if source == "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_20_test_testbeam2_hough/electrones_20_test2_features.csv":

        return REAL_H5_FILES["electron_20"]

    elif source == "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_50_test_testbeam2_hough/electrones_50_test2_features.csv":

        return REAL_H5_FILES["electron_50"]

    elif source == "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_80_test_testbeam2_hough/electrones_80_test2_features.csvv":
        return REAL_H5_FILES["electron_80"]

    elif source == "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_20_test_testbeam2_hough/piones_20_test2_features.csv":
        return REAL_H5_FILES["pion_20"]

    elif source == "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_50_test_testbeam2_hough/piones_50_test2_features.csv":
        return REAL_H5_FILES["pion_50"]

    elif source == "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_80_test_testbeam2_hough/piones_80_test2_features.csv":
        return REAL_H5_FILES["pion_80"]

    raise ValueError(
        f"Unknown source_file: {source}"
    )

    # energy
    if "20" in source_file:

        energy = 20

    elif "50" in source_file:

        energy = 50

    elif "80" in source_file:

        energy = 80

    else:

        raise ValueError(
            f"Cannot determine energy from {source_file}"
        )

    key = f"{particle}_{energy}"

    return REAL_H5_FILES[key]


# ==========================================================
# PLOT GALLERY
# ==========================================================

def plot_gallery(
    df,
    h5_mapping,
    particle_name,
    dataset_name,
    output_name
):

    mask = df["prediction"] == particle_name
    if "nhits" in df.columns:
        mask &= df["nhits"] < 200
    subset = df[mask]

    print(
        f"\n{dataset_name} | {particle_name} | "
        f"events = {len(subset)}"
    )

    if len(subset) == 0:

        print("No events found")
        return

    sample_df = subset.sample(
        min(N_EVENTS_PER_CLASS, len(subset)),
        random_state=42
    )

    fig, axes = plt.subplots(
        nrows=len(sample_df),
        ncols=2,
        figsize=(10, 4 * len(sample_df))
    )

    if len(sample_df) == 1:
        axes = np.array([axes])

    # ======================================================
    # LOOP EVENTS
    # ======================================================

    for idx, (_, row) in enumerate(sample_df.iterrows()):

        event_id = int(row["event_id"])

        # ==================================================
        # DETERMINE H5
        # ==================================================

        if dataset_name == "simulation":

            true_label = row["label"]

            h5file = h5_mapping[true_label]

        else:

            h5file = determine_real_h5(row)

        with h5py.File(h5file, "r") as f:
            offsets = f["offsets"][:]

            print("row event_id:", event_id)

            if "nhits" in row.index:
                print("row nhits:", row["nhits"])
            elif "nb_hits" in row.index:
                print("row nb_hits:", row["nb_hits"])
            else:
                print("row label:", row["label"])

            print("file events:", len(offsets) - 1)

            if event_id < len(offsets) - 1:
                print("start/end:", offsets[event_id], offsets[event_id + 1])
                print("n loaded:", offsets[event_id + 1] - offsets[event_id])
            else:
                print("event_id out of range for this file")

        # ==================================================
        # LOAD EVENT
        # ==================================================

        try:

            evt_x, evt_y, evt_z = load_event(
                h5file,
                event_id
            )

        except Exception as e:

            print(
                f"Could not load event {event_id}"
            )

            print(e)

            continue

        # ==================================================
        # XZ
        # ==================================================

        ax1 = axes[idx, 0]

        ax1.scatter(
            evt_z,
            evt_x,
            s=POINT_SIZE,
            alpha=ALPHA
        )

        ax1.set_xlabel("Layer Z")
        ax1.set_ylabel("X")

        ax1.set_title(
            f"{particle_name} | evt {event_id} | XZ"
        )

        # ==================================================
        # YZ
        # ==================================================

        ax2 = axes[idx, 1]

        ax2.scatter(
            evt_z,
            evt_y,
            s=POINT_SIZE,
            alpha=ALPHA
        )

        ax2.set_xlabel("Layer Z")
        ax2.set_ylabel("Y")

        ax2.set_title(
            f"{particle_name} | evt {event_id} | YZ"
        )

    plt.tight_layout()

    savepath = os.path.join(
        OUTDIR,
        output_name
    )

    plt.savefig(
        savepath,
        dpi=300
    )

    plt.close()

    print(f"Saved: {savepath}")


# ==========================================================
# LOAD SIMULATION CSV
# ==========================================================

print("\nLoading simulation CSV...")

sim_df = pd.read_csv(
    SIM_CLASSIFIED_CSV
)

# ==========================================================
# LOAD REAL CSVS
# ==========================================================

print("\nLoading real CSVs...")

real_e_df = pd.read_csv(
    REAL_ELECTRON_CSV
)

real_p_df = pd.read_csv(
    REAL_PION_CSV
)

# DEBUG
print(real_e_df.columns)
print(real_p_df.columns)

print(real_e_df["source_file"].value_counts())


# ==========================================================
# ADD SOURCE INFO
# ==========================================================

# IMPORTANTE:
# aquí defines de qué dataset viene cada csv


# si tienes separados por energía:
# puedes concatenarlos individualmente

real_df = pd.concat(
    [
        real_e_df,
        real_p_df
    ],
    ignore_index=True
)

real_e_80_df = real_e_df[
    real_e_df["source_file"] == "electrones_80_testbeam2.csv"
].copy()

real_p_80_df = real_p_df[
    real_p_df["source_file"] == "piones_testbeam_80_test.csv"
].copy()

pion80_as_electron_lowhits = real_p_80_df[
    (real_p_80_df["prediction"] == "electron") &
    (real_p_80_df["nhits"] < 250)
].copy()

print("\n=== PION 80 -> ELECTRON with nhits < 250 ===")
print("Events:", len(pion80_as_electron_lowhits))
print(pion80_as_electron_lowhits[["event_id", "nhits", "prediction", "source_file"]].head(20))

with h5py.File(REAL_H5_FILES["pion_80"], "r") as f:
    print(list(f.keys()))
    for k in ["x", "y", "z", "offsets"]:
        print(k, f[k].shape)

plot_gallery(
    df=pion80_as_electron_lowhits,
    h5_mapping=REAL_H5_FILES,
    particle_name="electron",
    dataset_name="real",
    output_name="real_p80_pred_electron_lowhits_gallery.png"
)

# ==========================================================
# SIMULATION GALLERIES
# ==========================================================

for particle in [
    "electron",
    "pion",
    "muon"
]:

    plot_gallery(
        df=sim_df,
        h5_mapping=SIM_H5_FILES,
        particle_name=particle,
        dataset_name="simulation",
        output_name=f"simulation_{particle}_gallery.png"
    )

# ==========================================================
# REAL DATA GALLERIES
# ==========================================================

for particle in [
    "electron",
    "pion",
    "muon"
]:

    plot_gallery(
        df=real_df,
        h5_mapping=REAL_H5_FILES,
        particle_name=particle,
        dataset_name="real",
        output_name=f"real_{particle}_gallery.png"
    )

# ==========================================================
# REAL DATA GALLERIES - ELECTRONS 80
# ==========================================================
muon_high_hits_e80 = real_e_80_df[
    (real_e_80_df["prediction"] == "muon") &
    (real_e_80_df["nhits"] > 1000)
].copy()

pion80_muon = real_p_80_df[
    (real_p_80_df["prediction"] == "muon") &
    (real_p_80_df["nhits"] > 1000)
].copy()

pion80_pion = real_p_80_df[
    (real_p_80_df["prediction"] == "pion")
].copy()

plot_gallery(
    df=muon_high_hits_e80,
    h5_mapping=REAL_H5_FILES,
    particle_name="muon",
    dataset_name="real",
    output_name="real_muon_highhits_e80_gallery.png"
)

plot_gallery(
    df=pion80_muon,
    h5_mapping=REAL_H5_FILES,
    particle_name="muon",
    dataset_name="real",
    output_name="real_pion80_muon_highhits_gallery.png"
)

plot_gallery(
    df=pion80_pion,
    h5_mapping=REAL_H5_FILES,
    particle_name="pion",
    dataset_name="real",
    output_name="real_pion80_muon_highhits_gallery.png"
)


for pred_class in ["electron", "muon", "pion"]:
    plot_gallery(
            df=real_e_80_df,
            h5_mapping=REAL_H5_FILES,
            particle_name=pred_class,
            dataset_name="real",
            output_name=f"real_e80_{pred_class}_gallery.png"
    )

# ==========================================================
# REAL DATA GALLERIES - ELECTRONS 80 GeV, nhits < 200
# ==========================================================

real_e_80_lowhits = real_e_80_df[
    real_e_80_df["nhits"] < 200
].copy()

print("\n=== ELECTRONS 80 GeV | ORIGINAL | nhits < 200 ===")
print(f"Events: {len(real_e_80_lowhits)}")
print(real_e_80_lowhits[["event_id", "nhits", "prediction", "source_file"]].head(20))

for pred_class in ["electron", "muon", "pion"]:
    plot_gallery(
        df=real_e_80_lowhits,
        h5_mapping=REAL_H5_FILES,
        particle_name=pred_class,
        dataset_name="real",
        output_name=f"real_e80_{pred_class}_nhitslt200_gallery.png"
    )

# ==========================================================
# REAL DATA GALLERIES - PIONS 80 GeV, nhits < 200
# ==========================================================

real_p_80_lowhits = real_p_80_df[
    real_p_80_df["nhits"] < 200
].copy()

print("\n=== PIONS 80 GeV | ORIGINAL | nhits < 200 ===")
print(f"Events: {len(real_p_80_lowhits)}")
print(real_p_80_lowhits[["event_id", "nhits", "prediction", "source_file"]].head(20))

for pred_class in ["electron", "muon", "pion"]:
    plot_gallery(
        df=real_p_80_lowhits,
        h5_mapping=REAL_H5_FILES,
        particle_name=pred_class,
        dataset_name="real",
        output_name=f"real_p80_{pred_class}_nhitslt200_gallery.png"
    )


# ==========================================================
# REAL DATA GALLERIES - PIONS 80
# ==========================================================

for pred_class in ["electron", "muon", "pion"]:
    plot_gallery(
        df=real_p_80_df,
        h5_mapping=REAL_H5_FILES,
        particle_name=pred_class,
        dataset_name="real",
        output_name=f"real_p80_{pred_class}_gallery.png"
    )

print(real_e_80_df[["event_id", "nhits", "prediction", "source_file"]].head(20))
print(real_e_80_df["event_id"].min(), real_e_80_df["event_id"].max())
print(real_e_80_df["event_id"].is_monotonic_increasing)
with h5py.File(REAL_H5_FILES["electron_80"], "r") as f:
    print(len(f["offsets"]) - 1)

print("\n===================================")
print("ALL EVENT GALLERIES GENERATED")
print("===================================")
print(f"Output directory: {OUTDIR}")

