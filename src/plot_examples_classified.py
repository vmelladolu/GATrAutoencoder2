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

    "electron_20": "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam2/electrones_20.h5",
    "electron_50": "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam2/electrones_50.h5",
    "electron_80": "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam2/electrones_80.h5",

    "pion_20": "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/piones_20_test.h5",
    "pion_50": "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/piones_50_test.h5",
    "pion_80": "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/piones_80_test.h5",
}

# CSVs clasificados
REAL_ELECTRON_CSV = "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/classified_testbeam2_electrones.csv"
REAL_PION_CSV     = "classified_testbeam_piones.csv"


# ==========================================================
# OUTPUT
# ==========================================================

OUTDIR = "event_display_gallery"

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

    f = h5py.File(h5file, "r")

    x = f["x"][:]
    y = f["y"][:]
    z = f["z"][:]
    offsets = f["offsets"][:]

    start = offsets[event_id]
    end   = offsets[event_id + 1]

    evt_x = x[start:end]
    evt_y = y[start:end]
    evt_z = z[start:end]

    f.close()

    return evt_x, evt_y, evt_z


# ==========================================================
# DETERMINE REAL H5 FROM CSV ORIGIN
# ==========================================================

def determine_real_h5(row):

    source_file = row["source_file"]

    source_file = source_file.lower()

    if "electron" in source_file:

        particle = "electron"

    elif "pion" in source_file:

        particle = "pion"

    else:

        raise ValueError(
            f"Cannot determine particle from {source_file}"
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

    subset = df[
        df["prediction"] == particle_name
    ]

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

# ==========================================================
# ADD SOURCE INFO
# ==========================================================

# IMPORTANTE:
# aquí defines de qué dataset viene cada csv

real_e_df["source_file"] = "electron_80"
real_p_df["source_file"] = "pion_80"

# si tienes separados por energía:
# puedes concatenarlos individualmente

real_df = pd.concat(
    [
        real_e_df,
        real_p_df
    ],
    ignore_index=True
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

print("\n===================================")
print("ALL EVENT GALLERIES GENERATED")
print("===================================")
print(f"Output directory: {OUTDIR}")
