import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

CSV_DIR = Path(
    "/media/FQM378/vmellado/GATrEnv/GATrAutoencoder"
)

H5_DIR = Path(
    "/media/FQM378/vmellado/GATrEnv/data/clasificador"
)

OUTPUT_DIR = Path(
    "/media/FQM378/vmellado/GATrEnv/GATrAutoencoder/event_displays"
)

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD CSVs
# ============================================================

csv_files = [
    CSV_DIR / "resultados_electron_train.csv",
    CSV_DIR / "resultados_muon_train.csv",
    CSV_DIR / "resultados_pion_train.csv",
    CSV_DIR / "resultados_electron_test.csv",
    CSV_DIR / "resultados_muon_test.csv",
    CSV_DIR / "resultados_pion_test.csv",
]

dfs = []

for f in csv_files:
    dfs.append(pd.read_csv(f))

df = pd.concat(dfs, ignore_index=True)

print(f"Total rows: {len(df)}")


# ============================================================
# SAMPLE EVENTS
# ============================================================

sample = df.sample(20, random_state=42)


# ============================================================
# MAP LABEL -> H5 FILE
# ============================================================

label_to_h5 = {
    "electron": H5_DIR / "50k_e-_Emin1GeV_Emax120GeV_continuous_fixed_position_5-5--20_sigmaMomentum_0.1_train.h5",
    "electron": H5_DIR / "50k_e-_Emin1GeV_Emax120GeV_continuous_fixed_position_5-5--20_sigmaMomentum_0.1_test.h5"

    "muon": H5_DIR / "50k_mu-_Emin1GeV_Emax120GeV_continuous_fixed_position_5-5--20_sigmaMomentum_0.1_train.h5",
    "muon": H5_DIR / "50k_mu-_Emin1GeV_Emax120GeV_continuous_fixed_position_5-5--20_sigmaMomentum_0.1_test.h5"

    "pion": H5_DIR / "50k_pi-_Emin1GeV_Emax120GeV_continuous_fixed_position_5-5--20_sigmaMomentum_0.1_train.h5",
    "pion": H5_DIR / "50k_pi-_Emin1GeV_Emax120GeV_continuous_fixed_position_5-5--20_sigmaMomentum_0.1_test.h5"
}


# ============================================================
# LOOP OVER EVENTS
# ============================================================

for _, row in sample.iterrows():

    event_id = int(row["event_id"])

    true_label = row["label"]

    pred_label = row.get("prediction", "unknown")

    h5_path = label_to_h5[true_label]

    print(f"Processing event {event_id}")

    with h5py.File(h5_path, "r") as f:

        event_ids = f["event_id"][:]

        i_all = f["i"][:]

        j_all = f["j"][:]

        energy_all = f["energy"][:]

        # select hits belonging to this event
        mask = (event_ids == event_id)

        i_hits = i_all[mask]

        j_hits = j_all[mask]

        e_hits = energy_all[mask]

        # build 2D image
        img = np.zeros((30, 30))

        for ii, jj, ee in zip(i_hits, j_hits, e_hits):

            img[ii, jj] += ee

        # ====================================================
        # PLOT
        # ====================================================

        plt.figure(figsize=(6, 6))

        plt.imshow(
            img.T,
            origin="lower",
            aspect="auto"
        )

        plt.title(
            f"event={event_id} | true={true_label} | pred={pred_label}"
        )

        plt.xlabel("i")

        plt.ylabel("j")

        plt.colorbar(label="Energy")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"event_{event_id}.png",
            dpi=200
        )

        plt.close()

print("Done.")
