import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# ==========================================================
# ARGUMENTS
# ==========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--csv",
    required=True,
    help="CSV clasificado"
)

parser.add_argument(
    "--h5",
    required=True,
    help="Archivo HDF5 original"
)

parser.add_argument(
    "--event",
    type=int,
    required=True,
    help="Índice del evento en el CSV"
)

parser.add_argument(
    "--save",
    default=None,
    help="Guardar figura"
)

args = parser.parse_args()

# ==========================================================
# LOAD CSV
# ==========================================================

print("Loading CSV...")

df = pd.read_csv(args.csv)

print(f"Total events in CSV: {len(df)}")

# ==========================================================
# SELECT EVENT
# ==========================================================

row = df.iloc[args.event]

print("\n==============================")
print("EVENT INFORMATION")
print("==============================")

for col in row.index:
    print(f"{col}: {row[col]}")

# ==========================================================
# EVENT ID
# ==========================================================

# MUY IMPORTANTE:
# el CSV debe contener event_idx
# que corresponde al índice del evento en el h5

event_idx = int(row["event_id"])

print(f"\nUsing H5 event index: {event_idx}")

# ==========================================================
# LOAD HDF5
# ==========================================================

print("\nLoading HDF5...")

f = h5py.File(args.h5, "r")

x = f["x"][:]
y = f["y"][:]
z = f["z"][:]
thr = f["thr"][:]

offsets = f["offsets"][:]

# ==========================================================
# GET EVENT HITS
# ==========================================================

start = offsets[event_idx]
end = offsets[event_idx + 1]

evt_x = x[start:end]
evt_y = y[start:end]
evt_z = z[start:end]
evt_thr = thr[start:end]

print(f"Hits in event: {len(evt_x)}")

# ==========================================================
# LABELS
# ==========================================================

true_label = row.get("label", "unknown")
pred_label = row.get("prediction_name", "unknown")

electron_score = row.get("electron_score", None)
pion_score = row.get("pion_score", None)
muon_score = row.get("muon_score", None)

# ==========================================================
# PLOT
# ==========================================================

plt.style.use("dark_background")

fig = plt.figure(figsize=(12, 10))

ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    evt_x,
    evt_y,
    evt_z,
    c=evt_thr,
    cmap="viridis",
    s=8,
    alpha=0.85
)

# ==========================================================
# AXES
# ==========================================================

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Layer")

# ==========================================================
# TITLE
# ==========================================================

title = (
    f"Event {event_idx}\n"
    f"True: {true_label} | Pred: {pred_label}"
)

if electron_score is not None:

    title += (
        f"\n"
        f"e={electron_score:.3f}  "
        f"π={pion_score:.3f}  "
        f"μ={muon_score:.3f}"
    )

ax.set_title(title)

# ==========================================================
# COLORBAR
# ==========================================================

cbar = plt.colorbar(sc)

cbar.set_label("Threshold")

# ==========================================================
# LAYOUT
# ==========================================================

plt.tight_layout()

# ==========================================================
# SAVE
# ==========================================================

if args.save is not None:

    plt.savefig(
        args.save,
        dpi=300
    )

    print(f"\nSaved figure: {args.save}")

plt.show()
