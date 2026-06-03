import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# ==========================================================
# LOAD REAL DATA
# ==========================================================
def load_nhits_from_h5(h5file):
    with h5py.File(h5file, "r") as f:
        offsets = f["offsets"][:]
    return np.diff(offsets)


real_files = [
    (
        "piones_testbeam_20_test.csv",
        "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/piones_20_test.h5"
    ),
    (
        "piones_testbeam_50_test.csv",
        "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/piones_50_test.h5"
    ),
    (
        "piones_testbeam_80_test.csv",
        "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/piones_80_test.h5"
    ),
]

dfs = []

for csv_path, h5_path in real_files:

    print(f"Loading CSV: {csv_path}")
    print(f"Loading H5:  {h5_path}")

    tmp = pd.read_csv(csv_path)

    nhits = load_nhits_from_h5(h5_path)

    # ==========================================================
    # DEBUG CHECK: CSV vs H5 ALIGNMENT
    # ==========================================================
    print("\n--- DEBUG CHECK ---")
    print(f"File: {csv_path}")
    print(f"CSV rows: {len(tmp)}")
    print(f"H5 events: {len(nhits)}")
    print("\nFirst 3 CSV rows:")
    print(tmp.head(3))
    print("\nFirst 10 nhits:")
    print(nhits[:10])

    if len(tmp) > 0:
        print("\nLast CSV row:")
        print(tmp.iloc[-1].to_dict())

    if len(nhits) > 0:
        print("\nLast nhits value:")
        print(nhits[-1])
    #--------------------------------------------------

    n = min(len(tmp), len(nhits))

    if len(tmp) != len(nhits):
        print(
            f"WARNING: length mismatch in {csv_path}: "
            f"{len(tmp)} rows in CSV vs {len(nhits)} events in H5. "
            f"Using first {n} events."
    )

    tmp = tmp.iloc[:n].copy()
    nhits = nhits[:n]

    tmp["source_file"] = csv_path
    tmp["event_id"] = np.arange(len(tmp), dtype=int)
    tmp["nhits"] = nhits

    dfs.append(tmp)

df = pd.concat(
    dfs,
    ignore_index=True
)
print(df.columns)

if "source_file" in df.columns:
    print(df["source_file"].value_counts())
# ==========================================================
# LATENT COLUMNS
# ==========================================================

latent_cols = sorted(
    [c for c in df.columns if c.startswith("f")],
    key=lambda x: int(x[1:])
)

print("Latent dimensions:", len(latent_cols))

# ==========================================================
# FEATURES
# ==========================================================

X = df[latent_cols].values.astype(np.float32)

# ==========================================================
# LOAD SCALER
# ==========================================================

scaler = joblib.load(
    "softmax_scaler.pkl"
)

X = scaler.transform(X)

# ==========================================================
# TORCH
# ==========================================================

X = torch.tensor(
    X,
    dtype=torch.float32
)

# ==========================================================
# MODEL
# ==========================================================

class SoftmaxClassifier(nn.Module):

    def __init__(self, input_dim, n_classes):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, n_classes)
        )

    def forward(self, x):

        return self.net(x)

# ==========================================================
# LOAD MODEL
# ==========================================================

checkpoint = torch.load(
    "softmax_classifier.pt",
    map_location="cpu"
)

model = SoftmaxClassifier(
    input_dim=checkpoint["latent_dim"],
    n_classes=checkpoint["n_classes"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded")

# ==========================================================
# INFERENCE
# ==========================================================

with torch.no_grad():

    logits = model(X)

    probs = torch.softmax(
        logits,
        dim=1
    ).numpy()

    preds = np.argmax(
        probs,
        axis=1
    )

# ==========================================================
# PHYSICS-BASED POST-PROCESSING
# ==========================================================

if "nhits" in df.columns:

    nhits_thr = 200
    muon_thr = 0.60

    low_hits = df["nhits"].values < nhits_thr
    confident_muon = probs[:, 1] > muon_thr

    force_muon = low_hits & confident_muon

    preds[force_muon] = 1   # 1 = muon

    probs[force_muon, :] = 0.0
    probs[force_muon, 1] = 1.0

# ==========================================================
# SAVE RESULTS
# ==========================================================

df["electron_score"] = probs[:, 0]
df["muon_score"] = probs[:, 1]
df["pion_score"] = probs[:, 2]

label_map = {
    0: "electron",
    1: "muon",
    2: "pion"
}

df["prediction"] = [
    label_map[p]
    for p in preds
]

out_file = "classified_testbeam1_piones2.csv"

df.to_csv(
    out_file,
    index=False
)

print(f"Saved: {out_file}")
