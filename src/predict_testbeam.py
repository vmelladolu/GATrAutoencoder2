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

def build_event_grpc_masks(h5_path, n_first=3, n_last=5, max_empty=2):
    with h5py.File(h5_path, "r") as f:
        offsets = f["offsets"][:]
        k_hits = f["k"][:]

    n_events = len(offsets) - 1

    first_signal = np.zeros(n_events, dtype=bool)
    last_signal = np.zeros(n_events, dtype=bool)
    complete_event = np.zeros(n_events, dtype=bool)
    unique_track = np.zeros(n_events, dtype=bool)

    for ev in range(n_events):
        start = offsets[ev]
        stop = offsets[ev + 1]
        ks = np.unique(k_hits[start:stop])

        if ks.size == 0:
            continue

        first_signal[ev] = np.any(ks < n_first)
        last_signal[ev] = np.any(ks >= (ks.max() - n_last + 1))

        sorted_ks = np.sort(ks)
        gaps = np.diff(sorted_ks)
        max_gap = gaps.max() if len(gaps) else 0
        complete_event[ev] = max_gap <= (max_empty + 1)

        unique_track[ev] = True

    return first_signal, last_signal, complete_event, unique_track

def inspect_h5(path):
    with h5py.File(path, "r") as f:
        print("Top-level keys:", list(f.keys()))

        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"DATASET {name} shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"GROUP    {name}")

        f.visititems(show)

def load_k_per_event(h5file):
    with h5py.File(h5file, "r") as f:
        offsets = f["offsets"][:]
        k_hits = f["k"][:]

    n_events = len(offsets) - 1
    K_event = np.zeros(n_events, dtype=np.int32)

    for ev in range(n_events):
        start = offsets[ev]
        stop = offsets[ev + 1]

        if stop <= start:
            K_event[ev] = 0
            continue

        K_event[ev] = len(np.unique(k_hits[start:stop]))

    return K_event



real_files = [
    (
        "electrones_testbeam_20_test.csv",
        "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/electrones_20_test.h5"
    ),
    (
        "electrones_testbeam_50_test.csv",
        "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/electrones_50_test.h5"
    ),
    (
        "electrones_testbeam_80_test.csv",
        "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam/electrones_80_test.h5"
    ),
]

dfs = []

for csv_path, h5_path in real_files:

    print(f"Loading CSV: {csv_path}")
    print(f"Loading H5:  {h5_path}")

    tmp = pd.read_csv(csv_path)

    nhits = load_nhits_from_h5(h5_path)

    inspect_h5(h5_path)
    K_event= load_k_per_event(h5_path)

    first_signal, last_signal, complete_event, unique_track = build_event_grpc_masks(h5_path)


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

    n = min(len(tmp), len(nhits), len(K_event), len(first_signal))

    if len(tmp) != len(nhits) or len(tmp) !=len(K_event) or len(first_signal) != n:
        print(
            f"WARNING: length mismatch in {csv_path}: "
            f"{len(tmp)} rows in CSV vs {len(nhits)} nhits vs {len(K_event)}. "
            f"Using first {n} events."
    )

    tmp = tmp.iloc[:n].copy()
    nhits = nhits[:n]
    K_event= K_event[:n]

    first_signal = first_signal[:n]
    last_signal = last_signal[:n]
    complete_event = complete_event[:n]
    unique_track = unique_track[:n]

    tmp["source_file"] = csv_path
    tmp["event_id"] = np.arange(len(tmp), dtype=int)
    tmp["nhits"] = nhits
    tmp["K_event"]= K_event

    tmp["K_event"] = pd.to_numeric(tmp["K_event"], errors="coerce")
    tmp["density"] = tmp["nhits"] / tmp["K_event"].replace(0, np.nan)

    tmp["first_signal"] = first_signal
    tmp["last_signal"] = last_signal
    tmp["complete_event"] = complete_event

    tmp["muon_mask"] = (
        (tmp["density"] < 3.5) &
        (tmp["nhits"] < 200) &
        first_signal &
        last_signal &
        complete_event &
        unique_track
    )

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
    [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
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

muon_mask = df["muon_mask"].to_numpy(dtype=bool)
preds[muon_mask] = 1
probs[muon_mask, :] = 0.0
probs[muon_mask, 1] = 1.0

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

out_file = "classified_testbeam1_electrones2.csv"

df.to_csv(
    out_file,
    index=False
)

print(f"Saved: {out_file}")
