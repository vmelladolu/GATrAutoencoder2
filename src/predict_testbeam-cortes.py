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

def load_event_xyzk(h5file):
    with h5py.File(h5file, "r") as f:
        offsets = f["offsets"][:]
        x_hits = f["x"][:]
        y_hits = f["y"][:]
        k_hits = f["k"][:]
    return offsets, x_hits, y_hits, k_hits

def compute_second_max_hits_per_layer(offsets, k_hits):
    n_events = len(offsets) - 1
    second_max_hits = np.zeros(n_events, dtype=np.int32)

    for ev in range(n_events):
        start = offsets[ev]
        stop = offsets[ev + 1]

        ks = k_hits[start:stop]
        if len(ks) == 0:
            second_max_hits[ev] = 0
            continue

        unique_layers, counts = np.unique(ks, return_counts=True)
        counts_sorted = np.sort(counts)[::-1]

        if len(counts_sorted) >= 2:
            second_max_hits[ev] = counts_sorted[1]
        else:
            second_max_hits[ev] = counts_sorted[0]

    return second_max_hits


def compute_longitudinal_per_event(offsets, k_hits, n_first_layers=14):
    n_events = len(offsets) - 1
    longitudinal = np.zeros(n_events, dtype=np.float32)
    nhits_first = np.zeros(n_events, dtype=np.int32)

    for ev in range(n_events):
        start = offsets[ev]
        stop = offsets[ev + 1]

        ks = k_hits[start:stop]
        totalhits = len(ks)

        if totalhits == 0:
            longitudinal[ev] = 0.0
            nhits_first[ev] = 0
            continue

        hits_first = np.sum(ks < n_first_layers)
        nhits_first[ev] = hits_first
        longitudinal[ev] = hits_first / totalhits

    return longitudinal, nhits_first

def compute_lateral_per_event(offsets, x_hits, y_hits, k_hits, layers_for_axis=10, radius=13):
    n_events = len(offsets) - 1

    lateral = np.zeros(n_events, dtype=np.float32)
    lateral_hits = np.zeros(n_events, dtype=np.int32)
    axis_x = np.full(n_events, np.nan, dtype=np.float32)
    axis_y = np.full(n_events, np.nan, dtype=np.float32)

    half_window = radius / 2.0

    for ev in range(n_events):
        start = offsets[ev]
        stop = offsets[ev + 1]

        xs = x_hits[start:stop]
        ys = y_hits[start:stop]
        ks = k_hits[start:stop]

        totalhits = len(xs)
        if totalhits == 0:
            continue

        mask_axis = ks < layers_for_axis
        if not np.any(mask_axis):
            continue

        x_mean = np.mean(xs[mask_axis])
        y_mean = np.mean(ys[mask_axis])

        axis_x[ev] = x_mean
        axis_y[ev] = y_mean

        mask_radius = (
            (xs >= x_mean - half_window) &
            (xs <= x_mean + half_window) &
            (ys >= y_mean - half_window) &
            (ys <= y_mean + half_window)
        )

        nhitsinradius = np.sum(mask_radius)
        lateral_hits[ev] = nhitsinradius
        lateral[ev] = nhitsinradius / totalhits

    return lateral, lateral_hits, axis_x, axis_y

def compute_ipstart_per_event(offsets, x_hits, y_hits, k_hits, radius=10, nearlayers=3, minhits=4):
    n_events = len(offsets) - 1
    ipstart = np.full(n_events, -1, dtype=np.int32)

    half_window = radius / 2.0

    for ev in range(n_events):
        start = offsets[ev]
        stop = offsets[ev + 1]

        xs = x_hits[start:stop]
        ys = y_hits[start:stop]
        ks = k_hits[start:stop]

        if len(xs) == 0:
            continue

        x_mean = np.mean(xs)
        y_mean = np.mean(ys)

        minx, maxx = x_mean - half_window, x_mean + half_window
        miny, maxy = y_mean - half_window, y_mean + half_window

        layers = np.unique(ks)
        layers.sort()

        if len(layers) < nearlayers:
            ipstart[ev] = -1
            continue

        consecutive = 0
        previous_layer = None
        first_layer_in_block = -1

        for layer in layers:
            layermask = ks == layer
            xs_layer = xs[layermask]
            ys_layer = ys[layermask]

            in_radius = (
                (xs_layer >= minx) & (xs_layer <= maxx) &
                (ys_layer >= miny) & (ys_layer <= maxy)
            )

            nhitsinradius = np.sum(in_radius)

            if nhitsinradius > minhits:
                if previous_layer is None or layer == previous_layer + 1:
                    consecutive += 1
                    if consecutive == 1:
                        first_layer_in_block = layer
                else:
                    consecutive = 1
                    first_layer_in_block = layer
            else:
                consecutive = 0
                first_layer_in_block = -1

            previous_layer = layer

            if consecutive >= nearlayers:
                ipstart[ev] = int(first_layer_in_block)
                break

    return ipstart


def compute_penetration_muon_mask(offsets, k_hits, density, min_density=5.0):
    n_events = len(offsets) - 1
    mask = np.zeros(n_events, dtype=bool)

    layer_ranges = [
        (1, 10, 7),
        (11, 20, 7),
        (21, 35, 9),
        (36, 48, 8),
    ]

    for ev in range(n_events):
        start = offsets[ev]
        stop = offsets[ev + 1]
        ks = np.unique(k_hits[start:stop])

        if len(ks) == 0:
            continue

        penetration_ok = True

        for lo, hi, min_layers in layer_ranges:
            n_layers = np.sum((ks >= lo) & (ks <= hi))
            if n_layers < min_layers:
                penetration_ok = False
                break

        if penetration_ok and density[ev] >= min_density:
            mask[ev] = True

    return mask

def compute_noise_filter_mask(density, second_max_hits, min_density=2.5, min_secondmax=5):
    return (density >= min_density) & (second_max_hits >= min_secondmax)


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

    inspect_h5(h5_path)
    K_event= load_k_per_event(h5_path)

    first_signal, last_signal, complete_event, unique_track = build_event_grpc_masks(h5_path)

    offsets, x_hits, y_hits, k_hits = load_event_xyzk(h5_path)

    second_max_hits = compute_second_max_hits_per_layer(offsets, k_hits)

    longitudinal_14, nhits_first14 = compute_longitudinal_per_event(
        offsets, k_hits, n_first_layers=14
    )

    lateral_x13, lateral_hits_x13, axis_x, axis_y = compute_lateral_per_event(
        offsets, x_hits, y_hits, k_hits,
        layers_for_axis=10,
        radius=13
    )

    ipstart = compute_ipstart_per_event(
        offsets, x_hits, y_hits, k_hits,
        radius=10,
        nearlayers=3,
        minhits=4
    )

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

    n = min(
        len(tmp),
        len(nhits),
        len(K_event),
        len(first_signal),
        len(second_max_hits),
        len(longitudinal_14),
        len(lateral_x13),
        len(ipstart)
    )

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

    second_max_hits = second_max_hits[:n]
    longitudinal_14 = longitudinal_14[:n]
    nhits_first14 = nhits_first14[:n]
    lateral_x13 = lateral_x13[:n]
    lateral_hits_x13 = lateral_hits_x13[:n]
    axis_x = axis_x[:n]
    axis_y = axis_y[:n]
    ipstart = ipstart[:n]

    tmp["source_file"] = csv_path
    tmp["event_id"] = np.arange(len(tmp), dtype=int)
    tmp["nhits"] = nhits
    tmp["K_event"]= K_event

    tmp["second_max_hits"] = second_max_hits
    tmp["nhits_first14"] = nhits_first14
    tmp["longitudinal_14"] = longitudinal_14
    tmp["lateral_hits_x13"] = lateral_hits_x13
    tmp["lateral_x13"] = lateral_x13

    #DEBUG LATERAL VS LONGITUDINAL
    print(tmp["longitudinal_14"].describe())
    print(tmp["lateral_x13"].describe())
    print(tmp[["longitudinal_14", "lateral_x13"]].head(20))

    tmp["axis_x"] = axis_x
    tmp["axis_y"] = axis_y
    tmp["ipstart"] = ipstart

    tmp["K_event"] = pd.to_numeric(tmp["K_event"], errors="coerce")
    tmp["density"] = tmp["nhits"] / tmp["K_event"].replace(0, np.nan)

    tmp["first_signal"] = first_signal
    tmp["last_signal"] = last_signal
    tmp["complete_event"] = complete_event

    noise_like_mask = ~compute_noise_filter_mask(
        density=tmp["density"].to_numpy(),
        second_max_hits=tmp["second_max_hits"].to_numpy(),
        min_density=2.5,
        min_secondmax=5
    )

    penetration_muon_mask = compute_penetration_muon_mask(
        offsets=offsets,
        k_hits=k_hits,
        density=tmp["density"].to_numpy(),
        min_density=5.0
    )[:n]

    tmp["noise_like_mask"] = noise_like_mask
    tmp["penetration_muon_mask"] = penetration_muon_mask

    tmp["muon_mask"] = (
        penetration_muon_mask &
        tmp["first_signal"] &
        tmp["last_signal"] &
        tmp["complete_event"]
    )

    tmp["lateral_accept"] = tmp["lateral_x13"] >= 0.4
    tmp["ipstart_accept"] = tmp["ipstart"] >= 0

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

out_file = "classified_testbeam1_piones-cortes.csv"

df.to_csv(
    out_file,
    index=False
)

print(f"Saved: {out_file}")
