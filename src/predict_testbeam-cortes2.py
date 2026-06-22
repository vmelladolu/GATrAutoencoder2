import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader

# ==========================================================
# HELPERS
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
            continue
        _, counts = np.unique(ks, return_counts=True)
        counts_sorted = np.sort(counts)[::-1]
        second_max_hits[ev] = counts_sorted[1] if len(counts_sorted) >= 2 else counts_sorted[0]

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

def inspect_h5(path):
    with h5py.File(path, "r") as f:
        print("Top-level keys:", list(f.keys()))
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"DATASET {name} shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"GROUP    {name}")
        f.visititems(show)

def build_real_df(real_files):
    dfs = []

    for csv_path, h5_path in real_files:
        print(f"Loading CSV: {csv_path}")
        print(f"Loading H5:  {h5_path}")

        tmp = pd.read_csv(csv_path)
        nhits = load_nhits_from_h5(h5_path)
        K_event = load_k_per_event(h5_path)
        first_signal, last_signal, complete_event, unique_track = build_event_grpc_masks(h5_path)
        offsets, x_hits, y_hits, k_hits = load_event_xyzk(h5_path)
        second_max_hits = compute_second_max_hits_per_layer(offsets, k_hits)
        longitudinal_14, nhits_first14 = compute_longitudinal_per_event(offsets, k_hits, n_first_layers=14)
        lateral_x13, lateral_hits_x13, axis_x, axis_y = compute_lateral_per_event(offsets, x_hits, y_hits, k_hits, layers_for_axis=10, radius=13)
        ipstart = compute_ipstart_per_event(offsets, x_hits, y_hits, k_hits, radius=10, nearlayers=3, minhits=4)

        n = min(len(tmp), len(nhits), len(K_event), len(first_signal), len(second_max_hits), len(longitudinal_14), len(lateral_x13), len(ipstart))

        tmp = tmp.iloc[:n].copy()
        nhits = nhits[:n]
        K_event = K_event[:n]

        tmp["source_file"] = csv_path
        tmp["event_id"] = np.arange(len(tmp), dtype=int)
        tmp["nhits"] = nhits
        tmp["K_event"] = K_event
        tmp["second_max_hits"] = second_max_hits[:n]
        tmp["nhits_first14"] = nhits_first14[:n]
        tmp["longitudinal_14"] = longitudinal_14[:n]
        tmp["lateral_hits_x13"] = lateral_hits_x13[:n]
        tmp["lateral_x13"] = lateral_x13[:n]
        tmp["axis_x"] = axis_x[:n]
        tmp["axis_y"] = axis_y[:n]
        tmp["ipstart"] = ipstart[:n]

        tmp["K_event"] = pd.to_numeric(tmp["K_event"], errors="coerce")
        tmp["density"] = tmp["nhits"] / tmp["K_event"].replace(0, np.nan)

        tmp["first_signal"] = first_signal[:n]
        tmp["last_signal"] = last_signal[:n]
        tmp["complete_event"] = complete_event[:n]

        noise_like_mask = ~compute_noise_filter_mask(
            density=tmp["density"].to_numpy(),
            second_max_hits=tmp["second_max_hits"].to_numpy(),
            min_density=2.5,
            min_secondmax=5
        )

        penetration_muon_mask = compute_penetration_muon_mask(
            offsets=offsets[:n + 1],
            k_hits=k_hits[:offsets[n]],
            density=tmp["density"].to_numpy(),
            min_density=5.0
        )

        tmp["noise_like_mask"] = noise_like_mask
        tmp["penetration_muon_mask"] = penetration_muon_mask
        tmp["muon_mask"] = penetration_muon_mask & tmp["first_signal"] & tmp["last_signal"] & tmp["complete_event"]
        tmp["lateral_accept"] = tmp["lateral_x13"] >= 0.4
        tmp["ipstart_accept"] = tmp["ipstart"] >= 0

        dfs.append(tmp)

    return pd.concat(dfs, ignore_index=True)

# ==========================================================
# TRAIN / VAL CSVs
# ==========================================================
train_csvs = [
    "resultados_electron_train.csv",
    "resultados_pion_train.csv",
    "resultados_muon_train.csv",
]

val_csvs = [
    "resultados_electron_test.csv",
    "resultados_pion_test.csv",
    "resultados_muon_test.csv",
]

NORMALIZATION = {
    "electrones": "electron",
    "electron": "electron",
    "muones": "muon",
    "muon": "muon",
    "pion": "pion",
    "piones": "pion",
}

train_df = pd.concat([pd.read_csv(p) for p in train_csvs], ignore_index=True)
val_df = pd.concat([pd.read_csv(p) for p in val_csvs], ignore_index=True)

train_df["label"] = train_df["label"].astype(str).str.lower().map(NORMALIZATION)
val_df["label"] = val_df["label"].astype(str).str.lower().map(NORMALIZATION)

latent_cols = sorted([c for c in train_df.columns if c.startswith("f") and c[1:].isdigit()], key=lambda x: int(x[1:]))
labels = sorted(train_df["label"].dropna().unique())
label_to_int = {lab: i for i, lab in enumerate(labels)}
int_to_label = {i: lab for lab, i in label_to_int.items()}

X_train = train_df[latent_cols].values.astype(np.float32)
y_train = train_df["label"].map(label_to_int).values.astype(np.int64)
X_val = val_df[latent_cols].values.astype(np.float32)
y_val = val_df["label"].map(label_to_int).values.astype(np.int64)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
joblib.dump(scaler, "softmax_scaler.pkl")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=512, shuffle=False)

class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SoftmaxClassifier(len(latent_cols), len(labels)).to(device)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(sorted(np.unique(y_train.numpy()))),
    y=y_train.numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

best_val_loss = float("inf")
patience = 12
patience_count = 0

for epoch in range(1, 101):
    model.train()
    tr_loss = tr_ok = tr_n = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss += loss.item() * xb.size(0)
        tr_ok += (logits.argmax(1) == yb).sum().item()
        tr_n += xb.size(0)

    model.eval()
    va_loss = va_ok = va_n = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            va_loss += loss.item() * xb.size(0)
            va_ok += (logits.argmax(1) == yb).sum().item()
            va_n += xb.size(0)

    tr_loss /= tr_n
    va_loss /= va_n
    tr_acc = tr_ok / tr_n
    va_acc = va_ok / va_n

    print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        patience_count = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "latent_dim": len(latent_cols),
            "n_classes": len(labels),
            "label_to_int": label_to_int,
            "latent_cols": latent_cols,
        }, "softmax_classifier.pt")
    else:
        patience_count += 1
        if patience_count >= patience:
            print("Early stopping")
            break

checkpoint = torch.load("softmax_classifier.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    all_true = []
    all_pred = []
    for xb, yb in val_loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
        all_pred.append(pred)
        all_true.append(yb.numpy())

all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)

print(classification_report(all_true, all_pred, target_names=[int_to_label[i] for i in range(len(labels))]))
print(confusion_matrix(all_true, all_pred))
print("Saved softmax_classifier.pt")

# ==========================================================
# REAL TESTBEAM INFERENCE
# ==========================================================
real_files = [
    (
        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_20_test_testbeam2_hough/electrones_20_test2_features.csv",
        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_20_test_hough.h5"
    ),
    (
        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_50_test_testbeam2_hough/electrones_50_test2_features.csv",
        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_50_test_hough.h5"
    ),
    (
        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_80_test_testbeam2_hough/electrones_80_test2_features.csv",
        "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_80_test_hough.h5"
    ),
]

df = build_real_df(real_files)

X = df[latent_cols].values.astype(np.float32)
scaler = joblib.load("softmax_scaler.pkl")
X = scaler.transform(X)
X = torch.tensor(X, dtype=torch.float32)

model = SoftmaxClassifier(checkpoint["latent_dim"], checkpoint["n_classes"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    logits = model(X)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

muon_mask = df["muon_mask"].to_numpy(dtype=bool)
muon_idx = label_to_int["muon"]
preds[muon_mask] = muon_idx
probs[muon_mask, :] = 0.0
probs[muon_mask, muon_idx] = 1.0

for label, idx in label_to_int.items():
    df[f"{label}_score"] = probs[:, idx]

df["prediction_int"] = preds
df["prediction"] = [int_to_label[p] for p in preds]

out_file = "classified_electrones_hough.csv"
df.to_csv(out_file, index=False)
print(f"Saved: {out_file}")
