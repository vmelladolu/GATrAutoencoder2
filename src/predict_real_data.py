import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# ==========================================================
# LOAD REAL DATA
# ==========================================================

real_csvs = [

    "electrones_testbeam_50_test.csv",
    "electrones_testbeam_80_test.csv",
    "electrones_testbeam_20_test.csv",
    "piones_testbeam_20_test.csv",

]

dfs = []

for path in real_csvs:

    print(f"Loading: {path}")

    dfs.append(
        pd.read_csv(path)
    )

df = pd.concat(
    dfs,
    ignore_index=True
)

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

out_file = "classified_testbeam.csv"

df.to_csv(
    out_file,
    index=False
)

print(f"Saved: {out_file}")
