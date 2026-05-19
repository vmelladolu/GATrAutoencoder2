import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from glob import glob

# ==========================================================
# LOAD TRAIN CSVs
# ==========================================================

train_csvs = [

    "resultados_electron_train.csv",
    "resultados_pion_train.csv",
    "resultados_muon_train.csv",

]

train_dfs = []

for path in train_csvs:

    print(f"Loading train: {path}")

    train_dfs.append(
        pd.read_csv(path)
    )

train_df = pd.concat(
    train_dfs,
    ignore_index=True
)

# ==========================================================
# LOAD TEST CSVs
# ==========================================================

test_csvs = [

    "resultados_electron_test.csv",
    "resultados_pion_test.csv",
    "resultados_muon_test.csv",

]

test_dfs = []

for path in test_csvs:

    print(f"Loading test: {path}")

    test_dfs.append(
        pd.read_csv(path)
    )

test_df = pd.concat(
    test_dfs,
    ignore_index=True
)

# ==========================================================
# LATENT COLUMNS
# ==========================================================

latent_cols = sorted(
    [c for c in train_df.columns if c.startswith("f")],
    key=lambda x: int(x[1:])
)

print("Latent dimensions:", len(latent_cols))

# ==========================================================
# NORMALIZE LABEL NAMES
# ==========================================================

NORMALIZATION = {
    "electrones": "electron",
    "electrones": "electron",
    "electron": "electron",

    "muones": "muon",
    "muon": "muon",

    "pion": "pion",
    "piones": "pion",
}

train_df["label"] = train_df["label"].astype(str).str.lower().map(NORMALIZATION)
test_df["label"] = test_df["label"].astype(str).str.lower().map(NORMALIZATION)

# ==========================================================
# LABEL ENCODING
# ==========================================================

all_labels = sorted(
    train_df["label"].unique()
)

label_to_int = {
    label: i
    for i, label in enumerate(all_labels)
}

print("Label mapping:")
print(label_to_int)

# Convertir labels a enteros
train_df["label"] = train_df["label"].map(label_to_int)
test_df["label"] = test_df["label"].map(label_to_int)

# ==========================================================
# FEATURES
# ==========================================================

X_train = train_df[latent_cols].values.astype(np.float32)
y_train = train_df["label"].values.astype(np.int64)

X_test = test_df[latent_cols].values.astype(np.float32)
y_test = test_df["label"].values.astype(np.int64)

# ==========================================================
# SCALE
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    "softmax_scaler.pkl"
)

# ==========================================================
# TORCH
# ==========================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
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

model = SoftmaxClassifier(
    input_dim=len(latent_cols),
    n_classes=len(np.unique(y_train.numpy()))
)

# ==========================================================
# TRAINING
# ==========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

EPOCHS = 30

for epoch in range(EPOCHS):

    model.train()

    logits = model(X_train)

    loss = criterion(
        logits,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    preds = torch.argmax(
        logits,
        dim=1
    )

    acc = (
        preds == y_train
    ).float().mean()

    print(
        f"Epoch {epoch+1} | "
        f"loss={loss.item():.4f} | "
        f"acc={acc.item():.4f}"
    )

# ==========================================================
# TEST
# ==========================================================

model.eval()

with torch.no_grad():

    logits = model(X_test)

    probs = torch.softmax(
        logits,
        dim=1
    )

    preds = torch.argmax(
        probs,
        dim=1
    )

print(
    classification_report(
        y_test.numpy(),
        preds.numpy()
    )
)

# ==========================================================
# SAVE
# ==========================================================

torch.save({

    "model_state_dict":
        model.state_dict(),

    "latent_dim":
        len(latent_cols),

    "n_classes":
        len(np.unique(y_train.numpy()))

}, "softmax_classifier.pt")

print("Saved softmax_classifier.pt")
