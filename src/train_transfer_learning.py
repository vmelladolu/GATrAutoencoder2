# src/train_transfer_classifier.py

import os
import torch
import torch.nn as nn
import numpy as np

from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, classification_report

from models.gatr_autoencoder import GATrAutoencoder
from utils.datasets import make_pf_splits
from utils.batch_utils import build_batch


# ==========================================================
# CONFIG
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUTOENCODER_CHECKPOINT = (
    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/checkpoint_best_epoch0402_loss0.078888.pt"
)

MODEL_CFG = "config/model_cfg.yml"

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3

N_CLASSES = 3

LABEL_MAP = {
    "electrones": 0,
    "piones": 1,
    "muones": 2,
}


# ==========================================================
# DATASETS
# ==========================================================

TRAIN_DATA_PATHS = [

    # ELECTRONES
    "/media/FQM378/vmellado/GATrEnv/GATrAutoencoder/resultados_electron_train.csv",

    # PIONES
    "/media/FQM378/vmellado/GATrEnv/GATrAutoencoder/resultados_pion_train.csv"

    # MUONES
    "/media/FQM378/vmellado/GATrEnv/GATrAutoencoder/resultados_muon_train.csv",
]


TEST_DATA_PATHS = [

    # ELECTRONES
    "/media/FQM378/vmellado/GATrEnv/data/Datos/electron_testbeam_test.csv",

    # PIONES
    "/media/FQM378/vmellado/GATrEnv/data/Datos/piones_testbeam_test.csv",
]


# ==========================================================
# MODEL
# ==========================================================

class TransferClassifier(nn.Module):

    def __init__(self, autoencoder, latent_dim=64, n_classes=3):

        super().__init__()

        self.autoencoder = autoencoder

        # freeze autoencoder
        for p in self.autoencoder.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(

            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, n_classes)
        )

    def forward(
        self,
        mv_v_part,
        mv_s_part,
        scalars,
        batch
    ):

        with torch.no_grad():

            outputs = self.autoencoder(
                mv_v_part,
                mv_s_part,
                scalars,
                batch
            )

            z = outputs["aggregate_latent"]

        logits = self.classifier(z)

        return logits


# ==========================================================
# LOAD YAML CONFIG
# ==========================================================

import yaml

cfg_models = yaml.safe_load(open(MODEL_CFG, "r"))

cfg_enc = cfg_models["encoder"]
cfg_dec = cfg_models["decoder"]
cfg_agg = cfg_models["aggregation"]


# ==========================================================
# LOAD AUTOENCODER
# ==========================================================

print("Loading autoencoder...")

autoencoder = GATrAutoencoder(
    cfg_enc=cfg_enc,
    cfg_dec=cfg_dec,
    cfg_agg=cfg_agg,
    latent_s_channels=2
)

checkpoint = torch.load(
    AUTOENCODER_CHECKPOINT,
    map_location=DEVICE
)

autoencoder.load_state_dict(
    checkpoint["model_state_dict"],strict=False
)

autoencoder.to(DEVICE)

autoencoder.eval()

print("Autoencoder loaded")


# ==========================================================
# CREATE CLASSIFIER
# ==========================================================

LATENT_DIM = (
    cfg_enc["out_mv_channels"] * 16
    +
    cfg_enc["out_s_channels"]
)

print(f"Latent dim = {LATENT_DIM}")

model = TransferClassifier(
    autoencoder=autoencoder,
    latent_dim=LATENT_DIM,
    n_classes=N_CLASSES
)

model.to(DEVICE)

print(model)


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading train dataset...")

train_dataset, _ = make_pf_splits(
    TRAIN_DATA_PATHS,
    val_ratio=0.0,
    mode="lazy"
)

print("Loading test dataset...")

test_dataset, _ = make_pf_splits(
    TEST_DATA_PATHS,
    val_ratio=0.0,
    mode="lazy"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Train events: {len(train_dataset)}")
print(f"Test events: {len(test_dataset)}")


# ==========================================================
# LOSS + OPTIMIZER
# ==========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LR
)


# ==========================================================
# TRAINING
# ==========================================================

print("Starting training...")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    correct = 0
    total = 0

    for batch in train_loader:

        data = build_batch(
            batch,
            use_scalar=False,
            use_one_hot=False,
            use_energy=False,
            z_norm=False
        )

        mv_v_part = data["mv_v_part"].to(DEVICE)
        mv_s_part = data["mv_s_part"].to(DEVICE)
        scalars = data["scalars"].to(DEVICE)
        batch_idx = data["batch_idx"].to(DEVICE)

        labels = batch.y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            mv_v_part,
            mv_s_part,
            scalars,
            batch_idx
        )

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()

        total += len(labels)

    acc = correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"loss={running_loss:.4f} | "
        f"acc={acc:.4f}"
    )


# ==========================================================
# TEST
# ==========================================================

print("\nEvaluating on test set...")

model.eval()

all_preds = []
all_labels = []

all_probs = []

with torch.no_grad():

    for batch in test_loader:

        data = build_batch(
            batch,
            use_scalar=False,
            use_one_hot=False,
            use_energy=False,
            z_norm=False
        )

        mv_v_part = data["mv_v_part"].to(DEVICE)
        mv_s_part = data["mv_s_part"].to(DEVICE)
        scalars = data["scalars"].to(DEVICE)
        batch_idx = data["batch_idx"].to(DEVICE)

        labels = batch.y.to(DEVICE)

        logits = model(
            mv_v_part,
            mv_s_part,
            scalars,
            batch_idx
        )

        probs = torch.softmax(logits, dim=1)

        preds = torch.argmax(probs, dim=1)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )


# ==========================================================
# METRICS
# ==========================================================

acc = accuracy_score(
    all_labels,
    all_preds
)

print(f"\nTEST ACCURACY = {acc:.4f}\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[
            "electron",
            "pion",
            "muon"
        ]
    )
)


# ==========================================================
# SAVE MODEL
# ==========================================================

torch.save(
    model.state_dict(),
    "transfer_classifier.pt"
)

print("\nSaved model:")
print("transfer_classifier.pt")


# ==========================================================
# SAVE SOFTMAX SCORES
# ==========================================================

import pandas as pd

df = pd.DataFrame()

df["true_label"] = all_labels
df["prediction"] = all_preds

all_probs = np.array(all_probs)

df["electron_score"] = all_probs[:, 0]
df["pion_score"] = all_probs[:, 1]
df["muon_score"] = all_probs[:, 2]

df.to_csv(
    "transfer_classifier_scores.csv",
    index=False
)

print("\nSaved scores:")
print("transfer_classifier_scores.csv")
