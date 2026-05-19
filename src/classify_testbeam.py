# src/classify_testbeam.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIG
# ==========================================================

CSV_PATHS = [

    # ELECTRON CONTAMINATED
    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electron_testbeam_test.csv",

    # PION CONTAMINATED
    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_testbeam_test.csv",
]


CLASSIFIER_PATH = (
    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/"
    "best_classifier.pkl"
)

SCALER_PATH = (
    "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/"
    "scaler.pkl"
)

OUTPUT_DIR = "testbeam_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# LOAD CSVs
# ==========================================================

print("Loading CSVs...")

dfs = []

for path in CSV_PATHS:

    print(f"Loading: {path}")

    df = pd.read_csv(path)

    df["source_file"] = os.path.basename(path)

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

print(f"Total events: {len(df)}")


# ==========================================================
# LATENT FEATURES
# ==========================================================

latent_cols = [
    c for c in df.columns
    if c.startswith("f")
]

print(f"Latent dimensions: {len(latent_cols)}")


X = df[latent_cols].values


# ==========================================================
# LOAD SCALER + CLASSIFIER
# ==========================================================

print("Loading scaler...")

scaler = joblib.load(SCALER_PATH)

print("Loading classifier...")

classifier = joblib.load(CLASSIFIER_PATH)


# ==========================================================
# SCALE
# ==========================================================

X_scaled = scaler.transform(X)


# ==========================================================
# PREDICT
# ==========================================================

print("Predicting...")

preds = classifier.predict(X_scaled)

print("Unique predictions:")
print(np.unique(preds))

df["prediction"] = preds


# ==========================================================
# OPTIONAL: SVC SCORES
# ==========================================================

if hasattr(classifier, "decision_function"):

    print("Computing decision scores...")

    scores = classifier.decision_function(X_scaled)

    if len(scores.shape) == 1:

        df["svc_score"] = scores

    else:

        for i in range(scores.shape[1]):

            df[f"score_class_{i}"] = scores[:, i]


# ==========================================================
# LABEL MAP
# ==========================================================

df["prediction_name"] = df["prediction"]


# ==========================================================
# SAVE CSV
# ==========================================================

output_csv = os.path.join(
    OUTPUT_DIR,
    "classified_testbeam.csv"
)

df.to_csv(output_csv, index=False)

print(f"\nSaved classified CSV:")
print(output_csv)


# ==========================================================
# CLASS DISTRIBUTION
# ==========================================================

print("\nPrediction distribution:")

print(
    df["prediction_name"]
    .value_counts()
)

print("\nFractions:")

print(
    df["prediction_name"]
    .value_counts(normalize=True)
)


# ==========================================================
# PLOT GLOBAL DISTRIBUTION
# ==========================================================

plt.figure(figsize=(7, 5))

df["prediction_name"].value_counts().plot(
    kind="bar"
)

plt.ylabel("Events")

plt.title("Predicted classes in test beam")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "prediction_distribution.png"
    )
)

plt.close()


# ==========================================================
# DISTRIBUTION PER FILE
# ==========================================================

for source_name in df["source_file"].unique():

    sub = df[
        df["source_file"] == source_name
    ]

    plt.figure(figsize=(7, 5))

    sub["prediction_name"].value_counts().plot(
        kind="bar"
    )

    plt.ylabel("Events")

    plt.title(source_name)

    plt.tight_layout()

    outname = (
        source_name.replace(".csv", "")
        + "_distribution.png"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            outname
        )
    )

    plt.close()


# ==========================================================
# LATENT SPACE PLOT
# ==========================================================

if "latent_0" in df.columns and "latent_1" in df.columns:

    plt.figure(figsize=(8, 7))

    for cls in sorted(df["prediction"].unique()):

        mask = df["prediction"] == cls

        plt.scatter(
            df.loc[mask, "latent_0"],
            df.loc[mask, "latent_1"],
            s=5,
            alpha=0.5,
            label=label_map[cls]
        )

    plt.xlabel("latent_0")
    plt.ylabel("latent_1")

    plt.legend()

    plt.title("Latent space classified")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "latent_space.png"
        )
    )

    plt.close()


print("\nDone.")
