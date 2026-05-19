# analyze_latent_space.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# ============================================================
# CONFIG
# ============================================================

ELECTRON_CSV = "resultados_electron_testbeam.csv"
#MUON_CSV = "resultados_muon_test.csv"
#PION_CSV = "resultados_pion_test.csv"

BASE = Path.home() / "/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder"

CLASSIFIER_PATH = BASE / "best_classifier.pkl"
SCALER_PATH = BASE / "scaler.pkl"

OUTPUT_DIR = "testbeam_electron_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading CSVs...")


df_e = pd.read_csv(ELECTRON_CSV)
#df_m = pd.read_csv(MUON_CSV)
#df_p = pd.read_csv(PION_CSV)


df = df_e #pd.concat([
    #df_e,
    #df_m,
    #df_p
#], ignore_index=True)

print("Total events:", len(df))

df["label"]= df["label"].replace({"piones":"pion"})

labels= ["electron","muon","pion"]
# ============================================================
# FEATURES
# ============================================================

feature_cols = [c for c in df.columns if c.startswith("f")]

X = df[feature_cols].values

y_true = df["label"].values

clusters = df["cluster"].values

mse = df["mse"].values


# ============================================================
# LOAD CLASSIFIER
# ============================================================

print("Loading classifier...")

clf = joblib.load(CLASSIFIER_PATH)
scaler = joblib.load(SCALER_PATH)

X_scaled = scaler.transform(X)

print("Predicting...")

y_pred = clf.predict(X_scaled)


# ============================================================
# 1. LATENT SPACE BY TRUE LABELS
# ============================================================

print("Plotting latent space by labels...")

plt.figure(figsize=(8,6))

for label in labels:

    mask = y_true == label

    plt.scatter(
        X[mask,0],
        X[mask,1],
        s=5,
        alpha=0.4,
        label=label
    )

plt.xlabel("Latent feature 0")
plt.ylabel("Latent feature 1")
plt.title("Latent space by particle")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "latent_by_particle.png"),
    dpi=300
)

plt.close()


# ============================================================
# 2. LATENT SPACE BY CLUSTER
# ============================================================

print("Plotting clusters...")

plt.figure(figsize=(8,6))

for cid in np.unique(clusters):

    mask = clusters == cid

    plt.scatter(
        X[mask,0],
        X[mask,1],
        s=5,
        alpha=0.4,
        label=f"cluster {cid}"
    )

plt.xlabel("Perfil longitudinal")
plt.ylabel("Perfil lateral")
plt.title("Latent space by GMM clusters")
plt.legend(ncol=2)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "latent_by_cluster.png"),
    dpi=300
)

plt.close()


# ============================================================
# 3. CONFUSION MATRIX
# ============================================================
classes= np.unique(np.concatenate([y_true, y_pred]))

print("Plotting confusion matrix...")

cm = confusion_matrix(y_true, y_pred,labels= labels)

fig, ax = plt.subplots(figsize=(7,6))


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)


disp.plot(ax=ax)

plt.title("Classifier confusion matrix")

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=300
)

plt.close()


# ============================================================
# 4. MSE DISTRIBUTION
# ============================================================

print("Plotting MSE distributions...")

plt.figure(figsize=(8,6))

mse_max= np.percentile(mse, 99)

for label in labels:

    mask = y_true == label

    plt.hist(
        mse[mask],
        bins=120,
        alpha=0.5,
        density=True,
        label=label
    )
plt.xlim(0,mse_max)

plt.xlabel("Reconstruction MSE")
plt.ylabel("Density")
plt.title("MSE distributions by particle")
plt.legend()
plt.yscale("log")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "mse_by_particle.png"),
    dpi=300
)

plt.close()


# ============================================================
# 5. OUTLIERS
# ============================================================

print("Finding outliers...")

threshold = np.percentile(mse, 99)

outliers = mse > threshold

print("Outlier threshold:", threshold)
print("Number of outliers:", np.sum(outliers))


# ============================================================
# 6. OUTLIERS IN LATENT SPACE
# ============================================================

print("Plotting outliers...")

plt.figure(figsize=(8,6))

plt.scatter(
    X[:,0],
    X[:,1],
    s=3,
    alpha=0.1,
    label="Normal"
)

plt.scatter(
    X[outliers,0],
    X[outliers,1],
    s=10,
    alpha=0.8,
    label="Outliers"
)

plt.xlabel("Latent feature 0")
plt.ylabel("Latent feature 1")
plt.title("Outliers in latent space")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "latent_outliers.png"),
    dpi=300
)

plt.close()


# ============================================================
# 7. OUTLIER STATISTICS
# ============================================================

print("\n==============================")
print("OUTLIER STATISTICS")
print("==============================")

outlier_df = df[outliers]

print("\nParticles in outliers:")
print(outlier_df["label"].value_counts())

print("\nClusters in outliers:")
print(outlier_df["cluster"].value_counts())


# ============================================================
# 8. WRONG CLASSIFICATIONS
# ============================================================

print("Analyzing classifier mistakes...")

wrong = y_true != y_pred

print("Wrong predictions:", np.sum(wrong))

plt.figure(figsize=(8,6))

plt.scatter(
    X[:,0],
    X[:,1],
    s=3,
    alpha=0.1,
    label="Correct"
)

plt.scatter(
    X[wrong,0],
    X[wrong,1],
    s=10,
    alpha=0.8,
    label="Misclassified"
)

plt.xlabel("Latent feature 0")
plt.ylabel("Latent feature 1")
plt.title("Classifier mistakes in latent space")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "classifier_mistakes.png"),
    dpi=300
)

plt.close()


# ============================================================
# 9. OUTLIERS VS CLASSIFIER ERRORS
# ============================================================

print("\n==============================")
print("OUTLIERS VS CLASSIFIER ERRORS")
print("==============================")

both = outliers & wrong

print("Events that are BOTH outliers and misclassified:")
print(np.sum(both))

print("Fraction:")
print(np.sum(both) / np.sum(outliers))


# ============================================================
# 10. SAVE OUTLIER TABLE
# ============================================================

print("Saving outlier table...")

outlier_table = df[outliers].copy()
outlier_table["predicted"] = y_pred[outliers]

outlier_table.to_csv(
    os.path.join(OUTPUT_DIR, "outliers.csv"),
    index=False
)

# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n==============================")
print("ANALYSIS FINISHED")
print("==============================")

print("Results saved in:")
print(OUTPUT_DIR)

print("\nGenerated plots:")
print("- latent_by_particle.png")
print("- latent_by_cluster.png")
print("- confusion_matrix.png")
print("- mse_by_particle.png")
print("- latent_outliers.png")
print("- classifier_mistakes.png")

print("\nGenerated tables:")
print("- outliers.csv")
