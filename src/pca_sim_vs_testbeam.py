# pca_sim_vs_testbeam.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

os.makedirs("debug_pca", exist_ok=True)

# =========================
# LOAD SIMULATION
# =========================

sim_e = pd.read_csv("resultados_electron_train.csv")
sim_p = pd.read_csv("resultados_pion_train.csv")
sim_m = pd.read_csv("resultados_muon_train.csv")

sim_e["domain"] = "electron_sim"
sim_p["domain"] = "pion_sim"
sim_m["domain"] = "muon_sim"

df_sim = pd.concat([sim_e, sim_p, sim_m], ignore_index=True)

# =========================
# LOAD TESTBEAM
# =========================

tb_e1 = pd.read_csv("/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_20_test_testbeam2/electrones_20_test2_features.csv")
tb_e2 = pd.read_csv("/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_50_test_testbeam2/electrones_50_test2_features.csv")
tb_e3 = pd.read_csv("/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/electrones_80_test_testbeam2/electrones_80_test2_features.csv")

tb_p1 = pd.read_csv("home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_20_test_testbeam2/piones_20_test2_features.csv")
tb_p2 = pd.read_csv("home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_50_test_testbeam2/piones_50_test2_features.csv")
tb_p3 = pd.read_csv("/home/vmellado/FQM378/vmellado/GATrEnv/GATrAutoencoder/piones_80_test_testbeam2/piones_80_test2_features.csv")


tb_e1["domain"] = "electron_20"
tb_e2["domain"] = "electron_50"
tb_e3["domain"] = "electron_80"
tb_p1["domain"] = "piones_20"
tb_p2["domain"] = "piones_50"
tb_p3["domain"] = "piones_80"

tb_e = pd.concat([tb_e1, tb_e2, tb_e3], ignore_index=True)
tb_p = pd.concat([tb_p1, tb_p2, tb_p3], ignore_index=True)

df_tb = pd.concat([tb_e, tb_p], ignore_index=True)

# =========================
# COMBINE
# =========================

feature_cols = sorted([c for c in df_sim.columns if c.startswith("f")])

df_all = pd.concat([
    df_sim[feature_cols + ["domain"]],
    df_tb[feature_cols + ["domain"]]
], ignore_index=True)

X = df_all[feature_cols].values

scaler = joblib.load("scaler.pkl")
X_scaled = scaler.transform(X)

# =========================
# PCA
# =========================

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df_all["pc1"] = X_pca[:, 0]
df_all["pc2"] = X_pca[:, 1]

df_all.to_csv("debug_pca/pca_projection.csv", index=False)

# =========================
# PLOT
# =========================

plt.figure(figsize=(9,7))

colors = {
    "sim_electron": "tab:blue",
    "sim_muon": "tab:green",
    "sim_pion": "tab:red",
    "tb_electron": "cyan",
    "tb_pion": "orange",
}

for dom in df_all["domain"].unique():
    mask = df_all["domain"] == dom
    plt.scatter(
        df_all.loc[mask, "pc1"],
        df_all.loc[mask, "pc2"],
        s=8,
        alpha=0.35,
        label=dom,
        c=colors.get(dom, None)
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA projection: simulation vs testbeam")
plt.legend(markerscale=2)
plt.tight_layout()
plt.savefig("debug_pca/pca_sim_vs_testbeam.png", dpi=250)
plt.close()

print("Explained variance:", pca.explained_variance_ratio_)
print("Saved in debug_pca/")
