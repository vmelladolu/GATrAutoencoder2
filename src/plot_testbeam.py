import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
sns.set_context("talk")

# ==========================================================
# LOAD CSV
# ==========================================================

CSV_FILE = "classified_testbeam_piones.csv"

print(f"Loading {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

print(df.head())
print(df.columns)

# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

OUTDIR = "testbeam_piones_plots"

os.makedirs(
    OUTDIR,
    exist_ok=True
)

print(f"\nSaving plots to: {OUTDIR}")

# ==========================================================
# AUTO-DETECT ENERGY / BEAM INFO
# ==========================================================

# Si no existen estas columnas, intentamos crearlas

if "beam_energy" not in df.columns:

    if "mc_energy" in df.columns:

        df["beam_energy"] = df["mc_energy"]

    else:

        df["beam_energy"] = -1


if "beam_type" not in df.columns:

    df["beam_type"] = "unknown"

# ==========================================================
# BASIC INFO
# ==========================================================

print("\n================ DATASET INFO ================")
print(df.describe())

print("\nPredictions:")
print(df["prediction"].value_counts())

# ==========================================================
# PREDICTION COUNTS
# ==========================================================

plt.figure(figsize=(8,6))

counts = df["prediction"].value_counts()

sns.barplot(
    x=counts.index,
    y=counts.values
)

plt.ylabel("Events")
plt.xlabel("Predicted particle")
plt.title("Predicted particle distribution")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTDIR, "prediction_counts.png"),
    dpi=300
)
plt.close()

# ==========================================================
# PREDICTION FRACTIONS
# ==========================================================

plt.figure(figsize=(7,7))

plt.pie(
    counts.values,
    labels=counts.index,
    autopct="%.1f%%"
)

plt.title("Particle composition")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTDIR, "prediction_fractions.png"),
    dpi=300
)
plt.close()

# ==========================================================
# SCORE DISTRIBUTIONS
# ==========================================================

score_columns = [
    c for c in df.columns
    if c.endswith("_score")
]

print("\nScore columns:")
print(score_columns)

for score_col in score_columns:

    plt.figure(figsize=(8,6))

    sns.histplot(
        df[score_col],
        bins=80,
        stat="density",
        kde=True
    )

    plt.xlabel(score_col)
    plt.ylabel("Density")

    plt.title(
        f"{score_col} distribution"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTDIR, f"{score_col}_distribution.png"),
        dpi=300
    )

    plt.close()

# ==========================================================
# SCORE COMPARISON BY PREDICTION
# ==========================================================

for score_col in score_columns:

    plt.figure(figsize=(9,6))

    sns.kdeplot(
        data=df,
        x=score_col,
        hue="prediction",
        fill=True,
        common_norm=False,
        alpha=0.3
    )

    plt.title(
        f"{score_col} by predicted particle"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTDIR, f"{score_col}_by_prediction.png"),
        dpi=300
    )

    plt.close()

# ==========================================================
# ENERGY DEPENDENCE
# ==========================================================

if "beam_energy" in df.columns:

    energies = sorted(df["beam_energy"].dropna().unique())

    print("\nBeam energies:")
    print(energies)

    for score_col in score_columns:

        plt.figure(figsize=(9,6))

        for energy in energies:

            subset = df[
                df["beam_energy"] == energy
            ]

            if len(subset) == 0:
                continue

            sns.kdeplot(
                subset[score_col],
                label=f"{energy} GeV"
            )

        plt.xlabel(score_col)
        plt.ylabel("Density")

        plt.title(
            f"{score_col} vs beam energy"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(OUTDIR, f"{score_col}_vs_energy.png"),
            dpi=300
        )

        plt.close()

# ==========================================================
# PREDICTIONS VS ENERGY
# ==========================================================

if "beam_energy" in df.columns:

    plt.figure(figsize=(10,6))

    pred_vs_energy = pd.crosstab(
        df["beam_energy"],
        df["prediction"],
        normalize="index"
    )

    pred_vs_energy.plot(
        kind="bar",
        stacked=True,
        figsize=(10,6)
    )

    plt.ylabel("Fraction")
    plt.xlabel("Beam energy [GeV]")

    plt.title(
        "Predicted particle composition vs beam energy"
    )

    plt.legend(title="Prediction")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTDIR, "prediction_vs_energy.png"),
        dpi=300
    )

    plt.close()

# ==========================================================
# CONFIDENCE DISTRIBUTION
# ==========================================================

score_matrix = df[score_columns].values

max_score = np.max(score_matrix, axis=1)

plt.figure(figsize=(8,6))

sns.histplot(
    max_score,
    bins=80,
    kde=True,
    stat="density"
)

plt.xlabel("Maximum softmax score")
plt.ylabel("Density")

plt.title(
    "Classifier confidence"
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "classifier_confidence.png"),
    dpi=300
)

plt.close()

# ==========================================================
# CONFIDENCE BY PARTICLE
# ==========================================================

plt.figure(figsize=(9,6))

sns.boxplot(
    x=df["prediction"],
    y=max_score
)

plt.xlabel("Predicted particle")
plt.ylabel("Maximum softmax score")

plt.title(
    "Classifier confidence by predicted particle"
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "confidence_by_particle.png"),
    dpi=300
)

plt.close()

# ==========================================================
# CORRELATION MATRIX
# ==========================================================

corr_cols = score_columns.copy()

if "beam_energy" in df.columns:
    corr_cols.append("beam_energy")

corr = df[corr_cols].corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    center=0
)

plt.title(
    "Feature correlation matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "correlation_matrix.png"),
    dpi=300
)

plt.close()

# ==========================================================
# SCATTER PLOTS
# ==========================================================

if len(score_columns) >= 2:

    plt.figure(figsize=(8,8))

    sns.scatterplot(
        data=df.sample(
            min(20000, len(df))
        ),
        x=score_columns[0],
        y=score_columns[1],
        hue="prediction",
        alpha=0.5,
        s=10
    )

    plt.title(
        "Softmax latent separation"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTDIR, "softmax_scatter.png"),
        dpi=300
    )

    plt.close()

# ==========================================================
# EVENT PURITY ESTIMATE
# ==========================================================

thresholds = np.linspace(0, 1, 50)

for score_col in score_columns:

    efficiencies = []

    scores = df[score_col].values

    for thr in thresholds:

        efficiencies.append(
            np.mean(scores > thr)
        )

    plt.figure(figsize=(8,6))

    plt.plot(
        thresholds,
        efficiencies
    )

    plt.xlabel("Score threshold")
    plt.ylabel("Fraction of accepted events")

    plt.title(
        f"Acceptance curve: {score_col}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTDIR, f"acceptance_{score_col}.png"),
        dpi=300
    )

    plt.close()

print("\n========================================")
print("Saved all plots successfully")
print(f"Plots directory: {OUTDIR}")
print("========================================")
