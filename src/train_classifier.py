import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# cargar datasets
df = pd.concat([
    pd.read_csv("resultados_electron_train.csv"),
    pd.read_csv("resultados_pion_train.csv"),
    pd.read_csv("resultados_muon_train.csv")
])

# añadir etiqueta manual
df["label"] = (
    ["electron"] * len(pd.read_csv("resultados_electron_train.csv")) +
    ["pion"] * len(pd.read_csv("resultados_pion_train.csv")) +
    ["muon"] * len(pd.read_csv("resultados_muon_train.csv"))
)

feature_cols = [c for c in df.columns if c.startswith("f")]

X = df[feature_cols].values
y = df["label"].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ============================================================
# Escalado
# ============================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ============================================================
# Modelos
# ============================================================

models = {
    "svc": SVC(kernel="rbf", probability=True),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "decision_tree": DecisionTreeClassifier(random_state=42),
}

results = {}

for name, model in models.items():

    print(f"Training {name}")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"{name} accuracy = {acc:.4f}")

    results[name]= acc

best_model_name = max(results, key=results.get)

best_model = models[best_model_name]

joblib.dump(best_model, "best_classifier.pkl")
joblib.dump(scaler,"scaler.pkl")
plt.figure(figsize=(8,6))

plt.bar(results.keys(), results.values())

plt.ylabel("Accuracy")

plt.title("Classifier comparison")

plt.tight_layout()

plt.savefig("classifier_comparison.png")
