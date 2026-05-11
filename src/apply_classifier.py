import pandas as pd
import joblib
from sklearn.metrics import classification_report

df = pd.concat([
    pd.read_csv("electron_test_features.csv"),
    pd.read_csv("pion_test_features.csv"),
    pd.read_csv("muon_test_features.csv")
])

df["label"] = (
    ["electron"] * len(pd.read_csv("electron_test_features.csv")) +
    ["pion"] * len(pd.read_csv("pion_test_features.csv")) +
    ["muon"] * len(pd.read_csv("muon_test_features.csv"))
)

feature_cols = [c for c in df.columns if c.startswith("f")] + ["mse"]

X = df[feature_cols].values
y_true = df["label"].values

clf = joblib.load("svc_model.pkl")
scaler = joblib.load("scaler.pkl")

X_scaled = scaler.transform(X)

y_pred = clf.predict(X_scaled)

print(classification_report(y_true, y_pred))
