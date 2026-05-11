import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# cargar datasets
df = pd.concat([
    pd.read_csv("electron_train_features.csv"),
    pd.read_csv("pion_train_features.csv"),
    pd.read_csv("muon_train_features.csv")
])

# añadir etiqueta manual
df["label"] = (
    ["electron"] * len(pd.read_csv("electron_train_features.csv")) +
    ["pion"] * len(pd.read_csv("pion_train_features.csv")) +
    ["muon"] * len(pd.read_csv("muon_train_features.csv"))
)

feature_cols = ["f0","f1"]

X = df[feature_cols].values
y = df["label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = SVC(kernel='rbf')
clf.fit(X_scaled, y)

joblib.dump(clf, "svc_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Modelo entrenado")
