"""Utilidades de datos para entrenar/evaluar un clasificador sobre el latente
del autoencoder GATr.

Punto clave: el AE se entrenó con stats de z_norm COMBINADAS de los 3 ficheros
(MultiFileFlatSDHCALDataset). Para que el latente quede DENTRO de distribución,
tanto la cabeza neuronal como el SVC deben usar exactamente esas mismas stats
combinadas — calculadas una vez sobre los *_train.h5 y reutilizadas para los
*_test.h5. (evaluate_autoencoder.py normaliza por fichero, lo cual NO es
consistente con el entrenamiento.)
"""

import os
import numpy as np
import torch

from .datasets import (
    FlatSDHCALDataset,
    MultiFileFlatSDHCALDataset,
    _compute_stats_from_dataset_subset,
)

# Mapeo canónico clase -> entero (coincide con evaluate_autoencoder.py)
LABEL_INT_TO_NAME = {0: "electron", 1: "pion", 2: "muon"}
NAME_TO_LABEL_INT = {v: k for k, v in LABEL_INT_TO_NAME.items()}


def compute_or_load_combined_stats(train_paths, norm_type, out_yaml,
                                   filters=None):
    """Calcula (o carga) stats de normalización COMBINADAS sobre los *_train.h5.

    Replica la normalización que vio el AE al entrenarse con los 3 ficheros a la
    vez. Guarda/lee de ``out_yaml`` para reutilizar entre train y eval.

    Returns:
        stats (dict) en el mismo formato que usa ``_apply_preprocessing_inplace``.
    """
    import yaml as _yaml

    if out_yaml and os.path.exists(out_yaml):
        with open(out_yaml, "r") as f:
            raw = _yaml.safe_load(f)
        stats = raw.get("stats", raw) if isinstance(raw, dict) else raw
        print(f"[clf_data] Stats combinadas cargadas de '{out_yaml}'")
        return stats

    print(f"[clf_data] Calculando stats combinadas desde {len(train_paths)} ficheros de train...")
    ds = MultiFileFlatSDHCALDataset(train_paths, preprocessing_cfg=None,
                                    filters=filters)
    all_idx = np.arange(ds.len())
    stats = _compute_stats_from_dataset_subset(ds, all_idx)

    if out_yaml:
        os.makedirs(os.path.dirname(os.path.abspath(out_yaml)), exist_ok=True)
        with open(out_yaml, "w") as f:
            _yaml.dump({"norm_type": norm_type, "stats": stats}, f,
                       default_flow_style=False)
        print(f"[clf_data] Stats combinadas guardadas en '{out_yaml}'")
    return stats


class LabeledFlatDataset(torch.utils.data.Dataset):
    """Envuelve un FlatSDHCALDataset y adjunta ``data.y = label`` por evento."""

    def __init__(self, ds, label, indices=None):
        self.ds = ds
        self.label = int(label)
        self.indices = (np.arange(ds.len()) if indices is None
                        else np.asarray(indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data = self.ds.get(int(self.indices[idx]))
        data.y = torch.tensor(self.label, dtype=torch.long)
        return data


def build_labeled_datasets(paths_with_labels, stats, norm_type,
                           preprocessing_cfg, max_events=0, seed=42):
    """Carga cada fichero de partícula, aplica las stats COMBINADAS y adjunta y.

    Args:
        paths_with_labels: lista de (path, label_int, label_name).
        stats:             stats combinadas (de compute_or_load_combined_stats).
        norm_type:         "z_norm" | "minmax" | None.
        preprocessing_cfg: dict con use_scalar/use_one_hot/...
        max_events:        límite por clase (0 = todos).

    Returns:
        (concat_dataset, per_class) donde per_class es lista de dicts
        {label_int, label_name, n_events}.
    """
    labeled = []
    per_class = []
    for path, label_int, label_name in paths_with_labels:
        if path is None:
            continue
        ds = FlatSDHCALDataset(path, preprocessing_cfg=None, filters=None)
        if norm_type in ("z_norm", "minmax"):
            ds._apply_preprocessing_inplace(stats, norm_type, preprocessing_cfg)

        idx = np.arange(ds.len())
        if max_events > 0 and max_events < len(idx):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, size=max_events, replace=False))

        labeled.append(LabeledFlatDataset(ds, label_int, indices=idx))
        per_class.append({"label_int": label_int, "label_name": label_name,
                          "n_events": len(idx)})
        print(f"[clf_data] {label_name}: {len(idx)} eventos")

    if not labeled:
        raise ValueError("No se proporcionó ningún fichero de datos.")

    concat = torch.utils.data.ConcatDataset(labeled)
    return concat, per_class


def class_weights_from_counts(per_class, n_classes):
    """Pesos inversamente proporcionales a la frecuencia (para CrossEntropy)."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for c in per_class:
        counts[c["label_int"]] = c["n_events"]
    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    w = total / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)
