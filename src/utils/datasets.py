from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import os
import h5py

from torch_geometric.loader import DataLoader
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import lightning as L


# ============================================================
# Dataset rápido para formato plano (flatten_npz.py)
# ============================================================

# Mapeo de strings a enteros (debe coincidir con STRING_TO_INT_MAPPINGS en process_flat_npz.py)
STRING_TO_INT_MAPPINGS = {
    "particle_type": {"electron": 0, "pion": 1, "muon": 2, "unknown": -1},
}


class SDHCALRegressorDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_paths,
        val_ratio,
        mode,
        preprocessing_cfg,
        filters_cfg,
        batch_size,
        train_num_workers,
        val_num_workers,
        seed=42,
        use_weighted_loss=False,
    ):
        super().__init__()
        self.data_paths = data_paths
        self.val_ratio = val_ratio
        self.mode = mode
        self.preprocessing_cfg = preprocessing_cfg
        self.filters_cfg = filters_cfg
        self.batch_size = batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.seed = seed
        self.train_dataset = None
        self.val_dataset = None
        self.class_weights = None
        self.use_weighted_loss = use_weighted_loss

    def setup(self, stage=None):
        if self.train_dataset is not None and self.val_dataset is not None:
            return
        self.train_dataset, self.val_dataset = make_pf_splits(
            self.data_paths,
            val_ratio=self.val_ratio,
            mode=self.mode,
            seed=self.seed,
            preprocessing_cfg=self.preprocessing_cfg,
            filters=self.filters_cfg,
            use_weighted_loss=self.use_weighted_loss,
        )
        base_dataset = self.train_dataset.dataset if hasattr(self.train_dataset, "dataset") else self.train_dataset
        self.class_weights = getattr(base_dataset, "weights", None)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_num_workers,
            persistent_workers=self.train_num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            persistent_workers=self.val_num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )


def _parse_filter_operator(value: str) -> tuple[str, float]:
    if value.startswith((">=", "<=")):
        op = value[:2]
        num = float(value[2:])
    elif value.startswith((">", "<")):
        op = value[0]
        num = float(value[1:])
    else:
        raise ValueError(f"Unsupported filter operator in '{value}'")
    return op, num


def _apply_filters_inmem(event_data: dict, n_events: int, filters: dict) -> np.ndarray:
    """Apply filters using pre-loaded in-memory event-level arrays."""
    if not filters:
        return np.ones(n_events, dtype=bool)

    mask = np.ones(n_events, dtype=bool)
    for filter_key, filter_value in filters.items():
        arr = event_data.get(filter_key)
        if arr is None:
            arr = event_data.get(f"filter_{filter_key}")
        if arr is None and filter_key == "status":
            arr = event_data.get("filter_status")
        if arr is None:
            raise KeyError(f"Filter key '{filter_key}' not found in dataset")
        arr = np.asarray(arr)
        if len(arr) != n_events:
            raise ValueError(
                f"Filter key '{filter_key}' has length {len(arr)} but expected {n_events}"
            )

        if isinstance(filter_value, str) and filter_value.startswith((">=", "<=", ">", "<")):
            op, num = _parse_filter_operator(filter_value)
            if op == ">=":
                mask &= arr >= num
            elif op == "<=":
                mask &= arr <= num
            elif op == ">":
                mask &= arr > num
            elif op == "<":
                mask &= arr < num
        else:
            compare_value = filter_value
            if filter_key in STRING_TO_INT_MAPPINGS and isinstance(filter_value, str):
                mapping = STRING_TO_INT_MAPPINGS[filter_key]
                compare_value = mapping.get(filter_value, filter_value)
            mask &= arr == compare_value

    return mask


# ============================================================
# Helpers de normalización / stats
# ============================================================

def _derive_stats_path(dataset_path: str) -> str:
    """Deriva la ruta del YAML de stats a partir del path del dataset.
    Ejemplo: /data/dataset.h5 → /data/dataset_stats.yml
    """
    from pathlib import Path
    p = Path(dataset_path)
    return str(p.parent / f"{p.stem}_stats.yml")


def _compute_stats_from_dataset_subset(
    dataset: "FlatSDHCALDataset",
    event_idx: np.ndarray,
) -> dict:
    """
    Calcula mean/std/min/max para los features hit-level usando SOLO los
    eventos indicados por ``event_idx`` (índices locales en el dataset,
    i.e. índices en ``dataset._event_indices``).

    Todos los arrays ya están cargados en RAM en ``dataset``.
    La asignación hit→evento se vectoriza con np.repeat sobre los offsets.
    """
    ev_global = dataset._event_indices[event_idx]  # global event indices in offsets

    # Build hit→event mapping (vectorized)
    offsets = dataset._offsets
    sizes = offsets[1:] - offsets[:-1]                           # (n_events_total,)
    hit_events = np.repeat(                                       # (n_hits_total,)
        np.arange(len(offsets) - 1, dtype=np.int32), sizes
    )
    ev_in_train = np.zeros(len(offsets) - 1, dtype=bool)
    ev_in_train[ev_global] = True
    hit_mask = ev_in_train[hit_events]                           # (n_hits_total,) bool
    hit_mask_t = torch.from_numpy(hit_mask)

    def _arr_stats(tensor: torch.Tensor) -> dict:
        arr = tensor[hit_mask_t].numpy().astype(np.float64)
        mean = float(arr.mean())
        std = float(arr.std())
        return {
            "mean": mean,
            "std": max(std, 1e-8),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "count": int(arr.size),
        }

    stats: dict = {
        "x":   _arr_stats(dataset._x),
        "y":   _arr_stats(dataset._y),
        "z":   _arr_stats(dataset._z),
        "i":   _arr_stats(dataset._i),
        "j":   _arr_stats(dataset._j),
        "k":   _arr_stats(dataset._k),
        "thr": _arr_stats(dataset._thr),
    }

    if dataset._time is not None:
        stats["time"] = _arr_stats(dataset._time)

    if dataset._energy is not None:
        en_arr = dataset._energy[ev_global].numpy().astype(np.float64)
        stats["energy"] = {
            "mean":  float(en_arr.mean()),
            "std":   max(float(en_arr.std()), 1e-8),
            "min":   float(en_arr.min()),
            "max":   float(en_arr.max()),
            "count": int(en_arr.size),
        }

    return stats


def _load_or_compute_stats(
    dataset: "FlatSDHCALDataset",
    train_idx: np.ndarray,
    preprocessing_cfg: dict,
    norm_type: str,
) -> dict:
    """
    Carga stats desde YAML si existe; si no, las calcula desde los índices
    de entrenamiento y las guarda en YAML.
    """
    import yaml as _yaml

    yaml_path = (
        preprocessing_cfg.get("norm_yaml_path")
        or preprocessing_cfg.get("z_norm_yaml_path")
    )
    if not yaml_path:
        yaml_path = _derive_stats_path(dataset._path)

    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            raw_s = _yaml.safe_load(f)
        stats = raw_s.get("stats", raw_s) if isinstance(raw_s, dict) else raw_s
        print(f"[Stats] Cargadas desde '{yaml_path}'")
        return stats

    print(
        f"[Stats] '{yaml_path}' no encontrado. "
        f"Calculando desde split de entrenamiento ({len(train_idx)} eventos)..."
    )
    stats = _compute_stats_from_dataset_subset(dataset, train_idx)
    os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
    with open(yaml_path, "w") as f:
        _yaml.dump({"norm_type": norm_type, "stats": stats}, f, default_flow_style=False)
    print(f"[Stats] Guardadas en '{yaml_path}'")
    return stats


class FlatSDHCALDataset(Dataset):
    """
    Dataset para formato plano HDF5 / NPZ.

    Todos los arrays (hit-level y event-level) se cargan en RAM durante
    ``__init__``.  El preprocesamiento (normalización + log-energy) se
    aplica UNA SOLA VEZ mediante ``_apply_preprocessing_inplace()``, que
    es invocado por ``make_pf_splits()`` con stats calculadas
    EXCLUSIVAMENTE sobre el split de entrenamiento.

    Normalización soportada (clave 'norm_type' en preprocessing_cfg):
      - 'z_norm'  : Escalado estándar  (x - mean) / std
      - 'minmax'  : Escalado Min-Max   (x - min)  / (max - min)
    """

    # Claves que corresponden a arrays hit-level (no event-level)
    _HIT_KEYS = frozenset({"x", "y", "z", "i", "j", "k", "thr", "time", "offsets"})

    def __init__(
        self,
        path: str,
        preprocessing_cfg: Optional[dict] = None,
        filters: Optional[dict] = None,
        use_weighted_loss: bool = False,
    ):
        super().__init__()

        self._path = path
        self._preprocessing_cfg = preprocessing_cfg  # stored for reference only
        self._stats = None

        ext = os.path.splitext(path)[1].lower()
        is_hdf5 = ext in [".h5", ".hdf5"]

        # ----------------------------------------------------------
        # Abrir archivo y cargar TODOS los arrays en RAM
        # ----------------------------------------------------------
        if is_hdf5:
            f = h5py.File(path, "r")
            available = list(f.keys())
        else:
            f = np.load(path, allow_pickle=bool(filters))
            available = list(f.files) if hasattr(f, "files") else list(f.keys())

        self._offsets = np.asarray(f["offsets"]).astype(np.int64)
        n_events_total = len(self._offsets) - 1

        def _to_tensor(key: str, required: bool = True) -> Optional[torch.Tensor]:
            if key in available:
                return torch.from_numpy(np.asarray(f[key]).astype(np.float32))
            if required:
                raise KeyError(f"Falta la clave '{key}' en {path}")
            return None

        # Hit-level arrays
        self._x   = _to_tensor("x")
        self._y   = _to_tensor("y")
        self._z   = _to_tensor("z")
        self._i   = _to_tensor("i")
        self._j   = _to_tensor("j")
        self._k   = _to_tensor("k")
        self._thr = _to_tensor("thr")
        self._time = _to_tensor("time", required=False)

        # Pre-compute one-hot threshold masks BEFORE any normalization of thr
        self._thr1 = (self._thr == 1).float()
        self._thr2 = (self._thr == 2).float()
        self._thr3 = (self._thr == 3).float()

        # Event-level arrays
        self._energy = _to_tensor("energy", required=False)

        # Cargar arrays event-level adicionales (necesarios para filtros)
        event_data: dict = {}
        for key in available:
            if key not in self._HIT_KEYS:
                try:
                    arr = np.asarray(f[key])
                    event_data[key] = arr
                except Exception:
                    pass

        # Cerrar archivo (ya no se necesita)
        if is_hdf5:
            f.close()

        # ----------------------------------------------------------
        # Aplicar filtros
        # ----------------------------------------------------------
        self._event_indices = np.arange(n_events_total, dtype=np.int64)
        self._n_events = n_events_total

        if filters:
            mask = _apply_filters_inmem(event_data, n_events_total, filters)
            n_passed = int(mask.sum())
            print(
                f"[FlatSDHCALDataset] Filtros aplicados: "
                f"{n_passed}/{n_events_total} eventos pasan "
                f"({100 * n_passed / max(1, n_events_total):.1f}%)"
            )
            self._event_indices = np.flatnonzero(mask).astype(np.int64)
            self._n_events = int(self._event_indices.size)

        # ----------------------------------------------------------
        # Class weights desde la distribución de energía
        # (computed from all filtered events, before any log transform)
        # ----------------------------------------------------------
        if self._energy is not None and use_weighted_loss:
            en_np = self._energy[self._event_indices].numpy()
            unique_vals, counts = np.unique(en_np, return_counts=True)
            n_unique = len(unique_vals)
            n_total = len(en_np)

            # Auto-detect: discrete (test beam) vs continuous (simulation)
            is_discrete = n_unique <= max(30, int(0.001 * n_total))

            if is_discrete:
                # Discrete: one weight per unique energy value
                self.n_events_per_energy = {
                    f"{float(v):g}": int(c) for v, c in zip(unique_vals, counts)
                }
                total = int(counts.sum())
                self.weights = {k: total / v for k, v in self.n_events_per_energy.items()}
                self.weights["__meta__"] = {"bin_half_width": 0.5, "type": "discrete"}
                print(
                    f"[Weights] Energía discreta detectada: "
                    f"{n_unique} valores únicos, pesos por valor."
                )
            else:
                # Continuous: histogram with fixed-width bins
                bin_width = 5.0  # GeV
                e_min, e_max = float(en_np.min()), float(en_np.max())
                bin_edges = np.arange(e_min, e_max + bin_width, bin_width)
                hist_counts, _ = np.histogram(en_np, bins=bin_edges)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                total = int(hist_counts.sum())

                self.n_events_per_energy = {}
                self.weights = {}
                for center, count in zip(bin_centers, hist_counts):
                    if count > 0:
                        key = f"{float(center):g}"
                        self.n_events_per_energy[key] = int(count)
                        self.weights[key] = total / int(count)
                self.weights["__meta__"] = {
                    "bin_half_width": bin_width / 2.0,
                    "type": "continuous",
                    "bin_width": bin_width,
                }
                n_bins = sum(1 for c in hist_counts if c > 0)
                print(
                    f"[Weights] Energía continua detectada: "
                    f"{n_unique} valores únicos → {n_bins} bines de {bin_width} GeV."
                )
        else:
            self.n_events_per_energy = None
            self.weights = None

    # ----------------------------------------------------------
    # Preprocessing (called ONCE by make_pf_splits after split)
    # ----------------------------------------------------------

    def _apply_preprocessing_inplace(
        self,
        stats: dict,
        norm_type: Optional[str],
        preprocessing_cfg: dict,
    ) -> None:
        """
        Aplica normalización y/o transformación logarítmica UNA SOLA VEZ
        sobre todos los arrays hit-level y event-level en RAM.

        Debe llamarse DESPUÉS de calcular las stats SOLO desde el split
        de entrenamiento (ver ``make_pf_splits``).
        """

        if norm_type in ("z_norm", "minmax") and stats:
            def _norm(tensor: torch.Tensor, key: str) -> torch.Tensor:
                s = stats.get(key)
                if s is None:
                    return tensor
                if norm_type == "z_norm":
                    return (tensor - s["mean"]) / s["std"]
                else:  # minmax
                    rng = s["max"] - s["min"]
                    return (tensor - s["min"]) / (rng if rng > 1e-8 else 1.0)

            self._x = _norm(self._x, "x")
            self._y = _norm(self._y, "y")
            self._z = _norm(self._z, "z")

            if preprocessing_cfg.get("use_scalar", False):
                self._k = _norm(self._k, "k")

            if not preprocessing_cfg.get("use_one_hot", False):
                self._thr = _norm(self._thr, "thr")

            if (
                preprocessing_cfg.get("use_time", False)
                and self._time is not None
                and "time" in stats
            ):
                self._time = _norm(self._time, "time")

        # Transformación logarítmica de energía (independiente de norm_type)
        if (
            preprocessing_cfg.get("use_energy", False)
            and preprocessing_cfg.get("use_log", False)
            and self._energy is not None
        ):
            self._energy = torch.log(self._energy + 1e-6)

        self._stats = stats
        self._preprocessing_cfg = preprocessing_cfg
        print(
            f"[FlatSDHCALDataset] Preprocesamiento aplicado"
            + (f" ({norm_type})" if norm_type else "")
            + " a todos los arrays en RAM."
        )

    # ----------------------------------------------------------
    # Dataset interface
    # ----------------------------------------------------------

    def len(self) -> int:
        return self._n_events

    def get(self, idx: int) -> Data:
        real_idx = int(self._event_indices[idx])
        start = int(self._offsets[real_idx])
        end = int(self._offsets[real_idx + 1])

        x  = self._x[start:end].unsqueeze(1)
        y  = self._y[start:end].unsqueeze(1)
        z  = self._z[start:end].unsqueeze(1)
        i  = self._i[start:end].unsqueeze(1)
        j  = self._j[start:end].unsqueeze(1)
        k  = self._k[start:end].unsqueeze(1)
        thr  = self._thr[start:end].unsqueeze(1)
        thr1 = self._thr1[start:end].unsqueeze(1)
        thr2 = self._thr2[start:end].unsqueeze(1)
        thr3 = self._thr3[start:end].unsqueeze(1)

        time = (
            self._time[start:end].unsqueeze(1)
            if self._time is not None
            else torch.zeros((end - start, 1), dtype=torch.float32)
        )

        energy = (
            self._energy[real_idx]
            if self._energy is not None
            else torch.tensor(0.0)
        )

        pos = torch.cat([x, y, z], dim=1)

        return Data(
            pos=pos,
            x=x, y=y, z=z,
            thr=thr, thr1=thr1, thr2=thr2, thr3=thr3,
            i=i, j=j, k=k,
            time=time,
            energy=energy,
        )


# ============================================================
# Dataset multi-fichero plano (HDF5 / NPZ con offsets)
# ============================================================

class MultiFileFlatSDHCALDataset(FlatSDHCALDataset):
    """
    Fusiona varios ficheros planos (HDF5 o NPZ con offsets) en un único dataset.

    Salta FlatSDHCALDataset.__init__ y construye los mismos atributos
    concatenando los arrays de todos los ficheros. Al heredar de
    FlatSDHCALDataset, isinstance() devuelve True y _apply_preprocessing_inplace,
    get() y len() se heredan sin cambios.
    """

    def __init__(
        self,
        paths: List[str],
        preprocessing_cfg: Optional[dict] = None,
        filters: Optional[dict] = None,
        use_weighted_loss: bool = False,
    ):
        # Inicializar solo el Dataset base (saltar FlatSDHCALDataset.__init__)
        Dataset.__init__(self)
        self._preprocessing_cfg = preprocessing_cfg
        self._stats = None
        self._path = paths[0]  # usado para derivar el YAML de stats

        xs, ys, zs, i_s, js, ks, thrs, times, energies = [], [], [], [], [], [], [], [], []
        all_offsets: List[np.ndarray] = []
        hit_offset = 0
        all_event_data: dict = {}

        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            is_hdf5 = ext in [".h5", ".hdf5"]
            print(f"[MultiFileFlatSDHCALDataset] Cargando: {path}")

            if is_hdf5:
                f = h5py.File(path, "r")
                available = list(f.keys())
            else:
                f = np.load(path, allow_pickle=False)
                available = list(f.files)

            offsets_file = np.asarray(f["offsets"]).astype(np.int64)
            if not all_offsets:
                all_offsets.append(offsets_file)
            else:
                # Saltar el primer elemento (ya está como último del bloque anterior)
                all_offsets.append(offsets_file[1:] + hit_offset)
            hit_offset += int(offsets_file[-1])

            def _load(key, req=True, _f=f, _available=available, _path=path):
                if key in _available:
                    return torch.from_numpy(np.asarray(_f[key]).astype(np.float32))
                if req:
                    raise KeyError(f"Falta '{key}' en {_path}")
                return None

            xs.append(_load("x"))
            ys.append(_load("y"))
            zs.append(_load("z"))
            i_s.append(_load("i"))
            js.append(_load("j"))
            ks.append(_load("k"))
            thrs.append(_load("thr"))
            times.append(_load("time", req=False))
            energies.append(_load("energy", req=False))

            # Arrays event-level (para filtros)
            for key in available:
                if key not in FlatSDHCALDataset._HIT_KEYS:
                    try:
                        all_event_data.setdefault(key, []).append(np.asarray(f[key]))
                    except Exception:
                        pass

            if is_hdf5:
                f.close()

        # Concatenar arrays hit-level
        self._offsets = np.concatenate(all_offsets)
        self._x   = torch.cat(xs)
        self._y   = torch.cat(ys)
        self._z   = torch.cat(zs)
        self._i   = torch.cat(i_s)
        self._j   = torch.cat(js)
        self._k   = torch.cat(ks)
        self._thr = torch.cat(thrs)

        # time: concatenar solo si TODOS los ficheros lo tienen
        if all(t is not None for t in times):
            self._time = torch.cat(times)
        else:
            self._time = None

        # energy: concatenar solo si TODOS los ficheros lo tienen
        if all(e is not None for e in energies):
            self._energy = torch.cat(energies)
        else:
            self._energy = None

        # One-hot thresholds (antes de cualquier normalización)
        self._thr1 = (self._thr == 1).float()
        self._thr2 = (self._thr == 2).float()
        self._thr3 = (self._thr == 3).float()

        # Aplicar filtros sobre el dataset combinado
        event_data = {k: np.concatenate(v) for k, v in all_event_data.items()}
        n_events_total = len(self._offsets) - 1
        self._event_indices = np.arange(n_events_total, dtype=np.int64)
        self._n_events = n_events_total

        if filters:
            mask = _apply_filters_inmem(event_data, n_events_total, filters)
            n_passed = int(mask.sum())
            print(
                f"[MultiFileFlatSDHCALDataset] Filtros aplicados: "
                f"{n_passed}/{n_events_total} eventos pasan "
                f"({100 * n_passed / max(1, n_events_total):.1f}%)"
            )
            self._event_indices = np.flatnonzero(mask).astype(np.int64)
            self._n_events = int(self._event_indices.size)

        # Class weights (igual que FlatSDHCALDataset)
        if self._energy is not None and use_weighted_loss:
            en_np = self._energy[self._event_indices].numpy()
            unique_vals, counts = np.unique(en_np, return_counts=True)
            n_unique = len(unique_vals)
            n_total = len(en_np)
            is_discrete = n_unique <= max(30, int(0.001 * n_total))
            if is_discrete:
                self.n_events_per_energy = {
                    f"{float(v):g}": int(c) for v, c in zip(unique_vals, counts)
                }
                total = int(counts.sum())
                self.weights = {k: total / v for k, v in self.n_events_per_energy.items()}
                self.weights["__meta__"] = {"bin_half_width": 0.5, "type": "discrete"}
            else:
                bin_width = 5.0
                e_min, e_max = float(en_np.min()), float(en_np.max())
                bin_edges = np.arange(e_min, e_max + bin_width, bin_width)
                hist_counts, _ = np.histogram(en_np, bins=bin_edges)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                total = int(hist_counts.sum())
                self.n_events_per_energy = {}
                self.weights = {}
                for center, count in zip(bin_centers, hist_counts):
                    if count > 0:
                        key = f"{float(center):g}"
                        self.n_events_per_energy[key] = int(count)
                        self.weights[key] = total / int(count)
                self.weights["__meta__"] = {
                    "bin_half_width": bin_width / 2.0,
                    "type": "continuous",
                    "bin_width": bin_width,
                }
        else:
            self.weights = None
            self.n_events_per_energy = None

        print(
            f"[MultiFileFlatSDHCALDataset] {len(paths)} ficheros fusionados. "
            f"Eventos: {self._n_events}"
        )


# ============================================================
# Dataset original (legacy, para npz jagged con pickle)
# ============================================================

class SDHCALDataset(Dataset):
    """
    Dataset SDHCAL (NPZ jagged con pickle).
    Mantenido por compatibilidad; para rendimiento usar FlatSDHCALDataset.
    """

    def __init__(self, paths: Sequence[str], mode: str = "lazy"):
        super().__init__()
        if isinstance(paths, str):
            paths = [paths]
        self.paths = list(paths)
        self.mode = mode

        self._files = []
        self._file_event_counts = []
        self._has_energy = []

        required_keys = {"x", "y", "z", "i", "j", "k", "thr"}
        for p in self.paths:
            npz = np.load(p, allow_pickle=True)
            available = set(npz.files)
            missing = required_keys - available
            if missing:
                raise KeyError(f"Faltan claves {missing} en {p}")
            self._has_energy.append("energy" in available)
            n_events = self._infer_num_events(npz)
            self._file_event_counts.append(n_events)
            if self.mode == "memory":
                self._files.append({k: npz[k] for k in npz.files})
            else:
                self._files.append(None)

        self._cum_counts = np.cumsum(self._file_event_counts)

    def len(self):
        return int(self._cum_counts[-1]) if len(self._cum_counts) > 0 else 0

    def _infer_num_events(self, npz):
        if "n_events" in npz:
            return int(npz["n_events"])
        if "x" in npz:
            return int(len(npz["x"]))
        raise KeyError("No se puede inferir el número de eventos en el npz")

    def _locate(self, idx: int):
        file_idx = int(np.searchsorted(self._cum_counts, idx, side="right"))
        prev = int(self._cum_counts[file_idx - 1]) if file_idx > 0 else 0
        local_idx = int(idx - prev)
        return file_idx, local_idx

    def _open_file(self, file_idx: int):
        cached = self._files[file_idx]
        if cached is not None:
            return cached
        return np.load(self.paths[file_idx], allow_pickle=True)

    def get(self, idx):
        file_idx, local_idx = self._locate(idx)
        npz = self._open_file(file_idx)

        x = torch.as_tensor(npz["x"][local_idx]).float()
        y = torch.as_tensor(npz["y"][local_idx]).float()
        z = torch.as_tensor(npz["z"][local_idx]).float()
        pos = torch.stack([x, y, z], dim=1).float()

        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        z = z.unsqueeze(1)

        i = torch.as_tensor(npz["i"][local_idx], dtype=torch.float32).unsqueeze(1)
        j = torch.as_tensor(npz["j"][local_idx], dtype=torch.float32).unsqueeze(1)
        k = torch.as_tensor(npz["k"][local_idx], dtype=torch.float32).unsqueeze(1)

        thr = torch.as_tensor(npz["thr"][local_idx])
        thr1 = (thr == 1).float().unsqueeze(1)
        thr2 = (thr == 2).float().unsqueeze(1)
        thr3 = (thr == 3).float().unsqueeze(1)
        thr = thr.float().unsqueeze(1)

        if self._has_energy[file_idx]:
            energy_val = float(npz["energy"][local_idx])
        else:
            energy_val = 0.0

        energy = torch.tensor(energy_val, dtype=torch.float32)

        return Data(
            pos=pos, x=x, y=y, z=z,
            thr=thr, thr1=thr1, thr2=thr2, thr3=thr3,
            i=i, j=j, k=k,
            energy=energy,
        )


class HitsDataset(SDHCALDataset):
    def __init__(self, paths: Sequence[str], mode: str = "lazy"):
        super().__init__(paths, mode=mode)

    def get(self, idx):
        return super().get(idx)


# ============================================================
# Splits train/val  (preprocessing aplicado aquí, no en get())
# ============================================================

def make_pf_splits(
    paths: List[str],
    val_ratio: float = 0.2,
    mode: str = "memory",
    seed: int = 42,
    preprocessing_cfg: Optional[dict] = None,
    filters: Optional[dict] = None,
    use_weighted_loss: bool = False,
) -> Tuple[Dataset, Dataset]:
    """
    Crea splits train/val.  Para datasets planos (FlatSDHCALDataset):
      1. Carga todos los datos en RAM (sin preprocesar).
      2. Divide en train/val.
      3. Calcula las stats de normalización SOLO desde el split de train.
      4. Aplica la normalización UNA VEZ sobre todos los arrays en RAM.
      5. Devuelve Subsets que apuntan al mismo dataset ya normalizado.
    """
    # Detección automática de formato
    if len(paths) == 1:
        path = paths[0]
        ext = os.path.splitext(path)[1].lower()
        print("Leyendo dataset desde:", path)

        if ext in [".h5", ".hdf5"]:
            with h5py.File(path, "r") as probe:
                is_flat = "offsets" in probe
            if is_flat:
                # Crear dataset SIN aplicar preprocessing (preprocessing_cfg=None aquí;
                # se aplica después del split)
                dataset = FlatSDHCALDataset(path, preprocessing_cfg=None, filters=filters, use_weighted_loss=use_weighted_loss)
                print(f"Detectado formato HDF5 plano. Eventos: {dataset.len()}")
            else:
                dataset = HitsDataset(paths, mode=mode)

        elif ext == ".npz":
            probe = np.load(path, allow_pickle=False)
            if "offsets" in probe:
                dataset = FlatSDHCALDataset(path, preprocessing_cfg=None, filters=filters, use_weighted_loss=use_weighted_loss)
            else:
                dataset = HitsDataset(paths, mode=mode)
        else:
            dataset = HitsDataset(paths, mode=mode)
    else:
        # Comprobar si todos los ficheros son planos (tienen offsets)
        all_flat = True
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            try:
                if ext in [".h5", ".hdf5"]:
                    with h5py.File(p, "r") as probe:
                        if "offsets" not in probe:
                            all_flat = False
                            break
                elif ext == ".npz":
                    probe = np.load(p, allow_pickle=False)
                    if "offsets" not in probe:
                        all_flat = False
                        break
                else:
                    all_flat = False
                    break
            except Exception:
                all_flat = False
                break

        if all_flat:
            dataset = MultiFileFlatSDHCALDataset(
                paths,
                preprocessing_cfg=None,
                filters=filters,
                use_weighted_loss=use_weighted_loss,
            )
        else:
            dataset = HitsDataset(paths, mode=mode)

    # ----------------------------------------------------------
    # Split
    # ----------------------------------------------------------
    N = dataset.len()
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)
    val_size = int(N * val_ratio)
    val_idx   = indices[:val_size]
    train_idx = indices[val_size:]

    # ----------------------------------------------------------
    # Preprocessing (solo para FlatSDHCALDataset)
    # ----------------------------------------------------------
    if isinstance(dataset, FlatSDHCALDataset) and preprocessing_cfg:
        norm_type: Optional[str] = preprocessing_cfg.get("norm_type")
        if norm_type is None and preprocessing_cfg.get("z_norm", False):
            norm_type = "z_norm"

        if norm_type in ("z_norm", "minmax"):
            stats = _load_or_compute_stats(dataset, train_idx, preprocessing_cfg, norm_type)
        else:
            stats = {}

        # Applies normalization AND/OR log-energy in one shot
        dataset._apply_preprocessing_inplace(stats, norm_type, preprocessing_cfg)

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)

    return train_ds, val_ds


# ============================================================
# Test manual desde CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test de make_pf_splits con dataset plano o jagged"
    )
    parser.add_argument("--path", type=str, required=True, help="Ruta al dataset (.npz o .h5)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fracción de validación")
    parser.add_argument("--seed", type=int, default=42, help="Seed para el split")

    args = parser.parse_args()

    print(f"\n[TEST] Cargando dataset: {args.path}")

    train_ds, val_ds = make_pf_splits(
        paths=[args.path],
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("\n[TEST] Split completado correctamente")
    print(f"Total train samples: {len(train_ds)}")
    print(f"Total val samples:   {len(val_ds)}")

    if len(train_ds) > 0:
        sample = train_ds[0]
        print("\n[TEST] Primera muestra (train):")
        print(sample)
        print("pos shape:", sample.pos.shape)
        print("energy:", sample.energy)

    print("\n[TEST] OK ✔")
