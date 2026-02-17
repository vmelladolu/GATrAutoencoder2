from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import os
import h5py

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


# ============================================================
# Dataset rápido para formato plano (flatten_npz.py)
# ============================================================

# Mapeo de strings a enteros (debe coincidir con STRING_TO_INT_MAPPINGS en process_flat_npz.py)
STRING_TO_INT_MAPPINGS = {
    "particle_type": {"electron": 0, "pion": 1, "muon": 2, "unknown": -1},
}


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


def _resolve_event_array(raw: np.lib.npyio.NpzFile, key: str) -> Optional[np.ndarray]:
    # Usar raw.files para verificar existencia sin cargar el array
    files = raw.files if hasattr(raw, "files") else list(raw.keys())

    if key in files:
        return raw[key]
    prefixed = f"filter_{key}"
    if prefixed in files:
        return raw[prefixed]
    if key == "status" and "filter_status" in files:
        return raw["filter_status"]
    return None


def _apply_filters_flat_npz(raw: np.lib.npyio.NpzFile, filters: dict) -> np.ndarray:
    if not filters:
        n_events = len(raw["offsets"]) - 1
        return np.ones(n_events, dtype=bool)

    n_events = len(raw["offsets"]) - 1
    mask = np.ones(n_events, dtype=bool)

    for filter_key, filter_value in filters.items():
        arr = _resolve_event_array(raw, filter_key)
        if arr is None:
            raise KeyError(f"Filter key '{filter_key}' not found in NPZ")
        if len(arr) != n_events:
            raise ValueError(
                f"Filter key '{filter_key}' has length {len(arr)} but expected {n_events} events"
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
            # Convertir valores string a int para columnas con mapeo conocido
            compare_value = filter_value
            if filter_key in STRING_TO_INT_MAPPINGS and isinstance(filter_value, str):
                mapping = STRING_TO_INT_MAPPINGS[filter_key]
                if filter_value in mapping:
                    compare_value = mapping[filter_value]
            mask &= arr == compare_value

    return mask


class FlatSDHCALDataset(Dataset):
    """
    Dataset que lee:
      - NPZ plano (carga completa en RAM)
      - HDF5 plano (lectura por evento, sin duplicar memoria)

    El preprocesamiento se aplica por evento.
    """

    def __init__(
        self,
        path: str,
        preprocessing_cfg: Optional[dict] = None,
        filters: Optional[dict] = None,
    ):
        super().__init__()

        self._path = path
        self._preprocessing_cfg = preprocessing_cfg
        self._stats = None

        ext = os.path.splitext(path)[1].lower()

        if ext in [".h5", ".hdf5"]:
            import h5py
            self._raw = h5py.File(path, "r")
            self._is_hdf5 = True
        else:
            self._raw = np.load(path, allow_pickle=bool(filters))
            self._is_hdf5 = False

        self._offsets = np.asarray(self._raw["offsets"]).astype(np.int64)
        self._n_events = len(self._offsets) - 1
        self._event_indices = np.arange(self._n_events, dtype=np.int64)

        # =========================
        # Cargar stats si hay norm
        # =========================
        if self._preprocessing_cfg and self._preprocessing_cfg.get("z_norm", False):
            import yaml
            with open(self._preprocessing_cfg["z_norm_yaml_path"], "r") as f:
                self._stats = yaml.safe_load(f)

        # =========================
        # Cargar en RAM SOLO si NPZ
        # =========================
        if not self._is_hdf5:
            self._x = torch.from_numpy(np.asarray(self._raw["x"]).astype(np.float32))
            self._y = torch.from_numpy(np.asarray(self._raw["y"]).astype(np.float32))
            self._z = torch.from_numpy(np.asarray(self._raw["z"]).astype(np.float32))
            self._i = torch.from_numpy(np.asarray(self._raw["i"]).astype(np.float32))
            self._j = torch.from_numpy(np.asarray(self._raw["j"]).astype(np.float32))
            self._k = torch.from_numpy(np.asarray(self._raw["k"]).astype(np.float32))
            self._thr = torch.from_numpy(np.asarray(self._raw["thr"]).astype(np.float32))

            if "energy" in self._raw:
                self._energy = torch.from_numpy(
                    np.asarray(self._raw["energy"]).astype(np.float32)
                )
            else:
                self._energy = None
        else:
            # HDF5 → no cargar arrays completos
            self._x = self._y = self._z = None
            self._i = self._j = self._k = None
            self._thr = None
            self._energy = None

        # =========================
        # Aplicar filtros
        # =========================
        if filters:
            mask = _apply_filters_flat_npz(self._raw, filters)
            n_passed = int(mask.sum())
            print(
                f"[FlatSDHCALDataset] Filtros aplicados: "
                f"{n_passed}/{self._n_events} eventos pasan "
                f"({100*n_passed/max(1,self._n_events):.1f}%)"
            )

            self._event_indices = np.flatnonzero(mask).astype(np.int64)
            self._n_events = int(self._event_indices.size)

    # ==========================================================
    # PREPROCESAMIENTO POR EVENTO
    # ==========================================================

    def _apply_preprocessing(self, x, y, z, k, thr, energy):
        if not self._preprocessing_cfg:
            return x, y, z, k, thr, energy

        # Normalización espacial
        if self._stats is not None:
            x = (x - self._stats["x"]["mean"]) / self._stats["x"]["std"]
            y = (y - self._stats["y"]["mean"]) / self._stats["y"]["std"]
            z = (z - self._stats["z"]["mean"]) / self._stats["z"]["std"]

            if self._preprocessing_cfg.get("use_scalar", False):
                k = (k - self._stats["k"]["mean"]) / self._stats["k"]["std"]

            if not self._preprocessing_cfg.get("use_one_hot", False):
                thr = (thr - self._stats["thr"]["mean"]) / self._stats["thr"]["std"]

        # Energía
        if (
            self._preprocessing_cfg.get("use_energy", False)
            and energy is not None
            and self._preprocessing_cfg.get("use_log", False)
        ):
            energy = torch.log(energy + 1e-6)

        return x, y, z, k, thr, energy

    # ==========================================================

    def len(self) -> int:
        return self._n_events

    def get(self, idx: int) -> Data:

        # Reabrir HDF5 si estamos en worker
        if self._is_hdf5 and not hasattr(self, "_worker_opened"):
            import h5py
            self._raw = h5py.File(self._path, "r")
            self._worker_opened = True

        real_idx = int(self._event_indices[idx])
        start = int(self._offsets[real_idx])
        end = int(self._offsets[real_idx + 1])

        if not self._is_hdf5:
            x = self._x[start:end].unsqueeze(1)
            y = self._y[start:end].unsqueeze(1)
            z = self._z[start:end].unsqueeze(1)
            i = self._i[start:end].unsqueeze(1)
            j = self._j[start:end].unsqueeze(1)
            k = self._k[start:end].unsqueeze(1)
            thr_raw = self._thr[start:end]
            energy = (
                self._energy[real_idx]
                if self._energy is not None
                else torch.tensor(0.0)
            )
        else:
            x = torch.from_numpy(
                np.asarray(self._raw["x"][start:end])
            ).float().unsqueeze(1)

            y = torch.from_numpy(
                np.asarray(self._raw["y"][start:end])
            ).float().unsqueeze(1)

            z = torch.from_numpy(
                np.asarray(self._raw["z"][start:end])
            ).float().unsqueeze(1)

            i = torch.from_numpy(
                np.asarray(self._raw["i"][start:end])
            ).float().unsqueeze(1)

            j = torch.from_numpy(
                np.asarray(self._raw["j"][start:end])
            ).float().unsqueeze(1)

            k = torch.from_numpy(
                np.asarray(self._raw["k"][start:end])
            ).float().unsqueeze(1)

            thr_raw = torch.from_numpy(
                np.asarray(self._raw["thr"][start:end])
            ).float()

            if "energy" in self._raw:
                energy = torch.tensor(
                    float(self._raw["energy"][real_idx]),
                    dtype=torch.float32,
                )
            else:
                energy = torch.tensor(0.0)

        thr = thr_raw.unsqueeze(1)
        thr1 = (thr_raw == 1).float().unsqueeze(1)
        thr2 = (thr_raw == 2).float().unsqueeze(1)
        thr3 = (thr_raw == 3).float().unsqueeze(1)

        # Aplicar preprocesamiento
        x, y, z, k, thr, energy = self._apply_preprocessing(
            x, y, z, k, thr, energy
        )

        pos = torch.cat([x, y, z], dim=1)

        return Data(
            pos=pos,
            x=x,
            y=y,
            z=z,
            thr=thr,
            thr1=thr1,
            thr2=thr2,
            thr3=thr3,
            i=i,
            j=j,
            k=k,
            energy=energy,
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
            # Validar claves una sola vez (npz.files es una lista, no lee datos)
            available = set(npz.files)
            missing = required_keys - available
            if missing:
                raise KeyError(f"Faltan claves {missing} en {p}")
            self._has_energy.append("energy" in available)
            n_events = self._infer_num_events(npz)
            self._file_event_counts.append(n_events)
            if self.mode == "memory":
                # Materializar los arrays en RAM para no releer el disco
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
# Splits train/val
# ============================================================

def make_pf_splits(
    paths: List[str],
    val_ratio: float = 0.2,
    mode: str = "memory",
    seed: int = 42,
    preprocessing_cfg: Optional[dict] = None,
    filters: Optional[dict] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Crea splits train/val. Acepta:
      - Un solo .npz plano (FlatSDHCALDataset, rápido)
      - Varios .npz jagged (HitsDataset, legacy)
    """
    # Detección automática: si es un solo archivo y tiene "offsets", es plano
    if len(paths) == 1:
        path = paths[0]
        ext = os.path.splitext(path)[1].lower()
        print("Leyendo dataset desde:", path)
        if ext in [".h5", ".hdf5"]:
            with h5py.File(path, "r") as probe:
                if "offsets" in probe:
                    dataset = FlatSDHCALDataset(
                        path,
                        preprocessing_cfg=preprocessing_cfg,
                        filters=filters,
                    )
                    print(f"Detectado formato HDF5 plano. Eventos: {dataset.len()}")
                else:
                    dataset = HitsDataset(paths, mode=mode)

        elif ext == ".npz":
            probe = np.load(path, allow_pickle=False)
            if "offsets" in probe:
                dataset = FlatSDHCALDataset(
                    path,
                    preprocessing_cfg=preprocessing_cfg,
                    filters=filters,
                )
            else:
                dataset = HitsDataset(paths, mode=mode)
        else:
            dataset = HitsDataset(paths, mode=mode)
    else:
        dataset = HitsDataset(paths, mode=mode)

    N = dataset.len()
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)
    val_size = int(N * val_ratio)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    return train_ds, val_ds



# ============================================================
# Test manual desde CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test de make_pf_splits con dataset plano o jagged"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Ruta al dataset (.npz o .h5)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fracción de validación",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para el split",
    )

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

    # Inspeccionar una muestra
    if len(train_ds) > 0:
        sample = train_ds[0]
        print("\n[TEST] Primera muestra (train):")
        print(sample)
        print("pos shape:", sample.pos.shape)
        print("energy:", sample.energy)

    print("\n[TEST] OK ✔")
