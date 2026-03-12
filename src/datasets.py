from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


# ============================================================
# Dataset rápido para formato plano (flatten_npz.py)
# ============================================================

class FlatSDHCALDataset(Dataset):
    """
    Dataset que lee un .npz plano (sin pickle) con arrays contiguos + offsets.
    La carga en memoria es instantánea y el acceso por evento es O(1).
    """

    def __init__(self, path: str):
        super().__init__()
        # Cargar TODO en RAM de golpe — arrays contiguos, sin pickle
        raw = np.load(path, allow_pickle=False)

        self._x = torch.from_numpy(raw["x"].astype(np.float32))
        self._y = torch.from_numpy(raw["y"].astype(np.float32))
        self._z = torch.from_numpy(raw["z"].astype(np.float32))
        self._i = torch.from_numpy(raw["i"].astype(np.float32))
        self._j = torch.from_numpy(raw["j"].astype(np.float32))
        self._k = torch.from_numpy(raw["k"].astype(np.float32))
        self._thr = torch.from_numpy(raw["thr"].astype(np.float32))
        self._offsets = raw["offsets"].astype(np.int64)

        if "energy" in raw:
            self._energy = torch.from_numpy(raw["energy"].astype(np.float32))
        else:
            self._energy = None

        self._n_events = len(self._offsets) - 1

    def len(self) -> int:
        return self._n_events

    def get(self, idx: int) -> Data:
        start = int(self._offsets[idx])
        end = int(self._offsets[idx + 1])

        x = self._x[start:end].unsqueeze(1)
        y = self._y[start:end].unsqueeze(1)
        z = self._z[start:end].unsqueeze(1)
        pos = torch.cat([x, y, z], dim=1)

        i = self._i[start:end].unsqueeze(1)
        j = self._j[start:end].unsqueeze(1)
        k = self._k[start:end].unsqueeze(1)

        thr_raw = self._thr[start:end]
        thr = thr_raw.unsqueeze(1)
        thr1 = (thr_raw == 1).float().unsqueeze(1)
        thr2 = (thr_raw == 2).float().unsqueeze(1)
        thr3 = (thr_raw == 3).float().unsqueeze(1)

        energy = self._energy[idx] if self._energy is not None else torch.tensor(0.0)

        return Data(
            pos=pos, x=x, y=y, z=z,
            thr=thr, thr1=thr1, thr2=thr2, thr3=thr3,
            i=i, j=j, k=k,
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
) -> Tuple[Dataset, Dataset]:
    """
    Crea splits train/val. Acepta:
      - Un solo .npz plano (FlatSDHCALDataset, rápido)
      - Varios .npz jagged (HitsDataset, legacy)
    """
    # Detección automática: si es un solo archivo y tiene "offsets", es plano
    if len(paths) == 1:
        try:
            probe = np.load(paths[0], allow_pickle=False)
            if "offsets" in probe:
                dataset = FlatSDHCALDataset(paths[0])
            else:
                dataset = HitsDataset(paths, mode=mode)
        except Exception:
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
