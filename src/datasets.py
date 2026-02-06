from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class SDHCALDataset(Dataset):
    """
        Dataset SDHCAL (NPZ):
        - nodos = hits
        - features = (i, j, k, thr1, thr2, thr3)
        - pos = (x,y,z)

        Cada archivo .npz contiene arrays por evento (jagged/object arrays):
            - "x", "y", "z": coordenadas por evento
            - "i", "j", "k": índices de celda/capa por evento
            - "thr": umbral por hit en {1,2,3} por evento
            - "energy": energía por evento (1D, un valor por evento)
    """

    def __init__(self, paths: Sequence[str], mode: str = "lazy"):
        super().__init__()
        if isinstance(paths, str):
            paths = [paths]
        self.paths = list(paths)
        self.mode = mode

        self._files = []
        self._file_event_counts = []

        for p in self.paths:
            npz = np.load(p, allow_pickle=True)
            n_events = self._infer_num_events(npz)
            self._file_event_counts.append(n_events)
            if self.mode == "memory":
                self._files.append(npz)
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

    def _get_event_array(self, arr, idx):
        return arr[idx]

    def get(self, idx):
        file_idx, local_idx = self._locate(idx)
        npz = self._open_file(file_idx)

        missing = [k for k in ("x", "y", "z", "i", "j", "k", "thr") if k not in npz]
        if missing:
            raise KeyError(f"Faltan claves en npz: {missing}")

        x = torch.as_tensor(self._get_event_array(npz["x"], local_idx))
        y = torch.as_tensor(self._get_event_array(npz["y"], local_idx))
        z = torch.as_tensor(self._get_event_array(npz["z"], local_idx))
        pos = torch.stack([x, y, z], dim=1).float()

        i = torch.as_tensor(self._get_event_array(npz["i"], local_idx), dtype=torch.float32)
        j = torch.as_tensor(self._get_event_array(npz["j"], local_idx), dtype=torch.float32)
        k = torch.as_tensor(self._get_event_array(npz["k"], local_idx), dtype=torch.float32)
        i = i.unsqueeze(1)
        j = j.unsqueeze(1)
        k = k.unsqueeze(1)
        
        thr = torch.as_tensor(self._get_event_array(npz["thr"], local_idx))
        
        thr1 = (thr == 1).float().unsqueeze(1)
        thr2 = (thr == 2).float().unsqueeze(1)
        thr3 = (thr == 3).float().unsqueeze(1)
        thr = thr.float().unsqueeze(1)

        if "energy" in npz:
            energy_val = float(self._get_event_array(npz["energy"], local_idx))
        else:
            energy_val = 0.0

        energy = torch.tensor(energy_val, dtype=torch.float32)

        return Data(pos=pos, thr=thr, thr1=thr1, thr2=thr2, thr3=thr3, i=i, j=j, k=k, energy=energy)


class HitsDataset(SDHCALDataset):
    def __init__(
        self,
        paths: Sequence[str],
        mode: str = "lazy",
    ):
        super().__init__(paths, mode=mode)

    def get(self, idx):
        data = super().get(idx)
        return data


def make_pf_splits(
    paths: List[str],
    val_ratio: float = 0.2,
    mode: str = "memory",
) -> Tuple[Dataset, Dataset]:
    """
    Crea splits de train/val para el dataset PFAutoRegressor.

    Args:
        paths: Lista de rutas a archivos .npz
        val_ratio: Fracción de datos para validación
        mode: 'memory' o 'lazy'
        preprocessor: Preprocesador opcional (si None, se crea uno por defecto)

    Returns:
        train_dataset, val_dataset
    """
    dataset = HitsDataset(paths, mode=mode)
    N = dataset.len()

    indices = np.random.permutation(N)
    val_size = int(N * val_ratio)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    return train_ds, val_ds
