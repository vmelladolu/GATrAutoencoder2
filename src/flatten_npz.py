"""
Convierte los .npz jagged (object arrays) en un formato plano y contiguo
que se puede cargar instantáneamente con np.load (sin pickle).

Uso:
    python flatten_npz.py input1.npz input2.npz --out flat_dataset.npz
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

KEYS = ("x", "y", "z", "i", "j", "k", "thr")


def _process_file(path: str):
    npz = np.load(path, allow_pickle=True)

    # npz["x"] es un array de objetos; cada objeto es un array 1-D
    obj_x = npz["x"]  # shape (n_events,), dtype=object
    n_events = len(obj_x)

    # 1) Longitudes de cada evento
    event_lengths = np.array([len(obj_x[i]) for i in range(n_events)], dtype=np.int64)

    # 2) Concatenar todos los hits de este archivo de golpe
    flat_data = {}
    for k in KEYS:
        arr = npz[k]  # array de objetos, shape (n_events,)
        flat_data[k] = np.concatenate(arr).astype(np.float32)

    # 3) Energía (un escalar por evento)
    energy = None
    if "energy" in npz:
        energy = np.asarray(npz["energy"], dtype=np.float32)

    return flat_data, event_lengths, energy, n_events, int(event_lengths.sum())


def flatten_npz_files(paths: list[str], out_path: str, workers: int) -> None:
    # Acumuladores globales
    chunks = {k: [] for k in KEYS}   # listas de arrays 1-D contiguos
    lengths = []                    # n_hits por evento (todos los archivos)
    all_energy = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(_process_file, paths),
            total=len(paths),
            desc="Leyendo npz",
        ))

    for path, (flat_data, event_lengths, energy, n_events, n_hits) in zip(paths, results):
        lengths.append(event_lengths)
        for k in KEYS:
            chunks[k].append(flat_data[k])
        if energy is not None:
            all_energy.append(energy)
        print(f"  {path}: {n_events} eventos, {n_hits} hits")

    # Unir todo
    all_lengths = np.concatenate(lengths)
    offsets = np.zeros(len(all_lengths) + 1, dtype=np.int64)
    np.cumsum(all_lengths, out=offsets[1:])

    data = {k: np.concatenate(chunks[k]) for k in KEYS}
    data["offsets"] = offsets

    if all_energy:
        data["energy"] = np.concatenate(all_energy)

    np.savez(out_path, **data)
    n_ev = len(all_lengths)
    n_hits = int(offsets[-1])
    print(f"Guardado {out_path}: {n_ev} eventos, {n_hits} hits totales")


def main():
    parser = argparse.ArgumentParser(description="Aplana npz jagged a formato contiguo")
    parser.add_argument("paths", nargs="+", help="Archivos .npz de entrada")
    parser.add_argument("--out", default="flat_dataset.npz", help="Archivo de salida")
    parser.add_argument("-j", "--workers", type=int, default=1, help="Procesos en paralelo")
    args = parser.parse_args()
    flatten_npz_files(args.paths, args.out, args.workers)


if __name__ == "__main__":
    main()