import argparse
import json
from typing import Dict, Iterable

import numpy as np
import tqdm as tqdm

def _update_stats(stats: Dict[str, Dict[str, float]], key: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if values.size == 0:
        return
    if key not in stats:
        stats[key] = {"count": 0.0, "sum": 0.0, "sumsq": 0.0}
    stats[key]["count"] += float(values.size)
    stats[key]["sum"] += float(values.sum())
    stats[key]["sumsq"] += float(np.square(values).sum())


def compute_stats(paths: Iterable[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for path in tqdm.tqdm(paths):
        npz = np.load(path, allow_pickle=True)

        for key in ("x", "y", "z", "i", "j", "k", "thr"):
            if key not in npz:
                raise KeyError(f"Falta la clave '{key}' en {path}")

        # Concatenar todos los eventos de golpe en un solo array 1-D
        for key in ("x", "y", "z", "i", "j", "k", "thr"):
            all_values = np.concatenate(npz[key])
            _update_stats(stats, key, all_values)

        if "energy" in npz:
            _update_stats(stats, "energy", npz["energy"])

    final: Dict[str, Dict[str, float]] = {}
    for key, s in stats.items():
        count = s["count"]
        mean = s["sum"] / max(count, 1.0)
        var = (s["sumsq"] / max(count, 1.0)) - mean * mean
        std = float(np.sqrt(max(var, 0.0)))
        final[key] = {"mean": float(mean), "std": std, "count": int(count)}
    return final


def main():
    parser = argparse.ArgumentParser(
        description="Calcula media y desviación estándar (z-score) para npz jagged."
    )
    parser.add_argument("paths", nargs="+", help="Lista de archivos .npz")
    parser.add_argument("--out", default="stats.json", help="Archivo JSON de salida")
    args = parser.parse_args()

    stats = compute_stats(args.paths)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print(f"Stats guardadas en {args.out}")


if __name__ == "__main__":
    main()
