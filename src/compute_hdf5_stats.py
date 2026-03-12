import argparse
import json
from collections import defaultdict
from typing import Dict, Iterable, DefaultDict
import yaml
import h5py
import numpy as np
import tqdm


FEATURE_KEYS = ("x", "y", "z", "i", "j", "k", "thr")
OPTIONAL_FEATURE_KEYS = ("time",)  # calculadas solo si existen en el archivo


def _update_stats(stats: Dict[str, Dict[str, float]], key: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if values.size == 0:
        return
    if key not in stats:
        stats[key] = {"count": 0.0, "sum": 0.0, "sumsq": 0.0}
    stats[key]["count"] += float(values.size)
    stats[key]["sum"] += float(values.sum())
    stats[key]["sumsq"] += float(np.square(values).sum())


def _finalize_stats(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    final: Dict[str, Dict[str, float]] = {}
    for key, s in stats.items():
        count = s["count"]
        mean = s["sum"] / max(count, 1.0)
        var = (s["sumsq"] / max(count, 1.0)) - mean * mean
        std = float(np.sqrt(max(var, 0.0)))
        final[key] = {"mean": float(mean), "std": std, "count": int(count)}
    return final


def _format_energy_value(v: float) -> str:
    # Claves de JSON estables para energías discretas (ej: "7.5", "70", "120")
    return f"{float(v):g}"


def _accumulate_energy_counts(
    energy_counts: DefaultDict[str, int], energy_values: np.ndarray
) -> None:
    unique_vals, counts = np.unique(energy_values, return_counts=True)
    for val, cnt in zip(unique_vals, counts):
        energy_counts[_format_energy_value(float(val))] += int(cnt)


def compute_stats_hdf5(paths: Iterable[str], chunk_size: int = 5_000_000) -> Dict[str, object]:
    stats: Dict[str, Dict[str, float]] = {}
    energy_counts: DefaultDict[str, int] = defaultdict(int)
    total_events = 0

    for path in tqdm.tqdm(paths, desc="HDF5 files"):
        with h5py.File(path, "r") as h5:
            for key in FEATURE_KEYS:
                if key not in h5:
                    raise KeyError(f"Falta la clave '{key}' en {path}")
            if "energy" not in h5:
                raise KeyError(f"Falta la clave 'energy' en {path}")

            # Número de eventos: preferir offsets (robusto), fallback a len(energy)
            if "offsets" in h5:
                total_events += int(len(h5["offsets"]) - 1)
            else:
                total_events += int(len(h5["energy"]))

            # Estadísticas hit-level (obligatorias)
            for key in FEATURE_KEYS:
                ds = h5[key]
                n = len(ds)
                for start in range(0, n, chunk_size):
                    stop = min(start + chunk_size, n)
                    values = np.asarray(ds[start:stop], dtype=np.float64)
                    _update_stats(stats, key, values)

            # Estadísticas hit-level (opcionales, solo si existen en el archivo)
            for key in OPTIONAL_FEATURE_KEYS:
                if key not in h5:
                    continue
                ds = h5[key]
                n = len(ds)
                for start in range(0, n, chunk_size):
                    stop = min(start + chunk_size, n)
                    values = np.asarray(ds[start:stop], dtype=np.float64)
                    _update_stats(stats, key, values)

            # Estadísticas y conteo por energía (event-level)
            energy_ds = h5["energy"]
            n_e = len(energy_ds)
            for start in range(0, n_e, chunk_size):
                stop = min(start + chunk_size, n_e)
                energy_values = np.asarray(energy_ds[start:stop], dtype=np.float64)
                _update_stats(stats, "energy", energy_values)
                _accumulate_energy_counts(energy_counts, energy_values)

    output: Dict[str, object] = {
        "stats": _finalize_stats(stats),
        "events": {
            "total_events": int(total_events),
            "counts_by_energy": dict(sorted(energy_counts.items(), key=lambda x: float(x[0]))),
        },
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula media/std para HDF5 plano y devuelve total de eventos y conteo por energía."
    )
    parser.add_argument("paths", nargs="+", help="Lista de archivos .h5/.hdf5")
    parser.add_argument("--out", default="stats_hdf5.yml", help="Archivo YAML de salida")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000_000,
        help="Tamaño de bloque para leer arrays grandes sin desbordar memoria",
    )
    args = parser.parse_args()

    result = compute_stats_hdf5(args.paths, chunk_size=args.chunk_size)
    # save as yml
    
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.dump(result, f, default_flow_style=False)

    # with open(args.out, "w", encoding="utf-8") as f:
        # json.dump(result, f, indent=2, sort_keys=True)

    print(f"Stats guardadas en {args.out}")
    print(f"Total de eventos: {result['events']['total_events']}")
    print("Energías encontradas:", len(result["events"]["counts_by_energy"]))


if __name__ == "__main__":
    main()
