import argparse
import json
import numpy as np
import os
from typing import Optional
import yaml
import time
from tqdm import tqdm
# ==========================================================
# CONFIGURACIÓN
# ==========================================================

HIT_KEYS = ("x", "y", "z", "i", "j", "k", "thr")

STRING_TO_INT_MAPPINGS = {
    "particle_type": {"electron": 0, "pion": 1, "muon": 2, "unknown": -1},
}


# ==========================================================
# UTILIDADES FILTRO
# ==========================================================

def _parse_filter_operator(value: str):
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

    if key in raw.files:
        return raw[key]

    prefixed = f"filter_{key}"
    if prefixed in raw.files:
        return raw[prefixed]

    if key == "status" and "filter_status" in raw.files:
        return raw["filter_status"]

    return None


def _apply_filters_flat_npz(raw: np.lib.npyio.NpzFile, filters: dict):

    n_events = len(raw["offsets"]) - 1

    if not filters:
        return np.ones(n_events, dtype=bool)

    mask = np.ones(n_events, dtype=bool)

    for key, value in filters.items():

        arr = _resolve_event_array(raw, key)

        if arr is None:
            raise KeyError(f"Filter key '{key}' not found")

        if len(arr) != n_events:
            raise ValueError(f"Array '{key}' is not per-event")

        if isinstance(value, str) and value.startswith((">=", "<=", ">", "<")):
            op, num = _parse_filter_operator(value)

            if op == ">=":
                mask &= arr >= num
            elif op == "<=":
                mask &= arr <= num
            elif op == ">":
                mask &= arr > num
            elif op == "<":
                mask &= arr < num

        else:
            compare_value = value
            if key in STRING_TO_INT_MAPPINGS and isinstance(value, str):
                compare_value = STRING_TO_INT_MAPPINGS[key].get(value, -1)

            mask &= arr == compare_value

    return mask


# ==========================================================
# ESCRITURA OPTIMIZADA
# ==========================================================

def save_npz(output_path, data_dict):
    np.savez_compressed(output_path, **data_dict)


def save_hdf5(output_path, data_dict, chunk_hits=1_000_000):

    import h5py

    with h5py.File(output_path, "w") as f:

        for key, arr in data_dict.items():

            if arr.ndim == 1 and len(arr) > chunk_hits:
                chunks = (min(chunk_hits, len(arr)),)
            else:
                chunks = True

            f.create_dataset(
                key,
                data=arr,
                compression="lzf",
                # compression_opts=4,
                chunks=chunks
            )


def save_zarr(output_path, data_dict, chunk_hits=1_000_000):

    import zarr
    from numcodecs import Blosc

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

    root = zarr.open(output_path, mode="w")

    for key, arr in data_dict.items():

        if arr.ndim == 1 and len(arr) > chunk_hits:
            chunks = (min(chunk_hits, len(arr)),)
        else:
            chunks = arr.shape

        root.create_dataset(
            key,
            data=arr,
            compressor=compressor,
            chunks=chunks
        )


# ==========================================================
# FILTRADO EFICIENTE (SIN LISTAS INTERMEDIAS)
# ==========================================================

def filter_large_flat_npz(input_path, output_path, filters, fmt):

    raw = np.load(input_path, allow_pickle=True)

    offsets = raw["offsets"]
    n_events = len(offsets) - 1

    print(f"Eventos totales: {n_events}")

    mask = _apply_filters_flat_npz(raw, filters)
    passed = np.flatnonzero(mask)

    n_passed = len(passed)

    print(f"Eventos que pasan: {n_passed} ({100*n_passed/max(1,n_events):.2f}%)")

    # ======================================================
    # PRE-CALCULAR TOTAL HITS
    # ======================================================

    hits_per_event = offsets[1:] - offsets[:-1]
    total_hits = hits_per_event[mask].sum()

    print(f"Total hits finales: {total_hits}")

    # ======================================================
    # PRE-ALLOCACIÓN
    # ======================================================
    
    # ======================================================
    # FILTRADO VECTORIAL ULTRA RÁPIDO (SIN BUCLE PYTHON)
    # ======================================================

    print("Construyendo máscara de hits...")

    # Cada evento tiene X hits
    hits_per_event = offsets[1:] - offsets[:-1]

    # Repetimos la máscara de eventos por número de hits
    mask_hits = np.repeat(mask, hits_per_event)

    print("Copiando hits seleccionados...")

    new_hits = {
        k: raw[k][mask_hits]
        for k in HIT_KEYS
    }

    print("Reconstruyendo offsets...")

    selected_lengths = hits_per_event[mask]

    new_offsets = np.zeros(len(selected_lengths) + 1, dtype=np.int64)
    new_offsets[1:] = np.cumsum(selected_lengths)

    # new_hits = {
    #     k: np.empty(total_hits, dtype=raw[k].dtype)
    #     for k in HIT_KEYS
    # }

    # new_offsets = np.zeros(n_passed + 1, dtype=np.int64)

    # # ======================================================
    # # COPIA STREAMING
    # # ======================================================

    # write_pos = 0

    # for idx, evt in enumerate(passed):

    #     start = offsets[evt]
    #     end = offsets[evt + 1]
    #     length = end - start

    #     for k in HIT_KEYS:
    #         new_hits[k][write_pos:write_pos+length] = raw[k][start:end]

    #     new_offsets[idx + 1] = new_offsets[idx] + length
    #     write_pos += length

    #     if idx % 10000 == 0 and idx > 0:
    #         print(f"Procesados {idx}/{n_passed} eventos...")

    # ======================================================
    # CONSTRUIR OUTPUT
    # ======================================================

    output_data = {}
    print("Construyendo diccionario de salida...")
    for k in tqdm(HIT_KEYS):
        output_data[k] = new_hits[k]

    output_data["offsets"] = new_offsets
    print("Copiando otros arrays...")
    # Copiar arrays por evento
    for key in tqdm(raw.files):

        if key in HIT_KEYS or key == "offsets":
            continue

        arr = raw[key]

        if len(arr) == n_events:
            output_data[key] = arr[mask]
        else:
            output_data[key] = arr

    # ======================================================
    # GUARDADO
    # ======================================================

    print(f"Guardando en formato {fmt.upper()}...")

    if fmt == "npz":
        save_npz(output_path, output_data)
    elif fmt == "hdf5":
        save_hdf5(output_path, output_data)
    elif fmt == "zarr":
        save_zarr(output_path, output_data)
    else:
        raise ValueError("Formato no soportado")

    print("Proceso completado.")


# ==========================================================
# CLI
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Filtrado optimizado para NPZ planos grandes"
    )

    parser.add_argument("input", help="Archivo NPZ de entrada")
    parser.add_argument("--out", required=True, help="Archivo de salida")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Archivo YAML que contiene la clave 'filters'"
    )

    parser.add_argument(
        "--format",
        default="hdf5",
        choices=["npz", "hdf5", "zarr"],
        help="Formato de salida (recomendado: hdf5)"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
      cfg = yaml.safe_load(f)

    filters = cfg.get("filters", {})


    filter_large_flat_npz(
        args.input,
        args.out,
        filters,
        args.format
    )


if __name__ == "__main__":
    main()
