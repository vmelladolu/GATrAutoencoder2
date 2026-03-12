"""
Convierte archivos .pt de simulación al formato HDF5 plano consumido por FlatSDHCALDataset.

Formato de salida HDF5:
  Hit-level (todos los hits concatenados):
    x, y, z    : coordenadas espaciales (= I, J, K)
    i, j, k    : índices enteros de celda (= I, J, K)
    thr        : umbral del hit (1, 2 ó 3)
    time       : tiempo del hit [exclusivo de datos de simulación]

  Por evento:
    offsets    : índices de inicio/fin en arrays de hits (longitud = n_eventos + 1)
    energy     : energía MC del evento
    nb_hits                : número total de hits
    ratio_thr3             : fracción de hits en umbral 3
    nb_hits_in_last_layer  : hits en la última capa (K=47)
    first_interaction_layer: primera capa con actividad densa
    PID_label              : etiqueta de partícula (0=electron, 1=pion, 2=muon, ...)
    event_id               : identificador de evento original

Salida: dos archivos HDF5 (<stem>_train.h5 y <stem>_test.h5) generados a partir de --out.

Uso:
    python convert_pt_to_hdf5.py input1.pt [input2.pt ...] --out dataset.h5
    python convert_pt_to_hdf5.py input1.pt --out dataset.h5 --config filters.yml
    python convert_pt_to_hdf5.py input1.pt --out dataset.h5 --train-ratio 0.9 --seed 123

Ejemplo de filtros (YAML):
    filters:
      nb_hits_in_last_layer: "<=5"
      first_interaction_layer: ">=10"
      PID_label: 1        # categórico: valor exacto

    energy_bins:          # opcional: lista de energías MC permitidas (GeV)
      - 10
      - 20
      - 30
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
import yaml

# ============================================================
# Carga y conversión desde archivos .pt
# ============================================================

# Columnas de hit_level_features (definido en pt_file_io.py):
#   0: I   1: J   2: K   3: time   4: thr_onehot[0]   5: thr_onehot[1]   6: thr_onehot[2]
# thr one-hot: [1,0,0]=umbral1  [0,1,0]=umbral2  [0,0,1]=umbral3  → decode: argmax+1
_COL_I, _COL_J, _COL_K, _COL_TIME = 0, 1, 2, 3
_COL_THR_ONEHOT = slice(4, 7)


def load_pt_file(path: str) -> list[dict]:
    """
    Carga un archivo .pt que contiene una lista de dicts con la estructura
    producida por pt_file_io.build_raw_events:
      {
        'hit_level_features': Tensor (N_hits, 7),  # [I,J,K,time,thr_oh1,thr_oh2,thr_oh3]
        'nb_hits':                int,
        'ratio_thr3':             float,
        'nb_hits_in_last_layer':  int,
        'first_interaction_layer':float,
        'PID_label':              int,
        'mc_energy':              float,
        'event_id':               str,
      }
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(obj, list):
        raise ValueError(
            f"Se esperaba una lista de dicts en '{path}', "
            f"pero se obtuvo {type(obj)}."
        )
    if len(obj) == 0:
        raise ValueError(f"El archivo '{path}' contiene una lista vacía.")
    if not isinstance(obj[0], dict):
        raise ValueError(
            f"Los elementos de '{path}' deben ser dicts, "
            f"pero se obtuvo {type(obj[0])}."
        )

    required_keys = {
        "hit_level_features", "nb_hits", "ratio_thr3",
        "nb_hits_in_last_layer", "first_interaction_layer",
        "PID_label", "mc_energy", "event_id",
    }
    missing = required_keys - obj[0].keys()
    if missing:
        raise KeyError(f"Faltan claves en los eventos de '{path}': {missing}")

    return obj


def build_events_from_pt_list(raw_events: list[dict]) -> list[dict]:
    """
    Convierte la lista de dicts producida por pt_file_io al formato interno
    utilizado por write_hdf5.

    La columna thr viene codificada como one-hot (cols 4-6); se decodifica
    de vuelta a valores enteros 1/2/3 mediante argmax+1.
    """
    events = []
    for raw in tqdm(raw_events, desc="Convirtiendo eventos"):
        feats = raw["hit_level_features"]
        if isinstance(feats, torch.Tensor):
            feats = feats.numpy()

        I    = feats[:, _COL_I].astype(np.float32)
        J    = feats[:, _COL_J].astype(np.float32)
        K    = feats[:, _COL_K].astype(np.float32)
        time = feats[:, _COL_TIME].astype(np.float32)
        # Decodificar one-hot → thr ∈ {1, 2, 3}
        thr  = (np.argmax(feats[:, _COL_THR_ONEHOT], axis=1) + 1).astype(np.float32)

        events.append(
            {
                # Hit-level
                "I":    I,
                "J":    J,
                "K":    K,
                "thr":  thr,
                "time": time,
                # Por evento (ya calculados en pt_file_io)
                "nb_hits":               int(raw["nb_hits"]),
                "ratio_thr3":            float(raw["ratio_thr3"]),
                "nb_hits_in_last_layer": int(raw["nb_hits_in_last_layer"]),
                "first_interaction_layer": float(raw["first_interaction_layer"]),
                "PID_label":             int(raw["PID_label"]),
                "mc_energy":             float(raw["mc_energy"]),
                "event_id":              raw["event_id"],
            }
        )
    return events


# ============================================================
# Filtrado
# ============================================================

def _parse_operator(value: str) -> tuple[str, float]:
    if value.startswith((">=", "<=")):
        return value[:2], float(value[2:])
    if value.startswith((">", "<")):
        return value[0], float(value[1:])
    raise ValueError(f"Operador no reconocido en '{value}'. Use >=, <=, >, <")


def apply_filters(events: list[dict], filters: dict, energy_bins: list | None) -> list[dict]:
    """
    Aplica filtros sobre la lista de eventos:
      - filters: dict con claves nb_hits_in_last_layer, first_interaction_layer (continuo)
                 y PID_label (categórico).
      - energy_bins: lista de energías MC permitidas (o None para no filtrar).
    """
    CONTINUOUS_KEYS = {"nb_hits_in_last_layer", "first_interaction_layer"}
    CATEGORICAL_KEYS = {"PID_label"}

    n_total = len(events)
    passed = []

    for evt in events:
        ok = True

        for key, value in filters.items():
            evt_val = evt.get(key)
            if evt_val is None:
                raise KeyError(f"Clave de filtro '{key}' no encontrada en el evento")

            if key in CONTINUOUS_KEYS:
                if isinstance(value, str) and value[0] in "><":
                    op, threshold = _parse_operator(value)
                    if op == ">=" and not (evt_val >= threshold):
                        ok = False
                    elif op == "<=" and not (evt_val <= threshold):
                        ok = False
                    elif op == ">" and not (evt_val > threshold):
                        ok = False
                    elif op == "<" and not (evt_val < threshold):
                        ok = False
                else:
                    # Comparación directa
                    if evt_val != float(value):
                        ok = False

            elif key in CATEGORICAL_KEYS:
                if evt_val != int(value):
                    ok = False

            else:
                # Clave desconocida: intentar comparación directa
                if evt_val != value:
                    ok = False

            if not ok:
                break

        if ok and energy_bins is not None:
            if evt["mc_energy"] not in energy_bins:
                ok = False

        if ok:
            passed.append(evt)

    n_passed = len(passed)
    print(
        f"[Filtros] {n_passed}/{n_total} eventos pasan "
        f"({100 * n_passed / max(1, n_total):.1f}%)"
    )
    return passed


# ============================================================
# Split train / test
# ============================================================

def split_events(
    events: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Divide aleatoriamente los eventos en conjuntos de entrenamiento y test.

    Args:
        events      : lista completa de eventos filtrados.
        train_ratio : fracción destinada a entrenamiento (0 < train_ratio < 1).
        seed        : semilla para reproducibilidad del shuffle.

    Returns:
        (train_events, test_events)
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio debe estar en (0, 1), recibido: {train_ratio}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(events))

    n_train = int(len(events) * train_ratio)
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    train_events = [events[i] for i in train_idx]
    test_events  = [events[i] for i in test_idx]

    print(
        f"[Split] train: {len(train_events)} eventos  |  "
        f"test: {len(test_events)} eventos  "
        f"(ratio={train_ratio:.2f}, seed={seed})"
    )
    return train_events, test_events


def _derive_split_paths(out_path: str) -> tuple[str, str]:
    """
    Deriva rutas de salida train/test a partir de --out.
    Ejemplo: 'data/dataset.h5' → ('data/dataset_train.h5', 'data/dataset_test.h5')
    """
    p = Path(out_path)
    stem = p.stem          # sin extensión
    suffix = p.suffix      # '.h5' / '.hdf5'
    parent = p.parent
    if not os.path.exists(parent):
        os.makedirs(parent)
    return str(parent / f"{stem}_train{suffix}"), str(parent / f"{stem}_test{suffix}")


# ============================================================
# Escritura HDF5
# ============================================================

def write_hdf5(events: list[dict], out_path: str, chunk_hits: int = 1_000_000) -> None:
    """Escribe los eventos en formato HDF5 plano compatible con FlatSDHCALDataset."""

    if not events:
        print("[AVISO] No hay eventos para escribir. El archivo no se generará.")
        return

    n_events = len(events)
    total_hits = sum(e["nb_hits"] for e in events)

    print(f"Escribiendo {n_events} eventos ({total_hits} hits totales) en '{out_path}'...")

    # Pre-alocar arrays de hits
    x_arr   = np.empty(total_hits, dtype=np.float32)
    y_arr   = np.empty(total_hits, dtype=np.float32)
    z_arr   = np.empty(total_hits, dtype=np.float32)
    i_arr   = np.empty(total_hits, dtype=np.float32)
    j_arr   = np.empty(total_hits, dtype=np.float32)
    k_arr   = np.empty(total_hits, dtype=np.float32)
    thr_arr = np.empty(total_hits, dtype=np.float32)
    time_arr = np.empty(total_hits, dtype=np.float32)

    # Arrays por evento
    offsets_arr              = np.zeros(n_events + 1, dtype=np.int64)
    energy_arr               = np.empty(n_events, dtype=np.float32)
    nb_hits_arr              = np.empty(n_events, dtype=np.int32)
    ratio_thr3_arr           = np.empty(n_events, dtype=np.float32)
    nb_hits_last_arr         = np.empty(n_events, dtype=np.int32)
    first_interaction_arr    = np.empty(n_events, dtype=np.int32)
    pid_arr                  = np.empty(n_events, dtype=np.int32)
    event_id_arr             = np.empty(n_events, dtype=object)   # strings

    write_pos = 0
    for idx, evt in enumerate(tqdm(events, desc="Construyendo arrays")):
        n = evt["nb_hits"]

        x_arr[write_pos : write_pos + n]    = evt["I"]
        y_arr[write_pos : write_pos + n]    = evt["J"]
        z_arr[write_pos : write_pos + n]    = evt["K"]
        i_arr[write_pos : write_pos + n]    = evt["I"]
        j_arr[write_pos : write_pos + n]    = evt["J"]
        k_arr[write_pos : write_pos + n]    = evt["K"]
        thr_arr[write_pos : write_pos + n]  = evt["thr"]
        time_arr[write_pos : write_pos + n] = evt["time"]

        offsets_arr[idx + 1]            = write_pos + n
        energy_arr[idx]                 = evt["mc_energy"]
        nb_hits_arr[idx]                = evt["nb_hits"]
        ratio_thr3_arr[idx]             = evt["ratio_thr3"]
        nb_hits_last_arr[idx]           = evt["nb_hits_in_last_layer"]
        first_interaction_arr[idx]      = evt["first_interaction_layer"]
        pid_arr[idx]                    = evt["PID_label"]
        event_id_arr[idx]               = str(evt["event_id"])

        write_pos += n

    with h5py.File(out_path, "w") as f:

        def _ds(name, arr):
            if arr.ndim == 1 and len(arr) > chunk_hits:
                chunks = (min(chunk_hits, len(arr)),)
            else:
                chunks = True
            f.create_dataset(name, data=arr, compression="lzf", chunks=chunks)

        # Hit-level
        _ds("x",    x_arr)
        _ds("y",    y_arr)
        _ds("z",    z_arr)
        _ds("i",    i_arr)
        _ds("j",    j_arr)
        _ds("k",    k_arr)
        _ds("thr",  thr_arr)
        _ds("time", time_arr)

        # Por evento
        _ds("offsets",               offsets_arr)
        _ds("energy",                energy_arr)
        _ds("nb_hits",               nb_hits_arr)
        _ds("ratio_thr3",            ratio_thr3_arr)
        _ds("nb_hits_in_last_layer", nb_hits_last_arr)
        _ds("first_interaction_layer", first_interaction_arr)
        _ds("PID_label",             pid_arr)
        # event_id como strings (variable-length UTF-8)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("event_id", data=list(event_id_arr), dtype=dt)

    print(f"[OK] Archivo guardado: {out_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convierte archivos .pt de simulación a HDF5 plano "
            "compatible con FlatSDHCALDataset (GATrRegressor)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Uno o más archivos .pt de entrada",
    )
    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Prefijo del archivo HDF5 de salida (p. ej. dataset_sim.h5). "
            "Se generarán <stem>_train.h5 y <stem>_test.h5 automáticamente."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "YAML con filtros y/o energy_bins. "
            "Ejemplo:\n"
            "  filters:\n"
            "    nb_hits_in_last_layer: '<=5'\n"
            "    PID_label: 1\n"
            "  energy_bins: [10, 20, 30]"
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fracción de eventos para entrenamiento (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para el split train/test (default: 42)",
    )
    parser.add_argument(
        "--chunk-hits",
        type=int,
        default=1_000_000,
        help="Tamaño de chunk HDF5 para arrays de hits (default: 1 000 000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Leer configuración ---
    filters: dict = {}
    energy_bins: list | None = None

    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        filters = cfg.get("filters", {})
        raw_bins = cfg.get("energy_bins", None)
        if raw_bins is not None:
            energy_bins = [float(b) for b in raw_bins]
            print(f"[Config] Bins de energía seleccionados: {energy_bins}")
        if filters:
            print(f"[Config] Filtros activos: {filters}")

    # --- Cargar y procesar archivos .pt ---
    all_events: list[dict] = []

    for pt_path in args.inputs:
        print(f"\nCargando '{pt_path}'...")
        raw_events = load_pt_file(pt_path)
        print(f"  Eventos cargados: {len(raw_events)}")
        events = build_events_from_pt_list(raw_events)
        all_events.extend(events)

    print(f"\nTotal de eventos cargados: {len(all_events)}")

    # --- Aplicar filtros ---
    if filters or energy_bins is not None:
        all_events = apply_filters(all_events, filters, energy_bins)
    else:
        print("[Info] Sin filtros activos. Se exportan todos los eventos.")

    # --- Split train / test ---
    train_path, test_path = _derive_split_paths(args.out)
    train_events, test_events = split_events(
        all_events,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    # --- Escribir HDF5 ---
    print(f"\n--- Escribiendo train → '{train_path}' ---")
    write_hdf5(train_events, train_path, chunk_hits=args.chunk_hits)

    print(f"\n--- Escribiendo test  → '{test_path}' ---")
    write_hdf5(test_events, test_path, chunk_hits=args.chunk_hits)


if __name__ == "__main__":
    main()
