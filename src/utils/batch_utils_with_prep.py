
import os

import torch
import torch.nn as nn
import yaml
from torch_scatter import scatter_sum


# Module-level stats cache — evita releer el YAML o recalcular en cada batch
_STATS_CACHE: dict = {}


def _compute_batch_stats(mv_v_part, mv_s_part, thr, use_scalar, use_one_hot, time=None) -> dict:
    """Calcula mean/std por feature a partir de los tensores del batch actual."""
    def _s(t):
        t_f = t.detach().float().view(-1)
        return {"mean": float(t_f.mean()), "std": float(max(float(t_f.std()), 1e-8))}

    stats = {
        "x": _s(mv_v_part[:, 0]),
        "y": _s(mv_v_part[:, 1]),
        "z": _s(mv_v_part[:, 2]),
    }
    if use_scalar:
        stats["k"] = _s(mv_s_part)
    if not use_one_hot:
        stats["thr"] = _s(thr)
    if time is not None:
        stats["time"] = _s(time)
    return stats


def _get_stats(stats_path, mv_v_part, mv_s_part, thr, use_scalar, use_one_hot, time=None) -> dict:
    """
    Devuelve el diccionario de estadísticas para z-norm.

    Prioridad:
      1. Si ya está en el caché en memoria → devuelve directamente.
      2. Si stats_path existe en disco → carga el YAML y guarda en caché.
      3. En caso contrario → calcula desde el batch actual, guarda en disco
         (si se proporcionó stats_path) y guarda en caché.

    Tras la primera llamada, todas las siguientes son O(1) desde el caché.
    """
    cache_key = stats_path if stats_path is not None else "__default__"

    if cache_key in _STATS_CACHE:
        return _STATS_CACHE[cache_key]

    if stats_path is not None and os.path.exists(stats_path):
        print(f"[z_norm] Cargando stats desde '{stats_path}'")
        with open(stats_path, "r") as f:
            raw = yaml.safe_load(f)
        stats = raw.get("stats", raw) if isinstance(raw, dict) else raw
    else:
        print("[z_norm] Stats no encontradas. Calculando desde el primer batch...")
        stats = _compute_batch_stats(mv_v_part, mv_s_part, thr, use_scalar, use_one_hot, time)
        if stats_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
            with open(stats_path, "w") as f:
                yaml.dump({"norm_type": "z_norm", "stats": stats}, f, default_flow_style=False)
            print(f"[z_norm] Stats guardadas en '{stats_path}'")

    _STATS_CACHE[cache_key] = stats
    return stats


def _znorm(tensor: torch.Tensor, key: str, stats: dict) -> torch.Tensor:
    """Aplica z-score: (tensor - mean) / std usando stats[key]."""
    s = stats.get(key)
    if s is None:
        return tensor
    return (tensor - s["mean"]) / s["std"]


def build_batch(batch, use_scalar=False, use_energy=False, use_one_hot=False, use_log=False, z_norm=False, stats=None, use_time=False, norm_applied=False):
        """
        Convierte un Batch de PyG a los tensores que espera el modelo.

        Asume que:
            - batch.pos = (N, 3)
            - batch.x = (N, 6) con [i, j, k, thr1, thr2, thr3]
            - batch.batch = (N,) índices de evento

        Si z_norm=True, aplica normalización z-score a las posiciones,
        profundidad (si use_scalar), threshold escalar (si no use_one_hot)
        y tiempo (si use_time). Las estadísticas se cargan desde el fichero
        YAML indicado en 'stats'; si no existe, se calculan a partir del
        primer batch y se guardan en disco. Las llamadas sucesivas usan el
        caché en memoria.
        """

        # Posiciones
        mv_v_part = batch.pos

        N = batch.x.shape[0]
        device = batch.x.device

        # Si se va a usar la profundidad
        if use_scalar:
          mv_s_part = batch.k
        else:
          mv_s_part = torch.zeros((N, 1), dtype=torch.float32).to(device) # Placeholder para la parte escalar (profundidad)

        # Componente de threshold: valor escalar o one-hot
        if use_one_hot:
            thr_component = torch.cat([batch.thr1, batch.thr2, batch.thr3], dim=1)
        else:
            thr_component = batch.thr

        # Componente de tiempo (opcional)
        time_component = (
            batch.time
            if use_time and hasattr(batch, "time") and batch.time is not None
            else None
        )

        # ── Z-score normalization ──────────────────────────────────────────
        if z_norm and not norm_applied:
            _stats = _get_stats(
                stats, mv_v_part, mv_s_part, thr_component,
                use_scalar, use_one_hot, time_component,
            )
            mv_v_part = torch.stack([
                _znorm(mv_v_part[:, 0], "x", _stats),
                _znorm(mv_v_part[:, 1], "y", _stats),
                _znorm(mv_v_part[:, 2], "z", _stats),
            ], dim=1)
            if use_scalar:
                mv_s_part = _znorm(mv_s_part, "k", _stats)
            if not use_one_hot:
                thr_component = _znorm(thr_component, "thr", _stats)
            if time_component is not None:
                time_component = _znorm(time_component, "time", _stats)
        # ──────────────────────────────────────────────────────────────────

        scalars = thr_component
        if time_component is not None:
            scalars = torch.cat([scalars, time_component], dim=1)

        if use_energy and hasattr(batch, "energy"):
            logenergy = batch.energy.view(-1, 1)
            if use_log and not norm_applied:
                logenergy = torch.log(logenergy + 1e-6)  # Evitar log(0)
        else:
            logenergy = None

        batch_idx = batch.batch

        # Per-threshold hit counts (Cambio 4)
        # Always compute when thr1/thr2/thr3 are available (they are pre-computed
        # in FlatSDHCALDataset regardless of use_one_hot).
        if hasattr(batch, 'thr1') and batch.thr1 is not None:
            n_thr1 = scatter_sum(batch.thr1.squeeze(-1).float(), batch_idx, dim=0)
            n_thr2 = scatter_sum(batch.thr2.squeeze(-1).float(), batch_idx, dim=0)
            n_thr3 = scatter_sum(batch.thr3.squeeze(-1).float(), batch_idx, dim=0)
        else:
            n_thr1 = n_thr2 = n_thr3 = None

        return {
                "mv_v_part": mv_v_part,
                "mv_s_part": mv_s_part,
                "scalars": scalars,
                "logenergy": logenergy,
                "batch_idx": batch_idx,
                "n_thr1": n_thr1,
                "n_thr2": n_thr2,
                "n_thr3": n_thr3,
        }
