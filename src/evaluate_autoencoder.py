import os
import sys
import time
import argparse
import numpy as np
import yaml
import torch
from torch_geometric.loader import DataLoader

from models.gatr_autoencoder import GATrAutoencoder
from utils.datasets import FlatSDHCALDataset, _load_or_compute_stats
from utils.batch_utils import build_batch
from train_autoencoder import (
    _log,
    _plot_event_3d,
    _plot_event_projections,
    _plot_error_distributions,
)
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError:
    plt = None

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
LABEL_INT_TO_NAME = {0: "electron", 1: "pion", 2: "muon"}


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalúa un autoencoder GATr entrenado sobre datos SDHCAL."
    )

    # Checkpoint y config
    parser.add_argument("--ckpt", required=True, type=str,
                        help="Ruta al checkpoint (.pt o .ckpt de Lightning)")
    parser.add_argument("--cfg", "-c", type=str, default="config/model_cfg.yml",
                        help="Archivo YAML con la configuración del modelo")

    # Ficheros de datos (todos opcionales; al menos uno requerido)
    parser.add_argument("--electron_path", type=str, default=None,
                        help="HDF5 de electrones")
    parser.add_argument("--pion_path", type=str, default=None,
                        help="HDF5 de piones")
    parser.add_argument("--muon_path", type=str, default=None,
                        help="HDF5 de muones")

    # Preprocesado (mismos flags que train_autoencoder.py)
    parser.add_argument("--use_scalar", action="store_true",
                        help="Incluye la coordenada escalar (profundidad) en el modelo")
    parser.add_argument("--use_one_hot", action="store_true",
                        help="Usa one-hot encoding para las clases de threshold")
    parser.add_argument("--z_norm", action="store_true",
                        help="Alias para --norm z_norm")
    parser.add_argument("--norm", choices=["z_norm", "minmax"], default=None,
                        help="Tipo de normalización (z_norm | minmax)")
    parser.add_argument("--norm_yaml", type=str, default=None,
                        help="YAML de estadísticas de normalización. Si no existe, se calcula.")

    # Inferencia
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None,
                        help="Dispositivo (ej. 'cuda:0' o 'cpu')")
    parser.add_argument("--max_events", type=int, default=0,
                        help="Límite de eventos por clase (0 = todos)")

    # Salida
    parser.add_argument("--output_dir", "-o", type=str, default="eval_output",
                        help="Directorio donde guardar plots y métricas")
    parser.add_argument("--n_grid", type=int, default=8,
                        help="Resolución del grid para el plot PCA-grid (n_grid x n_grid)")

    # Reducción de dimensionalidad no lineal (t-SNE / UMAP)
    parser.add_argument("--max_reduction_events", type=int, default=5000,
                        help="Máx. eventos usados para t-SNE/UMAP (0 = todos; t-SNE es O(n²))")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0,
                        help="Perplexity de t-SNE (típico: 5–50)")
    parser.add_argument("--umap_n_neighbors", type=int, default=15,
                        help="n_neighbors de UMAP")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="min_dist de UMAP")

    # Cargar defaults del YAML de training
    pre_args, _ = parser.parse_known_args()
    yaml_training = {}
    try:
        with open(pre_args.cfg, "r") as f:
            yaml_all = yaml.safe_load(f)
        yaml_training = yaml_all.get("training", {})
    except FileNotFoundError:
        pass
    hardcoded = {"batch_size": 64}
    hardcoded.update(yaml_training)
    parser.set_defaults(**hardcoded)

    return parser.parse_args()


# ============================================================
# Carga del modelo
# ============================================================

def load_model(ckpt_path, cfg_models, device):
    """
    Carga GATrAutoencoder desde un checkpoint.
    Soporta:
      - formato propio (.pt):    clave 'model_state_dict'
      - formato Lightning (.ckpt): clave 'state_dict' con prefijo 'model.'
      - state_dict directo
    """
    cfg_enc = cfg_models["encoder"]
    cfg_dec = dict(cfg_models["decoder"])
    cfg_agg = cfg_models.get("aggregation", {"type": "mean"})
    cfg_lat = cfg_models.get("latent", {})
    use_vae = cfg_lat.get("use_vae", False)
    event_embed_dim = cfg_lat.get("event_embed_dim", 32)

    if cfg_dec["out_s_channels"] != cfg_enc["in_s_channels"]:
        cfg_dec["out_s_channels"] = cfg_enc["in_s_channels"]

    model = GATrAutoencoder(
        cfg_enc=cfg_enc,
        cfg_dec=cfg_dec,
        cfg_agg=cfg_agg,
        latent_s_channels=2,
        use_vae=use_vae,
        event_embed_dim=event_embed_dim,
    )

    raw = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in raw:
        model.load_state_dict(raw["model_state_dict"])
        _log(f"Checkpoint cargado (train_autoencoder.py): {ckpt_path}")
    elif "state_dict" in raw:
        # Lightning: las claves tienen prefijo "model."
        sd = {k.removeprefix("model."): v for k, v in raw["state_dict"].items()
              if k.startswith("model.")}
        if not sd:
            sd = raw["state_dict"]
        model.load_state_dict(sd)
        _log(f"Checkpoint cargado (Lightning .ckpt): {ckpt_path}")
    else:
        model.load_state_dict(raw)
        _log(f"Checkpoint cargado (state_dict directo): {ckpt_path}")

    model.to(device)
    model.eval()
    return model


# ============================================================
# Carga de datasets con etiqueta de clase
# ============================================================

def load_labeled_dataset(path, label_int, label_name,
                          preprocessing_cfg, norm_type,
                          batch_size, max_events=0):
    """
    Carga un fichero HDF5, aplica preprocesado y devuelve (loader, label_int, label_name).
    max_events=0 usa todos los eventos disponibles.
    """
    _log(f"Cargando {label_name} desde: {path}")
    ds = FlatSDHCALDataset(path, preprocessing_cfg=None, filters=None)

    all_idx = np.arange(ds.len())
    if max_events > 0 and max_events < len(all_idx):
        rng = np.random.default_rng(42)
        all_idx = np.sort(rng.choice(all_idx, size=max_events, replace=False))

    if norm_type in ("z_norm", "minmax"):
        stats = _load_or_compute_stats(ds, all_idx, preprocessing_cfg, norm_type)
        ds._apply_preprocessing_inplace(stats, norm_type, preprocessing_cfg)

    subset = torch.utils.data.Subset(ds, all_idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    _log(f"  {label_name}: {len(all_idx)} eventos, {len(loader)} batches")
    return loader, label_int, label_name


# ============================================================
# Inferencia
# ============================================================

def run_inference(model, loaders_with_labels, device, use_scalar, use_one_hot):
    """
    Corre el modelo en modo eval sobre todos los loaders.

    Retorna dict con:
        embeddings        (N_ev, D)   — event_embedding del VAE/agregación
        labels            (N_ev,)     — etiqueta de clase (int) por evento
        energies          (N_ev,)     — energía del evento en GeV (0 si no disponible)
        xyz_true          (N_h, 3)    — coordenadas reales de los hits
        xyz_rec           (N_h, 3)    — coordenadas reconstruidas
        k_true            (N_h, 1)    — profundidad real
        k_rec             (N_h, 1)    — profundidad reconstruida
        thr_true          (N_h, F)    — threshold real
        thr_rec           (N_h, F)    — threshold reconstruido
        hit_labels        (N_h,)      — clase del hit (heredada del evento)
        event_ids         (N_h,)      — ID global de evento para cada hit
        per_event_labels  (N_ev,)     — clase de cada evento (global)
        event_hits        list[array] — lista de (Nhits_i, 3) xyz por evento
    """
    acc = {k: [] for k in [
        "embeddings", "labels", "energies", "xyz_true", "xyz_rec",
        "k_true", "k_rec", "thr_true", "thr_rec",
        "hit_labels", "event_ids", "per_event_labels",
    ]}
    event_hits = []        # list of (Nhits_i, 3) — one entry per global event
    global_ev_offset = 0  # running count of events processed so far

    for loader, label_int, label_name in loaders_with_labels:
        _log(f"  Inferencia {label_name} ({len(loader)} batches)...")
        t0 = time.time()
        for batch in tqdm(loader):
            data = build_batch(batch, use_scalar=use_scalar, use_one_hot=use_one_hot)
            mv_v = data["mv_v_part"].to(device)
            mv_s = data["mv_s_part"].to(device)
            sc   = data["scalars"].to(device)
            bidx = data["batch_idx"].to(device)

            with torch.no_grad():
                out = model(mv_v, mv_s, sc, bidx)

            xyz_t  = mv_v.cpu().numpy()
            xyz_r  = out["point_rec"].cpu().numpy()
            k_t    = mv_s.cpu().numpy()
            k_r    = out["scalar_rec"].cpu().numpy()
            thr_t  = sc.cpu().numpy()
            thr_r  = out["s_rec"].cpu().numpy()
            bidx_np = bidx.cpu().numpy()
            embs   = out["event_embedding"].cpu().numpy()
            B = embs.shape[0]

            # Hits globales de cada evento en este batch
            for ev in range(B):
                mask = bidx_np == ev
                event_hits.append(xyz_t[mask])

            global_ids = bidx_np + global_ev_offset

            if hasattr(batch, "energy") and batch.energy is not None:
                energies_batch = batch.energy.cpu().numpy().reshape(B)
            else:
                energies_batch = np.zeros(B, dtype=np.float32)

            acc["embeddings"].append(embs)
            acc["labels"].append(np.full(B, label_int, dtype=np.int32))
            acc["energies"].append(energies_batch)
            acc["xyz_true"].append(xyz_t)
            acc["xyz_rec"].append(xyz_r)
            acc["k_true"].append(k_t)
            acc["k_rec"].append(k_r)
            acc["thr_true"].append(thr_t)
            acc["thr_rec"].append(thr_r)
            acc["hit_labels"].append(np.full(len(bidx_np), label_int, dtype=np.int32))
            acc["event_ids"].append(global_ids)
            acc["per_event_labels"].append(np.full(B, label_int, dtype=np.int32))

            global_ev_offset += B

        _log(f"    {label_name} listo en {time.time()-t0:.1f}s")

    out = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    out["event_hits"] = event_hits
    return out


# ============================================================
# Métricas por evento
# ============================================================

def compute_per_event_mse(results):
    """MSE XYZ promediado por evento. Devuelve array (N_events,)."""
    xyz_t = results["xyz_true"]
    xyz_r = results["xyz_rec"]
    ev_ids = results["event_ids"]

    sq_err = ((xyz_r - xyz_t) ** 2).sum(axis=1)
    n_ev = int(ev_ids.max()) + 1
    mse = np.zeros(n_ev, dtype=np.float64)
    cnt = np.zeros(n_ev, dtype=np.int64)
    np.add.at(mse, ev_ids, sq_err)
    np.add.at(cnt, ev_ids, 1)
    cnt = np.maximum(cnt, 1)
    return mse / cnt


# ============================================================
# Plots de reconstrucción
# ============================================================

def plot_reconstruction_errors(results, use_scalar, output_dir):
    """Histogramas de error relativo por variable. Guarda reconstruction_errors.png."""
    if plt is None:
        return {}

    eps = 1e-6
    xyz_t = results["xyz_true"]
    xyz_r = results["xyz_rec"]
    k_t   = results["k_true"]
    k_r   = results["k_rec"]
    thr_t = results["thr_true"]
    thr_r = results["thr_rec"]

    r_x   = (xyz_r[:, 0] - xyz_t[:, 0]) / np.abs(xyz_t[:, 0]).clip(min=eps)
    r_y   = (xyz_r[:, 1] - xyz_t[:, 1]) / np.abs(xyz_t[:, 1]).clip(min=eps)
    r_z   = (xyz_r[:, 2] - xyz_t[:, 2]) / np.abs(xyz_t[:, 2]).clip(min=eps)
    r_k   = (k_r[:, 0]   - k_t[:, 0])   / np.abs(k_t[:, 0]).clip(min=eps)
    r_thr = ((thr_r - thr_t) / np.abs(thr_t).clip(min=eps)).reshape(-1)

    fig = _plot_error_distributions(r_x, r_y, r_z, r_k, r_thr, use_scalar=use_scalar)
    if fig is not None:
        path = os.path.join(output_dir, "reconstruction_errors.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        _log(f"Guardado: {path}")

    return {
        "mse_x":   float(np.mean((xyz_r[:, 0] - xyz_t[:, 0]) ** 2)),
        "mse_y":   float(np.mean((xyz_r[:, 1] - xyz_t[:, 1]) ** 2)),
        "mse_z":   float(np.mean((xyz_r[:, 2] - xyz_t[:, 2]) ** 2)),
        "mse_k":   float(np.mean((k_r - k_t) ** 2)) if use_scalar else None,
        "mse_thr": float(np.mean((thr_r - thr_t) ** 2)),
    }


def plot_top_k_events(results, per_event_mse, k, output_dir):
    """
    Genera plots 3D y proyecciones 2D de los k mejores y k peores eventos
    (ranking por MSE XYZ).
    """
    if plt is None:
        return

    n_ev = len(per_event_mse)
    k = min(k, n_ev // 2)
    sorted_idx = np.argsort(per_event_mse)
    best_idx  = sorted_idx[:k]
    worst_idx = sorted_idx[n_ev - k:][::-1]

    xyz_t     = results["xyz_true"]
    xyz_r     = results["xyz_rec"]
    ev_ids    = results["event_ids"]
    ev_labels = results["per_event_labels"]

    for group_name, indices in [("best", best_idx), ("worst", worst_idx)]:
        for rank, ev_idx in enumerate(indices):
            hit_mask = ev_ids == ev_idx
            xyz_orig = xyz_t[hit_mask]
            xyz_recon = xyz_r[hit_mask]
            ev_label  = int(ev_labels[ev_idx]) if ev_idx < len(ev_labels) else -1
            particle  = LABEL_INT_TO_NAME.get(ev_label, f"class_{ev_label}")
            mse_val   = per_event_mse[ev_idx]
            suptitle  = f"#{rank+1} {group_name} — {particle} | MSE={mse_val:.4f}"

            fig3d = _plot_event_3d(xyz_orig, xyz_recon)
            if fig3d is not None:
                fig3d.suptitle(suptitle, fontsize=10)
                path = os.path.join(output_dir, f"{group_name}_event_{rank+1}_3d.png")
                fig3d.savefig(path, dpi=120, bbox_inches="tight")
                plt.close(fig3d)

            fig2d = _plot_event_projections(xyz_orig, xyz_recon)
            if fig2d is not None:
                fig2d.suptitle(suptitle, fontsize=10)
                path = os.path.join(output_dir, f"{group_name}_event_{rank+1}_proj.png")
                fig2d.savefig(path, dpi=120, bbox_inches="tight")
                plt.close(fig2d)

    _log(f"Plots top-{k} best/worst guardados en {output_dir}/")


# ============================================================
# Helpers para plots del espacio latente
# ============================================================

def _active_classes(labels, loaders_with_labels):
    """Devuelve [(label_int, label_name, color), ...] de las clases presentes."""
    classes = []
    seen = set()
    for _, li, lname in loaders_with_labels:
        if li not in seen and np.any(labels == li):
            seen.add(li)
            classes.append((li, lname, COLORS[li % len(COLORS)]))
    return classes


# ============================================================
# Helpers genéricos de scatter 2D / 3D (usados por PCA, t-SNE, UMAP)
# ============================================================

def _subsample_for_reduction(embeddings, labels, max_events, energies=None):
    """
    Subsampleo estratificado para métodos costosos (t-SNE, UMAP).
    Devuelve (emb_sub, lab_sub, indices_sub, energies_sub).
    Con max_events=0 devuelve los arrays originales sin copiar.
    """
    if max_events <= 0 or len(embeddings) <= max_events:
        sel_idx = np.arange(len(embeddings))
        en_sub = energies[sel_idx] if energies is not None else None
        return embeddings, labels, sel_idx, en_sub
    classes = np.unique(labels)
    n_per_class = max(1, max_events // len(classes))
    rng = np.random.RandomState(42)
    selected = []
    for c in classes:
        idx = np.where(labels == c)[0]
        n = min(n_per_class, len(idx))
        selected.append(rng.choice(idx, size=n, replace=False))
    sel_idx = np.sort(np.concatenate(selected))
    en_sub = energies[sel_idx] if energies is not None else None
    return embeddings[sel_idx], labels[sel_idx], sel_idx, en_sub


_PARTICLE_CMAPS = {0: "Blues", 1: "Reds", 2: "Greens"}


def _plot_scatter_2d(z2d, labels, classes, axis_labels, title, output_dir, filename,
                     energies=None):
    """Scatter 2D coloreado por clase (y por energía si se proporcionan).

    Con ``energies`` (array shape (N,) en GeV), cada clase usa un gradiente
    logarítmico del colormap asociado a su color base.  Sin energías, el
    comportamiento es el mismo que antes (color sólido por clase).
    """
    if plt is None:
        return
    from matplotlib.lines import Line2D

    use_energy = energies is not None and np.any(energies > 0)

    if use_energy:
        from matplotlib.colors import LogNorm
        en_pos = np.where(energies > 0, energies, np.nan)
        vmin = np.nanmin(en_pos)
        vmax = np.nanmax(en_pos)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        n_classes = len(classes)
        fig_w = 7 + 1.2 * n_classes
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        scatters = []
        for li, lname, color in classes:
            mask = labels == li
            cmap = _PARTICLE_CMAPS.get(li, "Greys")
            sc = ax.scatter(z2d[mask, 0], z2d[mask, 1],
                            s=8, alpha=0.7,
                            c=energies[mask], cmap=cmap, norm=norm)
            scatters.append((sc, lname, color))
        # Colorbars: uno por clase, desplazados hacia la derecha
        for idx, (sc, lname, color) in enumerate(scatters):
            cbar = fig.colorbar(sc, ax=ax, pad=0.02 + idx * 0.10,
                                fraction=0.03, aspect=30)
            cbar.set_label(f"{lname} energy [GeV]", color=color)
            cbar.ax.yaxis.set_tick_params(color=color)
        # Leyenda de clase con marcadores de color sólido
        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color, markersize=8, label=lname)
            for _, lname, color in classes
        ]
        if len(classes) > 1:
            ax.legend(handles=legend_handles, loc="upper left")
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        for li, lname, color in classes:
            mask = labels == li
            ax.scatter(z2d[mask, 0], z2d[mask, 1], s=8, alpha=0.5, c=color, label=lname)
        if len(classes) > 1:
            ax.legend()

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    _log(f"Guardado: {path}")


def _plot_scatter_3d(z3d, labels, classes, axis_labels, title, output_dir, filename_base,
                     energies=None):
    """
    Scatter 3D coloreado por clase (y por energía si se proporcionan).
    Genera filename_base.png (matplotlib) y filename_base_interactive.html (plotly si disponible).
    """
    use_energy = energies is not None and np.any(energies > 0)

    if plt is not None:
        from matplotlib.lines import Line2D
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        if use_energy:
            from matplotlib.colors import LogNorm
            en_pos = np.where(energies > 0, energies, np.nan)
            vmin, vmax = np.nanmin(en_pos), np.nanmax(en_pos)
            norm = LogNorm(vmin=vmin, vmax=vmax)
            scatters = []
            for li, lname, color in classes:
                mask = labels == li
                cmap = _PARTICLE_CMAPS.get(li, "Greys")
                sc = ax.scatter(z3d[mask, 0], z3d[mask, 1], z3d[mask, 2],
                                s=5, alpha=0.5,
                                c=energies[mask], cmap=cmap, norm=norm)
                scatters.append((sc, lname, color))
            for idx, (sc, lname, color) in enumerate(scatters):
                cbar = fig.colorbar(sc, ax=ax, pad=0.05 + idx * 0.08,
                                    fraction=0.025, aspect=25, shrink=0.6)
                cbar.set_label(f"{lname} energy [GeV]", color=color)
            legend_handles = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color, markersize=8, label=lname)
                for _, lname, color in classes
            ]
            if len(classes) > 1:
                ax.legend(handles=legend_handles, loc="upper left")
        else:
            for li, lname, color in classes:
                mask = labels == li
                ax.scatter(z3d[mask, 0], z3d[mask, 1], z3d[mask, 2],
                           s=5, alpha=0.4, c=color, label=lname)
            if len(classes) > 1:
                ax.legend()
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
        ax.set_title(title)
        fig.tight_layout()
        path_png = os.path.join(output_dir, filename_base + ".png")
        fig.savefig(path_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        _log(f"Guardado: {path_png}")

    if HAS_PLOTLY:
        if use_energy:
            import math
            en_pos = np.where(energies > 0, energies, np.nan)
            log_min = math.log10(float(np.nanmin(en_pos)))
            log_max = math.log10(float(np.nanmax(en_pos)))
            _PLOTLY_COLORSCALES = {0: "Blues", 1: "Reds", 2: "Greens"}
            n_classes = len(classes)
            traces = []
            for idx, (li, lname, color) in enumerate(classes):
                mask = labels == li
                colorscale = _PLOTLY_COLORSCALES.get(li, "Greys")
                colorbar_x = 1.02 + idx * 0.12
                traces.append(go.Scatter3d(
                    x=z3d[mask, 0], y=z3d[mask, 1], z=z3d[mask, 2],
                    mode="markers",
                    marker=dict(
                        size=3, opacity=0.6,
                        color=np.log10(np.where(energies[mask] > 0, energies[mask], np.nan)),
                        colorscale=colorscale,
                        cmin=log_min, cmax=log_max,
                        colorbar=dict(
                            title=f"{lname}<br>log₁₀(E/GeV)",
                            x=colorbar_x, thickness=12, len=0.6,
                        ),
                    ),
                    name=lname,
                ))
        else:
            traces = [
                go.Scatter3d(
                    x=z3d[labels == li, 0],
                    y=z3d[labels == li, 1],
                    z=z3d[labels == li, 2],
                    mode="markers",
                    marker=dict(size=3, opacity=0.5, color=color),
                    name=lname,
                )
                for li, lname, color in classes
            ]
        fig_p = go.Figure(
            data=traces,
            layout=go.Layout(
                title=title + " (interactivo)",
                scene=dict(
                    xaxis_title=axis_labels[0],
                    yaxis_title=axis_labels[1],
                    zaxis_title=axis_labels[2],
                ),
            ),
        )
        path_html = os.path.join(output_dir, filename_base + "_interactive.html")
        fig_p.write_html(path_html)
        _log(f"Guardado: {path_html}")


# ============================================================
# PCA 2D
# ============================================================

def plot_pca_2d(embeddings, labels, loaders_with_labels, output_dir, energies=None):
    """
    Proyección PCA 2D del espacio latente, coloreada por clase de partícula.
    Con ``energies`` usa gradientes de color logarítmicos por clase.
    Devuelve (z2d, pca_object) para reutilizar en plot_pca_grid.
    """
    if plt is None or len(embeddings) < 4:
        return None
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        _log("sklearn no disponible — omitiendo PCA 2D.")
        return None

    pca = PCA(n_components=2)
    z2d = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_
    classes = _active_classes(labels, loaders_with_labels)
    _plot_scatter_2d(
        z2d, labels, classes,
        axis_labels=[f"PC1 ({var[0]*100:.1f}%)", f"PC2 ({var[1]*100:.1f}%)"],
        title="Espacio latente — PCA 2D",
        output_dir=output_dir, filename="pca_2d.png",
        energies=energies,
    )
    return z2d, pca


# ============================================================
# PCA 3D (matplotlib estático + plotly interactivo si disponible)
# ============================================================

def plot_pca_3d(embeddings, labels, loaders_with_labels, output_dir, energies=None):
    """
    Proyección PCA 3D:
      - Genera pca_3d.png (matplotlib estático).
      - Si plotly está instalado, genera pca_3d_interactive.html.
    Con ``energies`` usa gradientes de color logarítmicos por clase.
    """
    if len(embeddings) < 4:
        return
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        _log("sklearn no disponible — omitiendo PCA 3D.")
        return

    pca3 = PCA(n_components=3)
    z3d = pca3.fit_transform(embeddings)
    var = pca3.explained_variance_ratio_
    classes = _active_classes(labels, loaders_with_labels)
    _plot_scatter_3d(
        z3d, labels, classes,
        axis_labels=[f"PC{i+1} ({var[i]*100:.1f}%)" for i in range(3)],
        title="Espacio latente — PCA 3D",
        output_dir=output_dir, filename_base="pca_3d",
        energies=energies,
    )


# ============================================================
# t-SNE 2D y 3D
# ============================================================

def plot_tsne(embeddings, labels, loaders_with_labels, output_dir,
              event_hits=None, n_grid=8, max_events=5000, perplexity=30.0,
              energies=None):
    """
    Proyección t-SNE 2D y 3D del espacio latente.

    t-SNE preserva la estructura LOCAL (vecindades) mejor que PCA, a costa de
    perder la interpretación global de las distancias entre clusters.
    Complejidad O(n·log n) con la implementación Barnes-Hut de sklearn.

    Args:
        event_hits  list[array] o None — hits XYZ por evento para el grid (None = omitir grid)
        n_grid      int  — resolución del grid XZ
        max_events  int  — subsampleo estratificado (0 = todos; recomendado ≤ 10 000)
        perplexity  float — parámetro de t-SNE (típico 5–50)
    """
    if plt is None or len(embeddings) < 4:
        return
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        _log("sklearn no disponible — omitiendo t-SNE.")
        return

    emb_s, lab_s, sel_idx, en_s = _subsample_for_reduction(
        embeddings, labels, max_events, energies=energies)
    classes = _active_classes(lab_s, loaders_with_labels)
    n = len(emb_s)
    perp = min(perplexity, max(5.0, n / 4.0 - 1))  # sklearn requiere perplexity < n/4

    _log(f"Calculando t-SNE 2D (n={n}, perplexity={perp:.0f})...")
    t0 = time.time()
    z2d = TSNE(n_components=2, perplexity=perp, random_state=42,
               n_jobs=-1).fit_transform(emb_s)
    _log(f"  t-SNE 2D listo en {time.time()-t0:.1f}s")
    _plot_scatter_2d(
        z2d, lab_s, classes,
        axis_labels=["t-SNE 1", "t-SNE 2"],
        title=f"Espacio latente — t-SNE 2D (n={n})",
        output_dir=output_dir, filename="tsne_2d.png",
        energies=en_s,
    )
    if event_hits is not None:
        hits_sub = [event_hits[i] for i in sel_idx]
        plot_pca_grid(z2d, hits_sub, lab_s, loaders_with_labels,
                      n_grid, output_dir, method_name="t-SNE")

    _log(f"Calculando t-SNE 3D (n={n}, perplexity={perp:.0f})...")
    t0 = time.time()
    z3d = TSNE(n_components=3, perplexity=perp, random_state=42,
               n_jobs=-1).fit_transform(emb_s)
    _log(f"  t-SNE 3D listo en {time.time()-t0:.1f}s")
    _plot_scatter_3d(
        z3d, lab_s, classes,
        axis_labels=["t-SNE 1", "t-SNE 2", "t-SNE 3"],
        title=f"Espacio latente — t-SNE 3D (n={n})",
        output_dir=output_dir, filename_base="tsne_3d",
        energies=en_s,
    )


# ============================================================
# UMAP 2D y 3D (solo si umap-learn está instalado)
# ============================================================

def plot_umap(embeddings, labels, loaders_with_labels, output_dir,
              event_hits=None, n_grid=8, max_events=0, n_neighbors=15, min_dist=0.1,
              energies=None):
    """
    Proyección UMAP 2D y 3D del espacio latente.

    UMAP preserva tanto estructura LOCAL como GLOBAL mejor que t-SNE y es
    significativamente más rápido. Requiere el paquete `umap-learn`:
        pip install umap-learn

    Args:
        event_hits   list[array] o None — hits XYZ por evento para el grid (None = omitir grid)
        n_grid       int  — resolución del grid XZ
        max_events   int  — subsampleo (0 = todos; UMAP escala bien con n)
        n_neighbors  int  — tamaño del vecindario local (↑ → más estructura global)
        min_dist     float — distancia mínima en el espacio proyectado (↑ → puntos más dispersos)
    """
    try:
        import umap as umap_lib
    except ImportError:
        _log("umap-learn no instalado — omitiendo UMAP. (pip install umap-learn)")
        return

    if len(embeddings) < 4:
        return

    emb_s, lab_s, sel_idx, en_s = _subsample_for_reduction(
        embeddings, labels, max_events, energies=energies)
    classes = _active_classes(lab_s, loaders_with_labels)
    n = len(emb_s)

    _log(f"Calculando UMAP 2D (n={n}, n_neighbors={n_neighbors}, min_dist={min_dist})...")
    t0 = time.time()
    reducer2d = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors,
                               min_dist=min_dist, random_state=42)
    z2d = reducer2d.fit_transform(emb_s)
    _log(f"  UMAP 2D listo en {time.time()-t0:.1f}s")
    _plot_scatter_2d(
        z2d, lab_s, classes,
        axis_labels=["UMAP 1", "UMAP 2"],
        title=f"Espacio latente — UMAP 2D (n={n})",
        output_dir=output_dir, filename="umap_2d.png",
        energies=en_s,
    )
    if event_hits is not None:
        hits_sub = [event_hits[i] for i in sel_idx]
        plot_pca_grid(z2d, hits_sub, lab_s, loaders_with_labels,
                      n_grid, output_dir, method_name="UMAP")

    _log(f"Calculando UMAP 3D (n={n}, n_neighbors={n_neighbors}, min_dist={min_dist})...")
    t0 = time.time()
    reducer3d = umap_lib.UMAP(n_components=3, n_neighbors=n_neighbors,
                               min_dist=min_dist, random_state=42)
    z3d = reducer3d.fit_transform(emb_s)
    _log(f"  UMAP 3D listo en {time.time()-t0:.1f}s")
    _plot_scatter_3d(
        z3d, lab_s, classes,
        axis_labels=["UMAP 1", "UMAP 2", "UMAP 3"],
        title=f"Espacio latente — UMAP 3D (n={n})",
        output_dir=output_dir, filename_base="umap_3d",
        energies=en_s,
    )


# ============================================================
# PCA grid: celdas con proyección XZ del evento representativo
# ============================================================

def plot_pca_grid(z2d, event_hits, labels, loaders_with_labels, n_grid, output_dir,
                  method_name="PCA"):
    """
    Divide un espacio 2D (PCA, t-SNE, UMAP…) en una cuadrícula n_grid×n_grid.
    En cada celda dibuja la proyección XZ del evento más cercano al centro de la celda.
    El borde de cada inset está coloreado según la clase de partícula del evento.

    Args:
        z2d         (N_ev, 2)   — proyección 2D ya calculada
        event_hits  list[array] — lista de (Nhits_i, 3) xyz por evento (misma longitud que N_ev)
        labels      (N_ev,)     — etiqueta de clase por evento
        n_grid      int         — resolución del grid (e.g., 8 → cuadrícula 8×8)
        method_name str         — nombre del método para títulos y fichero ("PCA", "t-SNE", "UMAP")
    """
    if plt is None:
        return
    if len(z2d) < n_grid * n_grid:
        _log(f"Pocos eventos ({len(z2d)}) para un grid {n_grid}×{n_grid} — omitiendo plot_pca_grid.")
        return

    classes = _active_classes(labels, loaders_with_labels)

    # Rango del espacio PCA con pequeño margen
    x_min, x_max = z2d[:, 0].min(), z2d[:, 0].max()
    y_min, y_max = z2d[:, 1].min(), z2d[:, 1].max()
    mx = (x_max - x_min) * 0.02
    my = (y_max - y_min) * 0.02
    x_min -= mx; x_max += mx
    y_min -= my; y_max += my

    dx = (x_max - x_min) / n_grid
    dy = (y_max - y_min) / n_grid
    diag = np.sqrt(dx ** 2 + dy ** 2)   # distancia máxima tolerable para asignar un evento a la celda

    fig, ax = plt.subplots(figsize=(16, 16))

    # Fondo: scatter de todos los puntos coloreados por clase
    for li, lname, color in classes:
        mask = labels == li
        ax.scatter(z2d[mask, 0], z2d[mask, 1],
                   s=3, alpha=0.15, c=color, label=lname, zorder=1)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(f"{method_name} 1", fontsize=12)
    ax.set_ylabel(f"{method_name} 2", fontsize=12)
    ax.set_title(f"{method_name} grid ({n_grid}×{n_grid}) — proyección XZ de eventos representativos",
                 fontsize=13)
    if len(classes) > 1:
        ax.legend(loc="upper right", markerscale=4, fontsize=10)

    # Fracción de eje que ocupa cada celda
    cell_frac  = 1.0 / n_grid
    margin_frac = cell_frac * 0.08     # margen interior de la celda
    inset_frac  = cell_frac - 2 * margin_frac

    for i in range(n_grid):
        for j in range(n_grid):
            # Centro de la celda en espacio PCA
            cx = x_min + (i + 0.5) * dx
            cy = y_min + (j + 0.5) * dy

            # Evento más cercano al centro
            dists = np.sqrt((z2d[:, 0] - cx) ** 2 + (z2d[:, 1] - cy) ** 2)
            nearest = int(np.argmin(dists))

            if dists[nearest] > diag:
                continue   # celda vacía — no hay eventos suficientemente cercanos

            hits = event_hits[nearest]
            if len(hits) == 0:
                continue

            # Posición del inset en fracción de ejes (0-1 mapea al rango de datos del ax)
            # Celda (i, j) → fracción [i*cell_frac, (i+1)*cell_frac] x [j*cell_frac, ...]
            x0 = i * cell_frac + margin_frac
            y0 = j * cell_frac + margin_frac

            axins = ax.inset_axes([x0, y0, inset_frac, inset_frac])
            axins.scatter(hits[:, 0], hits[:, 2], s=1, alpha=0.8, color="k")
            axins.set_xticks([])
            axins.set_yticks([])

            # Borde coloreado según la clase del evento más cercano
            ev_label = int(labels[nearest])
            border_color = next(
                (c for li, _, c in classes if li == ev_label), "gray"
            )
            for spine in axins.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(1.5)

    fig.tight_layout()
    filename = f"{method_name.lower().replace('-', '').replace(' ', '_')}_grid_xz.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    _log(f"Guardado: {path}")

# ============================================================
# Pintar hits por cluster
# ============================================================
def plot_hits_by_cluster(event_hits, preds, output_dir):
    if plt is None:
        return

    clusters = [cid for cid in np.unique(preds) if cid != -1]
    n_clusters = len(clusters)

    n_cols = 4
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, cid in enumerate(clusters):
        idx = np.where(preds == cid)[0]
        if len(idx) == 0:
            continue

        hits = np.concatenate([event_hits[j] for j in idx], axis=0)

        ax = axes[i]
        ax.scatter(hits[:,0], hits[:,2], s=1)
        ax.set_title(f"C{cid} (n={len(idx)})")
        ax.set_xticks([])
        ax.set_yticks([])

    # quitar subplots vacíos
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clusters_xz.png"))
    plt.close()

# ============================================================
# Pintar perfiles de los clusters
# ============================================================
def plot_shower_profiles(event_hits, preds, output_dir):
    if plt is None:
        return

    for cid in np.unique(preds):
        if cid == -1:
            continue

        idx = np.where(preds == cid)[0]

        longi = []
        lat = []

        for i in idx:
            hits = event_hits[i]
            if len(hits) == 0:
                continue

            longi.append(np.std(hits[:,2]))  # profundidad
            lat.append(np.std(hits[:,0]))    # lateral (puedes añadir Y también)

        if len(longi) == 0:
            continue

        # ---- HISTOGRAMA ----
        fig, axes = plt.subplots(1, 2, figsize=(10,4))

        axes[0].hist(longi, bins=50)
        axes[0].set_title(f"Cluster {cid} - Longitudinal")
        axes[0].set_xlabel("std(Z)")

        axes[1].hist(lat, bins=50)
        axes[1].set_title(f"Cluster {cid} - Lateral")
        axes[1].set_xlabel("std(X)")

        fig.suptitle(f"Cluster {cid} shower profiles (n={len(idx)})")

        path = os.path.join(output_dir, f"cluster_{cid}_profiles.png")
        fig.savefig(path)
        plt.close(fig)

# ============================================================
# Plot del perfil lateral vs longitudinal
# ============================================================
def plot_lateral_vs_longitudinal(features, preds, output_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7,6))

    for cid in np.unique(preds):
        mask = preds == cid
        plt.scatter(features[mask,1], features[mask,2],
                    s=5, alpha=0.5, label=f"cluster {cid}")

    plt.xlabel("Longitudinal")
    plt.ylabel("Lateral")
    plt.legend()
    plt.title("Lateral vs Longitudinal por cluster")

    plt.savefig(os.path.join(output_dir, "lateral_vs_longitudinal.png"))
    plt.close()

# ============================================================
# Hook de clustering / clasificación
# ============================================================

def _compute_pca3d(embeddings):
    """Proyección PCA 3D de los embeddings. Devuelve array (N,3) o None si sklearn no está."""
    if len(embeddings) < 4:
        return None
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return None
    return PCA(n_components=3).fit_transform(embeddings)

# ------- Calcular features------
def compute_event_features(event_hits):
    """
    Calcula features físicas por evento:
    - n_hits
    - longitudinal (std en Z)
    - lateral (spread en X,Y)
    """
    feats = []

    for hits in event_hits:
        if len(hits) == 0:
            feats.append((0, 0, 0))
            continue

        n_hits = len(hits)

        x = hits[:, 0]
        y = hits[:, 1]
        z = hits[:, 2]

        longitudinal = np.std(z)
        lateral = np.sqrt(np.var(x) + np.var(y))

        feats.append((n_hits, longitudinal, lateral))

    return np.array(feats)  # shape (N_events, 3)


def apply_latent_algorithm(embeddings, labels, label_names=None,
                            algorithm=None, mode="clustering",
                            output_dir=None, z2d=None, z3d=None):
    """
    Hook para aplicar algoritmos de sklearn (u otras librerías compatibles)
    sobre el espacio latente del autoencoder.

    Args:
        embeddings  np.ndarray (N, D)  — vectores del espacio latente (event_embedding)
        labels      np.ndarray (N,)    — etiquetas reales de clase (int: 0=electron, 1=pion, 2=muon)
        label_names list[str]          — nombres de las clases (para informes)
        algorithm   objeto sklearn     — ej. KMeans(3), SVC(), RandomForestClassifier()
                    Si None, imprime ayuda y retorna vacío sin hacer nada.
        mode        str                — "clustering" o "classification"
        output_dir  str o None         — si se proporciona, guarda CSV y plots de predicciones
        z2d         np.ndarray (N,2)   — proyección 2D (PCA/t-SNE/UMAP) para scatter plots
        z3d         np.ndarray (N,3)   — proyección 3D para scatter plots

    Returns:
        dict:
            predictions       — array de predicciones, o None si algorithm=None
            metrics           — dict con métricas (ARI/NMI para clustering;
                                accuracy + classification_report para clasificación)
            algorithm_fitted  — el objeto sklearn ya ajustado, o None
            test_indices      — índices del subset de test (solo en clasificación), o None

    Ejemplos de uso:
        # Clustering
        from sklearn.cluster import KMeans
        res = apply_latent_algorithm(embeddings, labels, algorithm=KMeans(n_clusters=3),
                                     mode="clustering", output_dir="eval_output/",
                                     z2d=z2d, z3d=z3d)

        # Clasificación
        from sklearn.svm import SVC
        res = apply_latent_algorithm(embeddings, labels, algorithm=SVC(kernel="rbf"),
                                     mode="classification", output_dir="eval_output/",
                                     z2d=z2d, z3d=z3d)

        # UMAP + HDBSCAN (librerías externas)
        import umap, hdbscan
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb2d   = reducer.fit_transform(embeddings)
        res = apply_latent_algorithm(emb2d, labels, algorithm=hdbscan.HDBSCAN(min_cluster_size=10),
                                     mode="clustering")
    """
    if label_names is None:
        uniq = np.unique(labels)
        label_names = [LABEL_INT_TO_NAME.get(int(li), f"class_{li}") for li in uniq]

    if algorithm is None:
        _log("[apply_latent_algorithm] No se ha proporcionado ningún algoritmo.")
        _log("  Modo clustering:      from sklearn.cluster import KMeans")
        _log("                        apply_latent_algorithm(emb, labels, algorithm=KMeans(3), mode='clustering',")
        _log("                                               output_dir=output_dir, z2d=z2d, z3d=z3d)")
        _log("  Modo clasificación:   from sklearn.svm import SVC")
        _log("                        apply_latent_algorithm(emb, labels, algorithm=SVC(), mode='classification',")
        _log("                                               output_dir=output_dir, z2d=z2d, z3d=z3d)")
        return {"predictions": None, "metrics": {}, "algorithm_fitted": None, "test_indices": None}

    _log(f"[apply_latent_algorithm] mode={mode!r}, algorithm={algorithm.__class__.__name__}")

    test_indices = None

    if mode == "clustering":
        algorithm.fit(embeddings)
        predictions = algorithm.predict(embeddings)
        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            metrics = {
                "ARI": float(adjusted_rand_score(labels, predictions)),
                "NMI": float(normalized_mutual_info_score(labels, predictions)),
            }
        except Exception as e:
            metrics = {"error": str(e)}
        _log(f"  Métricas clustering: {metrics}")

    elif mode == "classification":
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        uniq = np.unique(labels)
        can_stratify = all(np.sum(labels == li) >= 2 for li in uniq)
        all_idx = np.arange(len(embeddings))
        idx_tr, idx_te, X_tr, X_te, y_tr, y_te = train_test_split(
            all_idx, embeddings, labels, test_size=0.3,
            random_state=42, stratify=(labels if can_stratify else None),
        )
        algorithm.fit(X_tr, y_tr)
        predictions = algorithm.predict(X_te)
        test_indices = idx_te
        names_present = [
            LABEL_INT_TO_NAME.get(int(li), f"class_{li}") for li in uniq
        ]
        report = classification_report(y_te, predictions, target_names=names_present)
        metrics = {
            "accuracy": float(accuracy_score(y_te, predictions)),
            "classification_report": report,
        }
        _log(f"  Accuracy: {metrics['accuracy']:.4f}")
        _log(f"  Report:\n{report}")

    else:
        raise ValueError(f"mode debe ser 'clustering' o 'classification', recibido: {mode!r}")

    # ---- Guardar CSV y plots si se proporcionó output_dir ----
    if output_dir is not None:
        algo_name = algorithm.__class__.__name__
        prefix = f"{mode}_{algo_name}"

        # CSV de predicciones
        csv_path = os.path.join(output_dir, f"{prefix}_predictions.csv")
        if mode == "clustering":
            rows = [("event_id", "true_label", "prediction")]
            for i, (tl, pred) in enumerate(zip(labels, predictions)):
                rows.append((i, int(tl), int(pred)))
        else:  # classification: solo test set tiene predicción
            split_arr = np.full(len(labels), "train", dtype=object)
            pred_arr  = np.full(len(labels), -1, dtype=np.int32)
            split_arr[test_indices] = "test"
            pred_arr[test_indices]  = predictions
            rows = [("event_id", "true_label", "prediction", "split")]
            for i, (tl, pred, sp) in enumerate(zip(labels, pred_arr, split_arr)):
                rows.append((i, int(tl), int(pred), sp))
        with open(csv_path, "w") as f:
            for row in rows:
                f.write(",".join(str(x) for x in row) + "\n")
        _log(f"  Predicciones guardadas: {csv_path}")

        # Plots scatter 2D y 3D coloreados por predicción
        if plt is not None:
            if mode == "clustering":
                cluster_ids = np.unique(predictions)
                pred_classes = [(int(ci), f"cluster_{ci}", COLORS[int(ci) % len(COLORS)])
                                for ci in cluster_ids]
                plot_labels = predictions
                plot_z2d = z2d
                plot_z3d = z3d
            else:
                uniq_preds = np.unique(predictions)
                pred_classes = [
                    (int(li), LABEL_INT_TO_NAME.get(int(li), f"class_{li}"),
                     COLORS[int(li) % len(COLORS)])
                    for li in uniq_preds
                ]
                plot_labels = predictions
                plot_z2d = z2d[test_indices] if z2d is not None else None
                plot_z3d = z3d[test_indices] if z3d is not None else None

            if plot_z2d is not None:
                _plot_scatter_2d(
                    plot_z2d, plot_labels, pred_classes,
                    axis_labels=["PC1", "PC2"],
                    title=f"Predicciones {algo_name} ({mode}) — 2D",
                    output_dir=output_dir,
                    filename=f"{prefix}_preds_2d.png",
                )
            if plot_z3d is not None:
                _plot_scatter_3d(
                    plot_z3d, plot_labels, pred_classes,
                    axis_labels=["PC1", "PC2", "PC3"],
                    title=f"Predicciones {algo_name} ({mode}) — 3D",
                    output_dir=output_dir,
                    filename_base=f"{prefix}_preds_3d",
                )

    return {"predictions": predictions, "metrics": metrics,
            "algorithm_fitted": algorithm, "test_indices": test_indices}

# Ponemos un wrapper para que HDBSCAN imite .predict()

class HDBSCANWrapper:
    def __init__(self, **kwargs):
        import hdbscan
        self.model = hdbscan.HDBSCAN(**kwargs)

    def fit(self,X):
        self.labels_ = self.model.fit_predict(X)
        return self

    def predict(self,X):
        return self.labels_


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device is not None:
        device = torch.device(args.device)
    _log(f"Dispositivo: {device}")

    cfg_models = yaml.safe_load(open(args.cfg, "r"))

    # Modelo
    model = load_model(args.ckpt, cfg_models, device)
    _log(f"Modelo: {sum(p.numel() for p in model.parameters()):,} parámetros")

    norm_type = args.norm or ("z_norm" if args.z_norm else None)
    preprocessing_cfg = {
        "use_scalar":     args.use_scalar,
        "use_one_hot":    args.use_one_hot,
        "norm_type":      norm_type,
        "norm_yaml_path": args.norm_yaml,
        "z_norm":         args.z_norm,
    }

    # Datasets por clase de partícula (solo los que se hayan proporcionado)
    particle_specs = [
        (args.electron_path, 0, "electron"),
        (args.pion_path,     1, "pion"),
        (args.muon_path,     2, "muon"),
    ]
    loaders_with_labels = []
    for path, li, lname in particle_specs:
        if path is None:
            continue
        entry = load_labeled_dataset(
            path, li, lname, preprocessing_cfg, norm_type,
            args.batch_size, args.max_events,
        )
        loaders_with_labels.append(entry)

    if not loaders_with_labels:
        _log("ERROR: no se ha especificado ningún fichero de datos. "
             "Usa --electron_path, --pion_path y/o --muon_path.")
        return

    # Inferencia
    _log("Corriendo inferencia...")
    results = run_inference(
        model, loaders_with_labels, device, args.use_scalar, args.use_one_hot
    )
    embeddings = results["embeddings"]
    labels     = results["labels"]
    energies   = results["energies"]
    n_ev = len(embeddings)
    n_hits = len(results["xyz_true"])
    _log(f"Inferencia completada: {n_ev} eventos, {n_hits} hits")

    # Ranking por MSE por evento
    per_event_mse = compute_per_event_mse(results)

    # ---- Plots de reconstrucción ----
    _log("Generando plots de errores de reconstrucción...")
    metrics_summary = plot_reconstruction_errors(results, args.use_scalar, args.output_dir)

    _log("Generando plots top-5 best/worst...")
    plot_top_k_events(results, per_event_mse, k=5, output_dir=args.output_dir)

    # ---- Plots del espacio latente ----
    _log("Generando PCA 2D...")
    pca_result = plot_pca_2d(embeddings, labels, loaders_with_labels, args.output_dir,
                              energies=energies)

    _log("Generando PCA 3D...")
    plot_pca_3d(embeddings, labels, loaders_with_labels, args.output_dir,
                energies=energies)

    z2d = None
    if pca_result is not None:
        z2d, _ = pca_result
        _log(f"Generando PCA grid ({args.n_grid}×{args.n_grid})...")
        plot_pca_grid(
            z2d, results["event_hits"], labels,
            loaders_with_labels, args.n_grid, args.output_dir,
        )

    _log("Generando t-SNE 2D, 3D y grid XZ...")
    plot_tsne(
        embeddings, labels, loaders_with_labels, args.output_dir,
        event_hits=results["event_hits"],
        n_grid=args.n_grid,
        max_events=args.max_reduction_events,
        perplexity=args.tsne_perplexity,
        energies=energies,
    )

    _log("Generando UMAP 2D, 3D y grid XZ (si umap-learn está disponible)...")
    plot_umap(
        embeddings, labels, loaders_with_labels, args.output_dir,
        event_hits=results["event_hits"],
        n_grid=args.n_grid,
        max_events=args.max_reduction_events,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        energies=energies,
    )

    # ---- Hook de clustering / clasificación ----
    # Pasa output_dir, z2d y z3d para que, cuando se llame con un algoritmo,
    # se guarden automáticamente el CSV de predicciones y los plots 2D/3D.
    features = compute_event_features(results["event_hits"])

    z3d = _compute_pca3d(embeddings)

    hdb = HDBSCANWrapper(min_cluster_size=20)

    res = apply_latent_algorithm(embeddings, labels,algorithm=hdb,mode="clustering",
                           output_dir=args.output_dir, z2d=z2d, z3d=z3d)

    preds= res["predictions"]

    #------- Análisis de clusters-------
    print("Clusters encontrados:", np.unique(preds))

    for cid in np.unique(preds):
        n = np.sum(preds == cid)
        print(f"Cluster {cid}: {n} eventos")

    plot_hits_by_cluster(results["event_hits"],preds, args.output_dir)
    plot_shower_profiles(results["event_hits"], preds, args.output_dir)
    plot_lateral_vs_longitudinal(features, preds, args.output_dir)
    #------ Análisis de hits-------
    for cid in np.unique(preds):
        if cid == -1:
            continue

    idx= np.where(preds == cid)[0]
    sizes = [len(results["event_hits"][i]) for i in idx]

    print(f"Cluster {cid}: mean hits = {np.mean(sizes):.1f}, std= {np.std(sizes):.1f}")

    #------- Forma del shower-------
    for cid in np.unique(preds):
        if cid == -1:
            continue

    idx = np.where(preds == cid)[0]

    longi = []
    lat = []

    for i in idx:
        hits = results["event_hits"][i]
        if len(hits) == 0:
            continue

        longi.append(np.std(hits[:,2]))
        lat.append(np.std(hits[:,0]))

    print(f"Cluster {cid}: longitudinal={np.mean(longi):.2f}, lateral={np.mean(lat):.2f}")

    #---------Eventos raros-------
    threshold = np.percentile(per_event_mse, 95)
    high_error_idx = np.where(per_event_mse > threshold)[0]

    print("Eventos raros:", len(high_error_idx))
    print("Clusters de eventos raros:", preds[high_error_idx])

    # ---- Resumen de métricas ----
    summary_path = os.path.join(args.output_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Resumen de evaluación ===\n\n")
        f.write(f"Checkpoint : {args.ckpt}\n")
        f.write(f"Config     : {args.cfg}\n\n")
        f.write(f"N eventos  : {n_ev}\n")
        for _, li, lname in loaders_with_labels:
            cnt = int(np.sum(labels == li))
            f.write(f"  {lname:10s}: {cnt} eventos\n")
        f.write("\nMétricas de reconstrucción (MSE):\n")
        for k, v in (metrics_summary or {}).items():
            if v is not None:
                f.write(f"  {k}: {v:.6f}\n")
        f.write(
            f"\nPer-event MSE (XYZ): "
            f"mean={per_event_mse.mean():.6f}, "
            f"median={np.median(per_event_mse):.6f}, "
            f"p95={np.percentile(per_event_mse, 95):.6f}\n"
        )
    _log(f"Resumen guardado en: {summary_path}")
    _log("Evaluación completada.")


if __name__ == "__main__":
    main()
