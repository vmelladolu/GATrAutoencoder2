"""Clasificacion de particulas (e/pi/mu) sobre el latente de un AE GATr congelado.

Modos de uso:

  1. Evaluacion sim (held-out):
       Compara cabeza NN y SVC sobre *_test.h5 para ambos latentes.

  2. Inferencia test-beam (--testbeam_path):
       Sin ground truth. Produce histogramas de nHits y 2D lat vs long por clase predicha.
       Usa 'k' (layer index) como coordenada z — no 'z' en mm.
       Soporta filtrado por energia de haz (--tb_energies) y n_hits minimo (--min_hits_tb).

Ejemplo modo sim:
  python src/evaluate_classifier.py \\
    --ae_ckpt checkpoints_vae_sim/checkpoint_best_epoch0402_loss0.078888.pt \\
    --cfg config/model_cfg_clf_head.yml \\
    --electron_train .../e-_train.h5 --electron_test .../e-_test.h5 \\
    --pion_train     .../pi-_train.h5 --pion_test     .../pi-_test.h5 \\
    --muon_train     .../mu-_train.h5 --muon_test     .../mu-_test.h5 \\
    --use_scalar --use_one_hot --z_norm \\
    --head_aggregate checkpoints_clf_head/clf_head_aggregate_best.pt \\
    --head_embedding checkpoints_clf_head/clf_head_embedding_best.pt \\
    -o eval_clf_output --device cuda:0

Ejemplo modo test-beam:
  python src/evaluate_classifier.py \\
    --ae_ckpt checkpoints_vae_sim/checkpoint_best_epoch0402_loss0.078888.pt \\
    --cfg config/model_cfg_clf_head.yml \\
    --electron_train .../e-_train.h5 --pion_train .../pi-_train.h5 --muon_train .../mu-_train.h5 \\
    --use_scalar --use_one_hot --z_norm \\
    --head_aggregate checkpoints_clf_head/clf_head_aggregate_best.pt \\
    --testbeam_path /pnfs/.../data_jorge_flat.h5 \\
    --tb_energies 60 80 100 --min_hits_tb 20 --max_events_tb 50000 \\
    -o eval_clf_output --device cuda:0
"""

import os
import sys
import time
import argparse

import h5py
import numpy as np
import yaml

# Fallback para entornos sin torch (modo análisis --load_predictions)
LABEL_INT_TO_NAME = {0: "electron", 1: "pion", 2: "muon"}

try:
    import torch
    from torch_geometric.loader import DataLoader
    from models.gatr_autoencoder import GATrAutoencoder
    from models.classification_head import ClassificationHead
    from utils.batch_utils import build_batch
    from utils.clf_data import (
        LABEL_INT_TO_NAME,        # sobreescribe el fallback de arriba
        compute_or_load_combined_stats,
        build_labeled_datasets,
    )
   # from train_classifier_head import load_frozen_ae, LATENT_KEY
    _TORCH_AVAILABLE = True
except ImportError as e:
    print("DEBUG IMPORT ERROR:", repr(e))
    torch = None
    _TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # no-op si tqdm no está disponible
        return it

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax estable en numpy puro — no requiere torch ni scipy."""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _no_grad():
    """Devuelve torch.no_grad() si torch está disponible, si no un no-op."""
    if torch is not None:
        return torch.no_grad()
    from contextlib import nullcontext
    return nullcontext()


# ── lat/long import (filtering/filters.py, una carpeta arriba del repo) ──────

def _try_import_lat_long():
    """Intenta importar compute_hits_per_plane y lateral_longitudinal."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "filtering"),
        "/nfs/cms/arqolmo/SDHCAL_Energy/filtering",
    ]
    for d in candidates:
        d = os.path.abspath(d)
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        from filters import compute_hits_per_plane, lateral_longitudinal
        return compute_hits_per_plane, lateral_longitudinal
    except ImportError:
        _log("WARNING: filtering/filters.py no encontrado — plots lat/long deshabilitados")
        return None, None


# ============================================================
# Modo 1: evaluacion sobre sim (held-out)
# ============================================================

def extract_latents(ae, loader, device, use_scalar, use_one_hot):
    """Devuelve dict {aggregate, embedding, labels} concatenados."""
    agg, emb, lab = [], [], []
    with _no_grad():
        for batch in tqdm(loader):
            data = build_batch(batch, use_scalar=use_scalar, use_one_hot=use_one_hot)
            mv_v = data["mv_v_part"].to(device)
            mv_s = data["mv_s_part"].to(device)
            sc   = data["scalars"].to(device)
            bidx = data["batch_idx"].to(device)
            out  = ae(mv_v, mv_s, sc, bidx)
            agg.append(out["aggregate_latent"].cpu().numpy())
            emb.append(out["event_embedding"].cpu().numpy())
            lab.append(batch.y.cpu().numpy().reshape(-1))
    return {
        "aggregate": np.concatenate(agg),
        "embedding": np.concatenate(emb),
        "labels":    np.concatenate(lab),
    }


def _loader_from_specs(specs, stats, norm_type, preprocessing_cfg, batch_size,
                       max_events=0):
    concat, per_class = build_labeled_datasets(
        specs, stats, norm_type, preprocessing_cfg, max_events=max_events)
    return DataLoader(concat, batch_size=batch_size, shuffle=False), per_class


# ─── metricas y plots sim ─────────────────────────────────────────────────────

def evaluate_predictions(y_true, y_pred, n_classes):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    names = [LABEL_INT_TO_NAME.get(i, f"class_{i}") for i in range(n_classes)]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(n_classes))),
        "names": names,
    }


def plot_confusion(cm, names, title, path):
    if plt is None:
        return
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]*100:.0f}%",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    _log(f"Guardado: {path}")


# ─── NN head y SVC ────────────────────────────────────────────────────────────

def run_nn_head(head_ckpt, lat_train, lat_test, y_test, n_classes, device):
    raw = torch.load(head_ckpt, map_location=device)
    head = ClassificationHead(
        in_dim=raw["in_dim"], n_classes=raw["n_classes"],
        hidden_dim=raw.get("hidden_dim", 128),
        dropout=raw.get("dropout", 0.2),
        num_layers=raw.get("num_layers", 1)).to(device)
    head.load_state_dict(raw["head_state_dict"])
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(lat_test).float().to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()
    return evaluate_predictions(y_test, y_pred, n_classes)


def run_svc(lat_train, y_train, lat_test, y_test, n_classes,
            C, gamma, max_train, seed=42):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    if max_train > 0 and len(lat_train) > max_train:
        rng = np.random.default_rng(seed)
        sel = []
        per = max(1, max_train // n_classes)
        for c in range(n_classes):
            idx = np.where(y_train == c)[0]
            sel.append(rng.choice(idx, size=min(per, len(idx)), replace=False))
        sel = np.concatenate(sel)
        lat_train, y_train = lat_train[sel], y_train[sel]

    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma))
    clf.fit(lat_train, y_train)
    y_pred = clf.predict(lat_test)
    res = evaluate_predictions(y_test, y_pred, n_classes)
    res["n_train"] = len(y_train)
    return res


# ============================================================
# Modo 2: inferencia test-beam
# ============================================================

def _norm_const(sim_stats, key):
    s = sim_stats.get(key, {})
    return float(s.get("mean", 0.0)), float(s.get("std", 1.0))


def _load_head(head_ckpt, device, n_classes_default=3):
    """Carga una ClassificationHead desde checkpoint.

    Soporta tanto los checkpoints de cabeza (clf_head_*_best.pt) como los
    bundles de fine-tuning (finetune_*.pt), que comparten las claves
    ``head_state_dict``/``in_dim``/``hidden_dim``/... pero el bundle NO guarda
    ``n_classes`` (se infiere de la última Linear o del default).
    """
    if head_ckpt is None or not os.path.exists(head_ckpt):
        return None
    raw = torch.load(head_ckpt, map_location=device)
    sd = raw["head_state_dict"]
    # n_classes: del ckpt si está; si no, de la forma de la última capa lineal
    n_classes = raw.get("n_classes")
    if n_classes is None:
        last_w = [v for k, v in sd.items() if k.endswith(".weight")][-1]
        n_classes = int(last_w.shape[0]) if last_w is not None else n_classes_default
    head = ClassificationHead(
        in_dim=raw["in_dim"], n_classes=n_classes,
        hidden_dim=raw.get("hidden_dim", 128),
        dropout=raw.get("dropout", 0.2),
        num_layers=raw.get("num_layers", 1)).to(device)
    head.load_state_dict(sd)
    head.eval()
    return head


def load_finetuned_ae(ckpt_path, cfg_models, device):
    """Carga el AE *fine-tuned* desde un bundle de fine-tuning.

    El bundle (``finetune_*.pt``) guarda los pesos del AE en ``ae_state_dict``.
    Como el fine-tuning modificó el encoder/aggregation, la evaluación DEBE usar
    estos pesos (no el AE original de checkpoints_vae_sim) para que la cabeza
    reciba el mismo espacio latente con el que se entrenó.
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
        cfg_enc=cfg_enc, cfg_dec=cfg_dec, cfg_agg=cfg_agg,
        latent_s_channels=2, use_vae=use_vae, event_embed_dim=event_embed_dim,
    )
    raw = torch.load(ckpt_path, map_location=device)
    if "ae_state_dict" not in raw:
        raise KeyError(
            f"{ckpt_path} no contiene 'ae_state_dict' — ¿es un bundle de "
            f"fine-tuning? Para un AE normal usa --ae_ckpt en vez de --finetune_ckpt.")
    model.load_state_dict(raw["ae_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _log(f"AE fine-tuned cargado (congelado): {ckpt_path} "
         f"[epoch={raw.get('epoch','?')}, val_acc={raw.get('val_acc','?')}]")
    return model


def _filter_testbeam_events(h5_path, tb_energies, min_hits, max_events, seed=42):
    """
    Lee offsets y energy del HDF5, aplica filtros y devuelve event_indices.
    Tambien devuelve offsets completo y energy array para uso posterior.
    """
    _log(f"Leyendo metadatos de {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        offsets = f["offsets"][:].astype(np.int64)   # (N+1,)
        energy  = f["energy"][:].astype(np.float32)  # (N,)

    nhits = np.diff(offsets)
    n_total = len(energy)

    mask = np.ones(n_total, dtype=bool)

    if tb_energies and len(tb_energies) > 0:
        e_mask = np.zeros(n_total, dtype=bool)
        for e_val in tb_energies:
            e_mask |= np.abs(energy - float(e_val)) < 0.5
        mask &= e_mask
        _log(f"  Filtro energia {tb_energies} GeV: {e_mask.sum():,} eventos")

    if min_hits > 0:
        mask &= nhits >= min_hits
        _log(f"  Filtro n_hits >= {min_hits}: {mask.sum():,} eventos")

    event_indices = np.flatnonzero(mask)

    if max_events > 0 and len(event_indices) > max_events:
        rng = np.random.default_rng(seed)
        event_indices = rng.choice(event_indices, size=max_events, replace=False)
        event_indices = np.sort(event_indices)
        _log(f"  Subsampleo a {max_events:,} eventos")

    _log(f"  Total eventos seleccionados: {len(event_indices):,} / {n_total:,}")
    return event_indices, offsets, energy


def _testbeam_inference(ae, head_agg, head_emb,
                        h5_path, event_indices, offsets, energy_arr,
                        sim_stats, use_scalar, device, batch_size,
                        compute_hits_per_plane_fn, lateral_longitudinal_fn):
    """
    Pasa el test-beam por el encoder y las cabezas clasificadoras.
    Usa 'k' (layer index) como coordenada z (no 'z' en mm).
    Calcula lateral/longitudinal por evento usando filtering/filters.py.

    Devuelve dict con arrays de longitud len(event_indices):
      y_pred_agg, y_pred_emb  — predicciones argmax
      logits_agg, logits_emb  — logits crudos [N, n_classes]
      nhits, lat, long, energy
    """
    has_lat_long = (compute_hits_per_plane_fn is not None
                    and lateral_longitudinal_fn is not None)

    n_ev = len(event_indices)
    y_pred_agg  = np.full(n_ev, -1, dtype=np.int32)
    y_pred_emb  = np.full(n_ev, -1, dtype=np.int32)
    logits_agg  = np.full((n_ev, 3), np.nan, dtype=np.float32)
    logits_emb  = np.full((n_ev, 3), np.nan, dtype=np.float32)
    nhits_arr   = np.zeros(n_ev, dtype=np.int32)
    lat_arr     = np.full(n_ev, np.nan, dtype=np.float32)
    long_arr    = np.full(n_ev, np.nan, dtype=np.float32)
    energy_out  = np.zeros(n_ev, dtype=np.float32)

    # Normalizacion (usando z-stats para k como coordenada z)
    x_mean, x_std = _norm_const(sim_stats, "x")
    y_mean, y_std = _norm_const(sim_stats, "y")
    z_mean, z_std = _norm_const(sim_stats, "z")   # aplicado a k como z
    k_mean, k_std = _norm_const(sim_stats, "k")   # mv_s escalar

    ae.eval()
    if head_agg is not None: head_agg.eval()
    if head_emb is not None: head_emb.eval()

    with _no_grad(), h5py.File(h5_path, "r") as hf:
        h5_x   = hf["x"]
        h5_y   = hf["y"]
        h5_k   = hf["k"]
        h5_thr = hf["thr"]
        h5_i   = hf["i"] if has_lat_long else None
        h5_j   = hf["j"] if has_lat_long else None

        for b_start in tqdm(range(0, n_ev, batch_size), desc="TB inference"):
            batch_ev = event_indices[b_start: b_start + batch_size]
            B = len(batch_ev)

            pos_list, k_list, sc_list, bidx_list = [], [], [], []

            for local_i, ev in enumerate(batch_ev):
                s = int(offsets[ev])
                e = int(offsets[ev + 1])
                n = e - s
                global_i = b_start + local_i

                nhits_arr[global_i]  = n
                energy_out[global_i] = float(energy_arr[ev])

                x_raw   = np.asarray(h5_x[s:e],   dtype=np.float32)
                y_raw   = np.asarray(h5_y[s:e],   dtype=np.float32)
                k_raw   = np.asarray(h5_k[s:e],   dtype=np.float32)
                thr_raw = np.asarray(h5_thr[s:e], dtype=np.float32)

                # ── lat/long (CPU, usa i,j,k sin normalizar) ──
                if has_lat_long:
                    i_raw = np.asarray(h5_i[s:e], dtype=np.float32)
                    j_raw = np.asarray(h5_j[s:e], dtype=np.float32)
                    # k+1 para que sea 1-indexed; usar int32 para evitar
                    # float-indexing en compute_hits_per_plane (numpy >= 1.24)
                    k_int = np.asarray(h5_k[s:e], dtype=np.int32)
                    sig_pos = np.stack([i_raw, j_raw, (k_int + 1).astype(np.float32)],
                                       axis=1)
                    try:
                        nhitsRPC, meanI, meanJ = compute_hits_per_plane_fn(sig_pos)
                        ll = lateral_longitudinal_fn(
                            sig_pos, nhitsRPC, meanI, meanJ,
                            float(energy_arr[ev]))
                        lat_arr[global_i]  = ll["lateral"]
                        long_arr[global_i] = ll["longitudinal"]
                    except Exception as _ex:
                        if global_i < 3:
                            _log(f"  WARNING lat/long ev={ev}: {type(_ex).__name__}: {_ex}")

                # ── normalizacion para el encoder ──
                x_n     = (x_raw - x_mean) / x_std
                y_n     = (y_raw - y_mean) / y_std
                k_as_z  = (k_raw - z_mean) / z_std   # k como coordenada z
                k_n     = (k_raw - k_mean) / k_std   # k como escalar mv_s

                pos_list.append(np.stack([x_n, y_n, k_as_z], axis=1))
                k_list.append(k_n[:, None])

                # Test-beam vs sim: las etiquetas de threshold 1 y 2 estan
                # INTERCAMBIADAS en data_jorge_flat.h5 respecto a la sim.
                # Comprobado: TB crudo tiene thr2 (0.793) > thr1 (0.166), lo cual
                # es fisicamente imposible (thr es el nivel mas alto superado, asi
                # que thr1>=thr2>=thr3). Al intercambiar 1<->2, el TB coincide con
                # la sim (thr1~0.79, thr2~0.17, thr3~0.04). thr3 NO se intercambia.
                t1 = (thr_raw == 2).astype(np.float32)[:, None]  # TB '2' = thr1 fisico
                t2 = (thr_raw == 1).astype(np.float32)[:, None]  # TB '1' = thr2 fisico
                t3 = (thr_raw == 3).astype(np.float32)[:, None]
                sc_list.append(np.concatenate([t1, t2, t3], axis=1))
                bidx_list.append(np.full(n, local_i, dtype=np.int64))

            # ── forward GPU ──
            mv_v = torch.from_numpy(np.concatenate(pos_list)).to(device)
            mv_s = (torch.from_numpy(np.concatenate(k_list)).to(device)
                    if use_scalar
                    else torch.zeros(mv_v.shape[0], 1, device=device))
            sc_t = torch.from_numpy(np.concatenate(sc_list)).to(device)
            bidx = torch.from_numpy(np.concatenate(bidx_list)).to(device)

            out = ae(mv_v, mv_s, sc_t, bidx)

            if head_agg is not None:
                lg = head_agg(out["aggregate_latent"]).cpu().numpy()
                logits_agg[b_start: b_start + B] = lg.astype(np.float32)
                y_pred_agg[b_start: b_start + B] = lg.argmax(1).astype(np.int32)

            if head_emb is not None:
                lg = head_emb(out["event_embedding"]).cpu().numpy()
                logits_emb[b_start: b_start + B] = lg.astype(np.float32)
                y_pred_emb[b_start: b_start + B] = lg.argmax(1).astype(np.int32)

    n_valid_ll = int(np.isfinite(lat_arr).sum())
    _log(f"Lat/long OK: {n_valid_ll}/{n_ev} eventos ({100*n_valid_ll/max(n_ev,1):.1f}%)")

    return {
        "y_pred_agg":  y_pred_agg,
        "y_pred_emb":  y_pred_emb,
        "logits_agg":  logits_agg,
        "logits_emb":  logits_emb,
        "nhits":       nhits_arr,
        "lat":         lat_arr,
        "long":        long_arr,
        "energy":      energy_out,
    }


# ─── plots test-beam ──────────────────────────────────────────────────────────

def _energy_label(tb_energies):
    if not tb_energies:
        return "todas las energias"
    return f"E = {', '.join(str(int(e)) for e in sorted(tb_energies))} GeV"


def plot_nhits_by_class(nhits, y_pred, out_dir, tb_energies=None, n_classes=3,
                        prefix="tb"):
    if plt is None:
        return
    class_names = [LABEL_INT_TO_NAME.get(i, f"class_{i}") for i in range(n_classes)]
    colors      = ["steelblue", "tab:orange", "tab:green"]
    elabel      = _energy_label(tb_energies)

    hi   = int(np.percentile(nhits, 99)) + 1
    bins = np.linspace(0, hi, 60)

    data_per_cls   = [nhits[y_pred == cls] for cls in range(n_classes)]
    labels_per_cls = [f"{class_names[cls]} (n={len(data_per_cls[cls]):,})"
                      for cls in range(n_classes)]
    valid = [(data_per_cls[c], labels_per_cls[c], colors[c])
             for c in range(n_classes) if len(data_per_cls[c]) > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, log_y in zip(axes, [False, True]):
        if valid:
            ax.hist(
                [d for d, _, _ in valid],
                bins=bins,
                histtype="barstacked",
                color=[c for _, _, c in valid],
                label=[l for _, l, _ in valid],
                alpha=1.0,            # opaco: apilado limpio, sin aspecto solapado
                edgecolor="white",    # separa visualmente los segmentos apilados
                linewidth=0.3,
            )
        ax.set_xlabel("n_hits / evento")
        ax.set_ylabel("eventos")
        ax.set_title(f"n_hits por clase predicha — {elabel}" + (" [log Y]" if log_y else ""))
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_nhits_by_class.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    _log(f"Guardado: {path}")


def plot_lat_long_2d(lat, long_, y_pred, out_dir, tb_energies=None, n_classes=3,
                     prefix="tb"):
    if plt is None:
        return
    class_names = [LABEL_INT_TO_NAME.get(i, f"class_{i}") for i in range(n_classes)]
    cmaps       = ["Blues", "Oranges", "Greens"]
    line_colors = ["steelblue", "tab:orange", "tab:green"]
    elabel      = _energy_label(tb_energies)
    bins        = np.linspace(0, 1, 50)

    valid = np.isfinite(lat) & np.isfinite(long_)

    # ── Subplots individuales por clase ───────────────────────────────────────
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4.5),
                             sharex=True, sharey=True)
    for cls, ax in enumerate(axes):
        m = (y_pred == cls) & valid
        ax.set_title(f"{class_names[cls]}  n={m.sum():,} — {elabel}", fontsize=9)
        if m.sum() >= 2:
            counts, xe, ye = np.histogram2d(lat[m], long_[m], bins=bins)
            mesh = ax.pcolormesh(xe, ye, counts.T, cmap=cmaps[cls])
            fig.colorbar(mesh, ax=ax, fraction=0.046, label="eventos")
        ax.set_xlabel("lateral (Rate3)")
        ax.set_ylabel("longitudinal (Rate4)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_lat_long_2d.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    _log(f"Guardado: {path}")

    # ── Overlay con contornos ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    bins_c = np.linspace(0, 1, 30)
    for cls in range(n_classes):
        m = (y_pred == cls) & valid
        if m.sum() < 5:
            continue
        counts, xe, ye = np.histogram2d(lat[m], long_[m], bins=bins_c, density=True)
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        ax.contour(xc, yc, counts.T, levels=5, colors=[line_colors[cls]],
                   alpha=0.85, linewidths=1.3)
        ax.plot([], [], color=line_colors[cls], lw=1.3, label=class_names[cls])
    ax.set_xlabel("lateral (Rate3)"); ax.set_ylabel("longitudinal (Rate4)")
    ax.set_title(f"lat vs long — contornos por clase — {elabel}", fontsize=9)
    ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_lat_long_overlay.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    _log(f"Guardado: {path}")


# Rangos de nHits para los grids 3D (un fichero por rango). hi=None => abierto.
GRID_HITS_RANGES = [(0, 200), (200, 600), (600, 1000), (1000, None)]


def plot_3d_grid(h5_path, event_indices, offsets, nhits, y_pred,
                 out_dir, tb_energies=None, n_classes=3, n_cols=6, prefix="tb",
                 hits_range=None):
    """Grid 3D de hits: filas=clase predicha, columnas=bins cuantílicos de nHits.

    Cada celda muestra 1 evento representativo (el más cercano a la mediana de
    nHits del bin dentro de esa clase). Los bins son cuantílicos sobre todos los
    eventos seleccionados para garantizar representación de eventos escasos/densos.

    ``hits_range``: tupla ``(lo, hi)`` que restringe el pool a ``lo <= nHits < hi``
    (con ``hi=None`` para rango abierto por arriba). El rango se refleja en el
    nombre del fichero y en el título. ``None`` usa todos los eventos.
    """
    if plt is None:
        return

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: registra el projection
    except ImportError:
        _log("WARNING: mpl_toolkits.mplot3d no disponible — plot 3D omitido")
        return

    from matplotlib.lines import Line2D

    class_names = [LABEL_INT_TO_NAME.get(i, f"class_{i}") for i in range(n_classes)]
    elabel      = _energy_label(tb_energies)
    thr_colors  = {1: "steelblue", 2: "tab:orange", 3: "tab:red"}
    thr_sizes   = {1: 8, 2: 14, 3: 22}

    # Pool de eventos a representar (filtro opcional por rango de nHits)
    if hits_range is not None:
        lo_r, hi_r = hits_range
        pool_mask = nhits >= lo_r
        if hi_r is not None:
            pool_mask &= nhits < hi_r
        range_lbl  = f"{int(lo_r)}-{int(hi_r)}" if hi_r is not None else f"{int(lo_r)}plus"
        range_human = (f"{int(lo_r)}–{int(hi_r)} hits" if hi_r is not None
                       else f"≥{int(lo_r)} hits")
        fname_suffix = f"_h{range_lbl}"
    else:
        pool_mask    = np.ones_like(nhits, dtype=bool)
        range_human  = "todos los n_hits"
        fname_suffix = ""

    if pool_mask.sum() == 0:
        _log(f"WARNING: ningun evento en rango {range_human} — grid 3D omitido")
        return
    if hits_range is not None:
        _log(f"  grid 3D [{range_human}]: {pool_mask.sum():,} eventos")

    # Bins cuantílicos sobre el pool filtrado
    q          = np.linspace(0, 100, n_cols + 1)
    bin_edges  = np.percentile(nhits[pool_mask], q).astype(float)
    bin_edges[-1] += 1.0          # borde derecho inclusivo
    col_titles = [f"{int(bin_edges[j])}–{int(bin_edges[j+1]-1)} hits"
                  for j in range(n_cols)]

    # Seleccionar 1 evento por celda (clase × bin)
    cell_ev = {}
    for cls in range(n_classes):
        for col in range(n_cols):
            lo, hi = bin_edges[col], bin_edges[col + 1]
            mask = pool_mask & (y_pred == cls) & (nhits >= lo) & (nhits < hi)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                cell_ev[(cls, col)] = None
                continue
            med  = np.median(nhits[idxs])
            best = idxs[np.argmin(np.abs(nhits[idxs] - med))]
            cell_ev[(cls, col)] = int(best)

    fig = plt.figure(figsize=(4.0 * n_cols, 4.5 * n_classes))

    with h5py.File(h5_path, "r") as hf:
        for row in range(n_classes):
            for col in range(n_cols):
                ax = fig.add_subplot(n_classes, n_cols,
                                     row * n_cols + col + 1,
                                     projection="3d")
                try:
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False
                except AttributeError:
                    pass

                # ── título de celda ──
                title_parts = []
                if row == 0:
                    title_parts.append(col_titles[col])
                if col == 0:
                    title_parts.append(f"[{class_names[row]}]")
                ax.set_title("\n".join(title_parts), fontsize=7, pad=2)

                local_idx = cell_ev.get((row, col))
                if local_idx is None:
                    ax.text2D(0.5, 0.5, "sin\neventos",
                              ha="center", va="center", transform=ax.transAxes,
                              fontsize=8, color="gray")
                    continue

                ev_global = int(event_indices[local_idx])
                s = int(offsets[ev_global])
                e = int(offsets[ev_global + 1])

                x_raw   = np.asarray(hf["x"][s:e],   dtype=np.float32)
                y_raw   = np.asarray(hf["y"][s:e],   dtype=np.float32)
                k_raw   = np.asarray(hf["k"][s:e],   dtype=np.int32)
                thr_raw = np.asarray(hf["thr"][s:e], dtype=np.int32)

                for tval in [1, 2, 3]:
                    m = thr_raw == tval
                    if m.sum() == 0:
                        continue
                    ax.scatter(x_raw[m], y_raw[m], k_raw[m],
                               c=thr_colors[tval], s=thr_sizes[tval],
                               alpha=0.65, depthshade=False)

                n_ev = e - s
                ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_zlim(0, 47)
                ax.set_xlabel("x", fontsize=5, labelpad=-4)
                ax.set_ylabel("y", fontsize=5, labelpad=-4)
                ax.set_zlabel("k", fontsize=5, labelpad=-4)
                ax.tick_params(labelsize=4, pad=-3)
                ax.view_init(elev=25, azim=45)
                # número de hits del evento específico en el corner del título
                ax.set_title(
                    "\n".join(title_parts + [f"n={n_ev}"]),
                    fontsize=7, pad=2,
                )

    legend_els = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=thr_colors[t], markersize=6,
               label=f"thr={t}") for t in [1, 2, 3]
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"Muestra 3D — {elabel} — [{range_human}]\n"
        f"Filas: clase predicha  |  Columnas: bins cuantílicos de n_hits",
        fontsize=10, y=1.01,
    )
    fig.tight_layout(pad=0.4)
    path = os.path.join(out_dir, f"{prefix}_3d_grid{fname_suffix}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    _log(f"Guardado: {path}")


# ─── guardado / carga de predicciones crudas ─────────────────────────────────

def save_predictions(results, event_indices, offsets, h5_path,
                     tb_energies, min_hits, head_names, path):
    """Guarda predicciones crudas en un archivo NPZ para análisis posterior."""
    np.savez_compressed(
        path,
        # predicciones y logits
        y_pred_agg  = results["y_pred_agg"],
        y_pred_emb  = results["y_pred_emb"],
        logits_agg  = results["logits_agg"],
        logits_emb  = results["logits_emb"],
        # metadata por evento
        nhits       = results["nhits"],
        lat         = results["lat"],
        long        = results["long"],
        energy      = results["energy"],
        # índices para acceso posterior al HDF5
        event_indices = event_indices,
        offsets       = offsets,
        # metadatos del filtrado (guardados como arrays 0-d o 1-d)
        tb_energies   = np.array(tb_energies if tb_energies else [], dtype=np.float32),
        min_hits      = np.int32(min_hits),
        h5_path       = np.array(h5_path),
        head_names    = np.array(sorted(head_names)),
    )
    _log(f"Predicciones crudas guardadas en: {path}")


def load_predictions(path):
    """Carga predicciones guardadas por save_predictions.

    Returns:
        results      — dict compatible con el formato de _testbeam_inference
        event_indices, offsets, h5_path, tb_energies, head_names
    """
    d = np.load(path, allow_pickle=True)
    results = {
        "y_pred_agg": d["y_pred_agg"],
        "y_pred_emb": d["y_pred_emb"],
        "logits_agg": d["logits_agg"],
        "logits_emb": d["logits_emb"],
        "nhits":      d["nhits"],
        "lat":        d["lat"],
        "long":       d["long"],
        "energy":     d["energy"],
    }
    event_indices = d["event_indices"]
    offsets       = d["offsets"]
    h5_path       = str(d["h5_path"])
    tb_energies   = list(d["tb_energies"].tolist())
    min_hits      = int(d["min_hits"])
    head_names    = set(d["head_names"].tolist())
    _log(f"Predicciones cargadas desde: {path} ({len(event_indices):,} eventos)")
    return results, event_indices, offsets, h5_path, tb_energies, min_hits, head_names


def apply_confidence_filter(results, min_confidence, head_names):
    """Reemplaza predicciones de baja confianza por -1 (clase 'rechazado').

    La confianza se define como max(softmax(logits)).
    Devuelve copia de results con y_pred_* modificados y arrays 'conf_*'.
    """
    out = dict(results)
    for src in ("agg", "emb"):
        full_src = "aggregate" if src == "agg" else "embedding"
        if full_src not in head_names:
            out[f"conf_{src}"] = np.full(len(results["nhits"]), np.nan, dtype=np.float32)
            continue
        logits = results[f"logits_{src}"]
        valid  = np.isfinite(logits).all(axis=1)
        probs  = np.zeros_like(logits)
        if valid.any():
            probs[valid] = _softmax(logits[valid])
        conf = probs.max(axis=1).astype(np.float32)
        out[f"conf_{src}"] = conf
        if min_confidence > 0:
            rejected = valid & (conf < min_confidence)
            y_pred = results[f"y_pred_{src}"].copy()
            y_pred[rejected] = -1
            out[f"y_pred_{src}"] = y_pred
            n_rej = int(rejected.sum())
            _log(f"[{full_src}] Rechazados por confianza < {min_confidence:.2f}: "
                 f"{n_rej:,} / {len(conf):,} ({100*n_rej/max(len(conf),1):.1f}%)")
    return out


# ─── plot de confianza ────────────────────────────────────────────────────────

def plot_confidence(results, head_names, out_dir, tb_energies=None, prefix="tb"):
    """Histograma de confianza (max softmax) por clase predicha."""
    if plt is None:
        return

    class_names = [LABEL_INT_TO_NAME.get(i, f"class_{i}") for i in range(3)]
    colors      = ["steelblue", "tab:orange", "tab:green"]
    elabel      = _energy_label(tb_energies)

    for src, full_src in [("agg", "aggregate"), ("emb", "embedding")]:
        if full_src not in head_names:
            continue
        logits = results[f"logits_{src}"]
        valid  = np.isfinite(logits).all(axis=1)
        if not valid.any():
            continue
        probs  = _softmax(logits[valid])
        conf   = probs.max(axis=1)
        y_pred = results[f"y_pred_{src}"][valid]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Izq: distribución global de confianza
        ax = axes[0]
        ax.hist(conf, bins=50, color="steelblue", alpha=0.8)
        for thr in [0.5, 0.7, 0.9]:
            n_above = (conf >= thr).sum()
            ax.axvline(thr, color="red", ls="--", lw=1,
                       label=f"≥{thr:.0%}: {n_above:,} ({100*n_above/len(conf):.0f}%)")
        ax.set_xlabel("max(softmax) — confianza")
        ax.set_ylabel("eventos")
        ax.set_title(f"Distribución de confianza [{full_src}] — {elabel}")
        ax.legend(fontsize=8)

        # Der: confianza por clase predicha
        ax = axes[1]
        for cls in range(3):
            m = y_pred == cls
            if m.sum() == 0:
                continue
            ax.hist(conf[m], bins=40, alpha=0.65, color=colors[cls],
                    label=f"{class_names[cls]} (n={m.sum():,})")
        ax.set_xlabel("max(softmax) — confianza")
        ax.set_ylabel("eventos")
        ax.set_title(f"Confianza por clase predicha [{full_src}]")
        ax.legend(fontsize=8)

        fig.tight_layout()
        path = os.path.join(out_dir, f"{prefix}_{full_src}_confidence.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        _log(f"Guardado: {path}")


def _write_testbeam_summary(results, event_indices, energy_arr, tb_energies,
                             min_hits, head_names, out_dir, prefix="tb"):
    lines = ["=" * 65, "RESUMEN INFERENCIA TEST-BEAM", "=" * 65]
    lines.append(f"Eventos seleccionados : {len(event_indices):,}")
    lines.append(f"Filtro energia        : {_energy_label(tb_energies)}")
    lines.append(f"Filtro min_hits       : {min_hits}")

    class_names = [LABEL_INT_TO_NAME.get(i, str(i)) for i in range(3)]

    for src, y_pred in [("aggregate", results["y_pred_agg"]),
                        ("embedding", results["y_pred_emb"])]:
        if src not in head_names:
            continue
        lines.append(f"\n[Latente: {src}]")
        for cls, cname in enumerate(class_names):
            n = int((y_pred == cls).sum())
            pct = 100 * n / max(len(y_pred), 1)
            lines.append(f"  {cname:<10}: {n:>7,}  ({pct:5.1f}%)")

    txt = "\n".join(lines) + "\n"
    path = os.path.join(out_dir, f"{prefix}_summary.txt")
    with open(path, "w") as f:
        f.write(txt)
    _log(f"Guardado: {path}")
    print(txt)


def run_testbeam_mode(args, ae, head_ckpts, sim_stats, device):
    """Orquesta el modo test-beam completo."""
    compute_hits_per_plane_fn, lateral_longitudinal_fn = _try_import_lat_long()

    # Filtrado de eventos
    event_indices, offsets, energy_arr = _filter_testbeam_events(
        args.testbeam_path,
        tb_energies   = args.tb_energies,
        min_hits      = args.min_hits_tb,
        max_events    = args.max_events_tb,
    )
    if len(event_indices) == 0:
        _log("ERROR: ningun evento pasa los filtros de test-beam. Revisa --tb_energies / --min_hits_tb.")
        return

    # Cargar cabezas (respetando --latent_source)
    use_src = getattr(args, "latent_source", "both")
    head_agg = (_load_head(head_ckpts.get("aggregate"), device)
                if use_src in ("aggregate", "both") else None)
    head_emb = (_load_head(head_ckpts.get("embedding"), device)
                if use_src in ("embedding", "both") else None)
    if head_agg is None and head_emb is None:
        _log("ERROR: no hay cabezas cargadas. Especifica --head_aggregate / --head_embedding "
             "y comprueba --latent_source.")
        return

    head_names = set()
    if head_agg is not None: head_names.add("aggregate")
    if head_emb is not None: head_names.add("embedding")
    _log(f"Cabezas activas: {head_names} (--latent_source={use_src})")

    # Inferencia + lat/long
    _log("Lanzando inferencia test-beam...")
    results = _testbeam_inference(
        ae, head_agg, head_emb,
        args.testbeam_path, event_indices, offsets, energy_arr,
        sim_stats,
        use_scalar  = args.use_scalar,
        device      = device,
        batch_size  = args.batch_size,
        compute_hits_per_plane_fn = compute_hits_per_plane_fn,
        lateral_longitudinal_fn   = lateral_longitudinal_fn,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = "tb"

    # Guardar predicciones crudas
    save_path = getattr(args, "save_predictions", None)
    if save_path:
        save_predictions(results, event_indices, offsets,
                         args.testbeam_path, args.tb_energies,
                         args.min_hits_tb, head_names, save_path)

    # Aplicar umbral de confianza (opcionalmente)
    min_conf = getattr(args, "min_confidence", 0.0)
    results = apply_confidence_filter(results, min_conf, head_names)

    # Plots por latente activo
    plot_confidence(results, head_names, args.output_dir,
                    tb_energies=args.tb_energies, prefix=prefix)

    for src, y_pred in [("aggregate", results["y_pred_agg"]),
                        ("embedding", results["y_pred_emb"])]:
        if src not in head_names:
            continue
        pfx = f"{prefix}_{src}"
        _log(f"Generando plots para latente '{src}' ...")

        plot_nhits_by_class(
            results["nhits"], y_pred,
            args.output_dir,
            tb_energies = args.tb_energies,
            prefix      = pfx,
        )
        plot_lat_long_2d(
            results["lat"], results["long"], y_pred,
            args.output_dir,
            tb_energies = args.tb_energies,
            prefix      = pfx,
        )
        for hits_range in GRID_HITS_RANGES:
            plot_3d_grid(
                args.testbeam_path,
                event_indices, offsets,
                results["nhits"], y_pred,
                args.output_dir,
                tb_energies = args.tb_energies,
                prefix      = pfx,
                hits_range  = hits_range,
            )

    _write_testbeam_summary(
        results, event_indices, energy_arr,
        args.tb_energies, args.min_hits_tb,
        head_names, args.output_dir, prefix=prefix,
    )
    _log("Inferencia test-beam completada.")


# ============================================================
# Modo 3: análisis desde predicciones guardadas
# ============================================================

def run_analysis_mode(args):
    """Carga predicciones crudas de un NPZ y regenera plots/estadísticas.

    No necesita GPU ni el AE — es puramente análisis post-hoc.
    Permite explorar umbrales de confianza sin repetir la inferencia.
    """
    results, event_indices, offsets, h5_path, tb_energies, min_hits, head_names = \
        load_predictions(args.load_predictions)

    # Sobreescribir energías si se especifican por CLI (filtro adicional)
    if args.tb_energies:
        _log(f"Filtrando por energía: {args.tb_energies} GeV")
        energy = results["energy"]
        e_mask = np.zeros(len(energy), dtype=bool)
        for e_val in args.tb_energies:
            e_mask |= np.abs(energy - float(e_val)) < 0.5
        for key in results:
            arr = results[key]
            if isinstance(arr, np.ndarray) and arr.shape[0] == len(energy):
                results[key] = arr[e_mask]
        event_indices = event_indices[e_mask]
        _log(f"  → {e_mask.sum():,} eventos tras filtro de energía")
        tb_energies = args.tb_energies

    # Aplicar umbral de confianza
    min_conf = getattr(args, "min_confidence", 0.0)
    results = apply_confidence_filter(results, min_conf, head_names)

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = "tb"

    plot_confidence(results, head_names, args.output_dir,
                    tb_energies=tb_energies, prefix=prefix)

    for src, y_pred in [("aggregate", results["y_pred_agg"]),
                        ("embedding", results["y_pred_emb"])]:
        if src not in head_names:
            continue
        pfx = f"{prefix}_{src}"
        _log(f"Generando plots para latente '{src}' ...")

        plot_nhits_by_class(
            results["nhits"], y_pred,
            args.output_dir,
            tb_energies = tb_energies,
            prefix      = pfx,
        )
        plot_lat_long_2d(
            results["lat"], results["long"], y_pred,
            args.output_dir,
            tb_energies = tb_energies,
            prefix      = pfx,
        )
        for hits_range in GRID_HITS_RANGES:
            plot_3d_grid(
                h5_path,
                event_indices, offsets,
                results["nhits"], y_pred,
                args.output_dir,
                tb_energies = tb_energies,
                prefix      = pfx,
                hits_range  = hits_range,
            )

    # Resumen en texto
    energy_arr_dummy = results["energy"]  # ya filtrado
    _write_testbeam_summary(
        results, event_indices, energy_arr_dummy,
        tb_energies, min_hits, head_names, args.output_dir, prefix=prefix,
    )
    _log("Análisis completado.")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ae_ckpt", default=None,
                   help="Checkpoint del autoencoder. Requerido salvo con --load_predictions.")
    p.add_argument("--finetune_ckpt", default=None,
                   help="Bundle de fine-tuning (ae_state_dict + head_state_dict). "
                        "Si se da, ignora --ae_ckpt y --head_aggregate/--head_embedding: "
                        "carga el AE fine-tuned y la cabeza del propio bundle, ruteando al "
                        "latent_source guardado en el checkpoint.")
    p.add_argument("--cfg", "-c", default="config/model_cfg_clf_head.yml")

    # Sim datasets
    for name in ["electron", "pion", "muon"]:
        p.add_argument(f"--{name}_train", default=None)
        p.add_argument(f"--{name}_test",  default=None)

    # Preprocesamiento
    p.add_argument("--use_scalar",   action="store_true")
    p.add_argument("--use_one_hot",  action="store_true")
    p.add_argument("--z_norm",       action="store_true")
    p.add_argument("--norm",         choices=["z_norm", "minmax"], default=None)
    p.add_argument("--stats_yaml",   default="config/clf_combined_train_stats.yml")

    # Cabezas NN
    p.add_argument("--head_aggregate", default=None)
    p.add_argument("--head_embedding", default=None)

    # SVC (solo modo sim)
    p.add_argument("--svc_C",         type=float, default=1.0)
    p.add_argument("--svc_gamma",     default="scale")
    p.add_argument("--svc_max_train", type=int,   default=20000)

    # Comun
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--max_events",  type=int, default=0)
    p.add_argument("--device",      default=None)
    p.add_argument("--output_dir",  "-o", default="eval_clf_output")

    # ── Test-beam ────────────────────────────────────────────────────────────
    p.add_argument("--testbeam_path", default=None,
                   help="Ruta al HDF5 de test-beam. Activa el modo test-beam.")
    p.add_argument("--tb_energies", nargs="+", type=float, default=None,
                   metavar="E_GEV",
                   help="Energias de haz a seleccionar (GeV). "
                        "Valores validos: 5 8 10 15 20 25 30 40 50 60 70 80 90 100 110 120. "
                        "Sin este arg se usan todos los eventos.")
    p.add_argument("--min_hits_tb", type=int, default=20,
                   help="Minimo de hits por evento en test-beam (default: 20).")
    p.add_argument("--max_events_tb", type=int, default=50000,
                   help="Maximo de eventos test-beam tras filtrar (default: 50000; 0=sin limite).")
    p.add_argument("--latent_source", choices=["aggregate", "embedding", "both"],
                   default="both",
                   help="Latente a usar: 'aggregate', 'embedding', o 'both' (default).")

    # ── Guardar / cargar predicciones crudas ─────────────────────────────────
    p.add_argument("--save_predictions", default=None, metavar="PATH.npz",
                   help="Guarda predicciones crudas (logits, latentes, metadata) "
                        "en un archivo .npz tras la inferencia.")
    p.add_argument("--load_predictions", default=None, metavar="PATH.npz",
                   help="Carga predicciones guardadas con --save_predictions y "
                        "salta la inferencia (no necesita GPU ni AE). "
                        "Solo regenera plots y estadísticas.")
    p.add_argument("--min_confidence", type=float, default=0.0,
                   help="Umbral mínimo de max(softmax) para aceptar una predicción. "
                        "Eventos por debajo quedan como clase -1 (rechazado). "
                        "Default: 0.0 (sin umbral).")

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Modo análisis (sin GPU, sin AE) ─────────────────────────────────────
    if getattr(args, "load_predictions", None):
        run_analysis_mode(args)
        return

    if not _TORCH_AVAILABLE:
        print("ERROR: torch no está disponible en este entorno.\n"
              "Para modo análisis usa --load_predictions PATH.npz (no requiere torch).\n"
              "Para inferencia instala torch y las dependencias del modelo.")
        sys.exit(1)

    if not args.ae_ckpt and not args.finetune_ckpt:
        print("ERROR: especifica --ae_ckpt (AE + cabezas por separado) "
              "o --finetune_ckpt (bundle de fine-tuning) en modo inferencia.")
        sys.exit(1)

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    _log(f"Dispositivo: {device}")

    cfg_models   = yaml.safe_load(open(args.cfg))
    n_classes    = cfg_models.get("classifier", {}).get("n_classes", 3)
    norm_type    = args.norm or ("z_norm" if args.z_norm else None)
    preprocessing_cfg = {"use_scalar": args.use_scalar,
                         "use_one_hot": args.use_one_hot,
                         "norm_type": norm_type, "z_norm": args.z_norm}

    train_paths  = [p for p in [args.electron_train, args.pion_train, args.muon_train]
                    if p is not None]

    # ── AE + cabezas: bundle de fine-tuning vs ckpts separados ──────────────
    if args.finetune_ckpt:
        # El bundle trae AE fine-tuned + cabeza; rutea al latent_source guardado.
        ae = load_finetuned_ae(args.finetune_ckpt, cfg_models, device)
        raw_ft = torch.load(args.finetune_ckpt, map_location="cpu")
        ft_src = raw_ft.get("latent_source", getattr(args, "latent_source", "aggregate"))
        if getattr(args, "latent_source", None) not in (None, "both", ft_src):
            _log(f"WARNING: --latent_source={args.latent_source} ignorado; el bundle "
                 f"fue entrenado sobre '{ft_src}'.")
        args.latent_source = ft_src
        head_ckpts = {"aggregate": None, "embedding": None}
        head_ckpts[ft_src] = args.finetune_ckpt
        _log(f"Bundle de fine-tuning: cabeza '{ft_src}' desde {args.finetune_ckpt}")
    else:
        head_ckpts = {"aggregate": args.head_aggregate, "embedding": args.head_embedding}
        # AE congelado (original)
        ae = load_frozen_ae(args.ae_ckpt, cfg_models, device)

    # Stats combinadas (sim train)
    sim_stats = {}
    if norm_type in ("z_norm", "minmax") and train_paths:
        sim_stats = compute_or_load_combined_stats(train_paths, norm_type, args.stats_yaml)

    # ── Modo test-beam ───────────────────────────────────────────────────────
    if args.testbeam_path:
        if not train_paths:
            _log("WARNING: --testbeam_path activo pero no se dieron *_train paths. "
                 "Las stats de normalizacion se cargaran del YAML existente.")
            if norm_type and os.path.exists(args.stats_yaml):
                import yaml as _yaml
                raw = _yaml.safe_load(open(args.stats_yaml))
                sim_stats = raw.get("stats", raw)
            else:
                _log("ERROR: no hay stats disponibles. Especifica *_train o --stats_yaml existente.")
                sys.exit(1)
        run_testbeam_mode(args, ae, head_ckpts, sim_stats, device)
        return

    # ── Modo evaluacion sim ──────────────────────────────────────────────────
    if not train_paths:
        _log("ERROR: especifica al menos un *_train path.")
        sys.exit(1)

    train_specs = [(args.electron_train, 0, "electron"),
                   (args.pion_train,     1, "pion"),
                   (args.muon_train,     2, "muon")]
    test_specs  = [(args.electron_test, 0, "electron"),
                   (args.pion_test,     1, "pion"),
                   (args.muon_test,     2, "muon")]

    _log("Extrayendo latentes de TRAIN...")
    train_loader, _ = _loader_from_specs(train_specs, sim_stats, norm_type,
                                         preprocessing_cfg, args.batch_size,
                                         args.max_events)
    L_train = extract_latents(ae, train_loader, device, args.use_scalar, args.use_one_hot)

    _log("Extrayendo latentes de TEST...")
    test_loader, _ = _loader_from_specs(test_specs, sim_stats, norm_type,
                                        preprocessing_cfg, args.batch_size,
                                        args.max_events)
    L_test = extract_latents(ae, test_loader, device, args.use_scalar, args.use_one_hot)

    y_train, y_test = L_train["labels"], L_test["labels"]
    _log(f"Train: {len(y_train)} eventos | Test: {len(y_test)} eventos")

    rows = []
    use_src = getattr(args, "latent_source", "both")
    sources_to_eval = (["aggregate", "embedding"] if use_src == "both"
                       else [use_src])

    for latent_source in sources_to_eval:
        lat_tr = L_train[latent_source]
        lat_te = L_test[latent_source]
        _log(f"=== Latente '{latent_source}' (dim={lat_tr.shape[-1]}) ===")

        _log("  Ajustando SVC...")
        t0 = time.time()
        svc_res = run_svc(lat_tr, y_train, lat_te, y_test, n_classes,
                          args.svc_C, args.svc_gamma, args.svc_max_train)
        _log(f"  SVC acc={svc_res['accuracy']:.4f} f1={svc_res['f1_macro']:.4f} "
             f"(n_train={svc_res['n_train']}, {time.time()-t0:.1f}s)")
        plot_confusion(svc_res["confusion"], svc_res["names"],
                       f"SVC — {latent_source} (acc={svc_res['accuracy']:.3f})",
                       os.path.join(args.output_dir, f"cm_svc_{latent_source}.png"))
        rows.append(("SVC", latent_source, svc_res["accuracy"], svc_res["f1_macro"]))

        ckpt = head_ckpts[latent_source]
        if ckpt and os.path.exists(ckpt):
            _log(f"  Evaluando cabeza NN: {ckpt}")
            nn_res = run_nn_head(ckpt, lat_tr, lat_te, y_test, n_classes, device)
            _log(f"  NN  acc={nn_res['accuracy']:.4f} f1={nn_res['f1_macro']:.4f}")
            plot_confusion(nn_res["confusion"], nn_res["names"],
                           f"NN head — {latent_source} (acc={nn_res['accuracy']:.3f})",
                           os.path.join(args.output_dir, f"cm_nn_{latent_source}.png"))
            rows.append(("NN_head", latent_source, nn_res["accuracy"], nn_res["f1_macro"]))
        else:
            _log(f"  (sin checkpoint NN para '{latent_source}', omito)")

    summary = os.path.join(args.output_dir, "clf_comparison.txt")
    with open(summary, "w") as f:
        f.write("=== Comparacion de clasificacion sobre el latente del AE ===\n\n")
        f.write(f"AE ckpt : {args.ae_ckpt}\n")
        f.write(f"Test    : {len(y_test)} eventos held-out\n\n")
        f.write(f"{'method':<10}{'latent':<12}{'accuracy':>10}{'f1_macro':>10}\n")
        f.write("-" * 42 + "\n")
        for m, lat, acc, f1 in sorted(rows, key=lambda r: -r[2]):
            f.write(f"{m:<10}{lat:<12}{acc:>10.4f}{f1:>10.4f}\n")
    _log(f"Resumen guardado en: {summary}")
    with open(summary) as f:
        print(f.read())
    _log("Evaluacion completada.")


if __name__ == "__main__":
    main()
