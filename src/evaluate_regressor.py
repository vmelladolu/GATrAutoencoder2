import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import polars as pl
import torch
import yaml
from tqdm import tqdm 

import lightning as L

from torch_geometric.loader import DataLoader

from src.utils.logging import wprint
from src.utils.batch_utils import build_batch
from utils.datasets import FlatSDHCALDataset, HitsDataset
from utils.lightining_trainer import LightningGATrRegressor

from src.utils.results_utils import metrics, summarize_by_energy, plot_results  

# z_norm_yaml_path = "/nfs/cms/arqolmo/SDHCAL_Energy/GATrAutoencoder/stats_hdf5.yml"


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluación de GATrRegressor desde .pt/.ckpt con métricas y plots tipo MLPFit"
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Ruta a .pt o .ckpt")
    parser.add_argument("--cfg", "-c", type=str, default="config/model_cfg_regressor.yml", help="YAML config")
    parser.add_argument("--eval_cfg", type=str, default=None, help="YAML con parámetros de evaluación")
    parser.add_argument("--data_paths", nargs="+", default=None, help="Lista de ficheros de datos (HDF5/NPZ)")
    parser.add_argument("--mode", choices=["memory", "lazy"], default=None, help="Modo de carga de datos (solo legacy npz jagged)")
    parser.add_argument("--use_scalar", action="store_true", default=None, help="Incluir escalares")
    parser.add_argument("--use_one_hot", action="store_true", default=None, help="One-hot threshold")
    parser.add_argument("--use_time", action="store_true", default=None, help="Incluir tiempo de hit como input escalar extra")
    parser.add_argument("--use_log", action="store_true", default=None, help="Target entrenado en log(E)")
    parser.add_argument("--z_norm", action="store_true", default=None, help="Aplicar z-score a features")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size de evaluación")
    parser.add_argument("--num-workers", type=int, default=None, help="Workers del DataLoader")
    parser.add_argument("--seed", type=int, default=None, help="Semilla")
    parser.add_argument("--gpu", type=int, default=None, help="GPU id (single-GPU, legacy)")
    parser.add_argument("--devices", type=int, default=None, help="Número de GPUs para inferencia (default: 1)")
    parser.add_argument("--accelerator", type=str, default=None, help="Accelerator: gpu/cpu/auto")
    parser.add_argument("--strategy", type=str, default=None, help="Strategy: ddp/auto")
    parser.add_argument("-o", "--out", type=str, default=None, help="Directorio de salida")
    parser.add_argument("--plot", action="store_true", default=None, help="Generar plots con results_utils.plot_results")
    parser.add_argument("-s", "--style", action="count", default=0, help="Whether to use a style for matplotlib.")
    parser.add_argument("--plt-style", default="/nfs/cms/arqolmo/SDHCAL_Energy/utils/newams.mplstyle", required=False, help="Matplotlib style to use.")
    parser.add_argument("--z-norm-path", required=False, default=None, type=str)
    return parser.parse_args()


def _load_eval_cfg(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"El archivo --eval_cfg no contiene un diccionario YAML: {path}")
    return raw.get("evaluate", raw)


def _pick(cli_value, cfg: Dict[str, Any], key: str, default):
    if cli_value is not None:
        return cli_value
    return cfg.get(key, default)


def _make_loggers() -> Dict[str, logging.Logger]:
    logger = logging.getLogger("evaluate_regressor")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
    return {"io": logger, "processing": logger}


class _FitterProxy:
    """Adapter mínimo para reutilizar plot_results(fitter, df)."""

    def __init__(self, output_dir: str, target_col: str, loggers: Dict[str, logging.Logger]):
        self.cfg = {
            "output_results_dir": output_dir,
            "output_fig_path": os.path.join(output_dir, "Images"),
            "trained": True,
        }
        self.target_col = target_col
        self.loggers = loggers
        self.use_wandb = False


def _resolve_device(gpu: Optional[int]) -> torch.device:
    if torch.cuda.is_available():
        if gpu is not None:
            return torch.device(f"cuda:{gpu}")
        return torch.device("cuda")
    return torch.device("cpu")


class _PredictWrapper(L.LightningModule):
    """
    Wrapper ligero para usar Trainer.predict() con multi-GPU.
    Delega el forward al módulo original y devuelve predicciones + targets.
    """
    def __init__(self, module, use_scalar, use_one_hot, use_time, use_log, z_norm, stats):
        super().__init__()
        self.module = module
        self.use_scalar = use_scalar
        self.use_one_hot = use_one_hot
        self.use_time = use_time
        self.use_log = use_log
        self.z_norm = z_norm
        self.stats = stats

    def predict_step(self, batch, batch_idx):
        data = build_batch(
            batch,
            use_scalar=self.use_scalar,
            use_one_hot=self.use_one_hot,
            use_time=self.use_time,
            use_log=self.use_log,
            use_energy=True,
            z_norm=self.z_norm,
            stats=self.stats,
        )
        mv_v = data["mv_v_part"].to(self.device)
        mv_s = data["mv_s_part"].to(self.device)
        scalars = data["scalars"].to(self.device)
        batch_idx_t = data["batch_idx"].to(self.device)
        target = data["logenergy"].to(self.device)

        extra_global = {}
        for key in ("n_thr1", "n_thr2", "n_thr3"):
            val = data.get(key)
            if val is not None:
                extra_global[key] = val.to(self.device)

        y_pred = self.module.model(mv_v, mv_s, scalars, batch_idx_t,
                                   extra_global_features=extra_global or None).squeeze(-1)
        return {"y_pred": y_pred, "y_true": target.view(-1)}


def _resolve_trainer_runtime(args, eval_cfg):
    """Determina accelerator, devices y strategy para el Trainer."""
    accelerator = _pick(args.accelerator, eval_cfg, "accelerator", "auto")
    devices_cfg = _pick(args.devices, eval_cfg, "devices", 1)
    strategy = _pick(args.strategy, eval_cfg, "strategy", "auto")

    # Compatibilidad: --gpu prevalece si no se pasan --devices
    if args.gpu is not None and args.devices is None:
        devices_cfg = [args.gpu]
        accelerator = "gpu"
        strategy = "auto"

    if accelerator in ("gpu", "auto") and torch.cuda.is_available():
        available = torch.cuda.device_count()
        if isinstance(devices_cfg, int) and devices_cfg > available:
            wprint(f"Solicitadas {devices_cfg} GPUs, pero solo hay {available}. Se usarán {available}.")
            devices_cfg = available
        if isinstance(devices_cfg, int) and devices_cfg > 1 and strategy == "auto":
            strategy = "ddp"
    elif not torch.cuda.is_available():
        accelerator = "cpu"
        devices_cfg = 1
        strategy = "auto"

    return accelerator, devices_cfg, strategy


def _prepare_batch(batch, device, use_scalar, use_one_hot, use_time, use_log, z_norm, stats):
    """
    Convierte un Batch de PyG a tensores usando build_batch (misma lógica que el entrenamiento).
    Devuelve: (mv_v_part, mv_s_part, scalars, batch_idx, target, extra_global_features)
    """
    data = build_batch(
        batch,
        use_scalar=use_scalar,
        use_one_hot=use_one_hot,
        use_time=use_time,
        use_log=use_log,
        use_energy=True,
        z_norm=z_norm,
        stats=stats
    )
    mv_v = data["mv_v_part"].to(device)
    mv_s = data["mv_s_part"].to(device)
    scalars = data["scalars"].to(device)
    batch_idx = data["batch_idx"].to(device)
    target = data["logenergy"].to(device)
    extra_global = {}
    for key in ("n_thr1", "n_thr2", "n_thr3"):
        val = data.get(key)
        if val is not None:
            extra_global[key] = val.to(device)
    return mv_v, mv_s, scalars, batch_idx, target, extra_global or None


def _load_checkpoint_weights(module: LightningGATrRegressor, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if not isinstance(ckpt, dict):
        # State dict directo (bare OrderedDict)
        module.model.load_state_dict(ckpt, strict=True)
        return

    # Formato de on_fit_end: {"epoch": ..., "model_state_dict": {...}, ...}
    if "model_state_dict" in ckpt:
        module.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return

    # Formato de Lightning checkpoint: {"state_dict": {"model.layer...": ...}, ...}
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
        # Intentar cargar directamente en LightningModule
        try:
            module.load_state_dict(state, strict=True)
            return
        except Exception:
            pass
        # Intentar en modelo interno eliminando prefijo "model."
        if hasattr(module, "model"):
            stripped = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
            if stripped:
                module.model.load_state_dict(stripped, strict=True)
                return

    raise RuntimeError(
        f"No se pudo cargar checkpoint: {ckpt_path}\n"
        f"Claves encontradas: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
    )


def main():
    args = parse_args()

    if args.style > 0:
        import matplotlib.pyplot as plt
        plt.style.use(args.plt_style)
        wprint(f"Matplotlib style '{args.plt_style}' aplicado.")
        
    eval_cfg = _load_eval_cfg(args.eval_cfg)
    checkpoint_path = _pick(args.checkpoint, eval_cfg, "checkpoint", None)
    if checkpoint_path is None:
        raise ValueError("Debes indicar el checkpoint con --checkpoint o en --eval_cfg (clave 'checkpoint').")

    data_paths = _pick(
        args.data_paths,
        eval_cfg,
        "data_paths",
        ["/nfs/cms/arqolmo/SDHCAL_Energy/GATrAutoencoder/flat_all.npz"],
    )
    mode = _pick(args.mode, eval_cfg, "mode", "lazy")
    use_scalar = bool(_pick(args.use_scalar, eval_cfg, "use_scalar", False))
    use_one_hot = bool(_pick(args.use_one_hot, eval_cfg, "use_one_hot", False))
    use_time = bool(_pick(args.use_time, eval_cfg, "use_time", False))
    use_log = bool(_pick(args.use_log, eval_cfg, "use_log", False))
    z_norm = bool(_pick(args.z_norm, eval_cfg, "z_norm", False))
    z_norm_path = _pick(args.z_norm_path, eval_cfg, "z_norm_path", None)
    inverse_n1_n2 = bool(eval_cfg.get("inverse_n1_n2", False))
    batch_size = int(_pick(args.batch_size, eval_cfg, "batch_size", 64))
    num_workers = int(_pick(args.num_workers, eval_cfg, "num_workers", 10))
    seed = int(_pick(args.seed, eval_cfg, "seed", 42))
    output_dir = _pick(args.out, eval_cfg, "out", "./eval_regressor")
    do_plot = bool(_pick(args.plot, eval_cfg, "plot", False))

    accelerator, devices_cfg, strategy = _resolve_trainer_runtime(args, eval_cfg)

    os.makedirs(output_dir, exist_ok=True)
    L.pytorch.seed_everything(seed, workers=True)

    loggers = _make_loggers()
    device = _resolve_device(args.gpu)
    loggers["io"].info(f"Usando dispositivo: {device}  |  Trainer: accelerator={accelerator}, devices={devices_cfg}, strategy={strategy}")
    if args.eval_cfg:
        loggers["io"].info(f"Configuración de evaluación cargada desde: {args.eval_cfg}")

    cfg_models = yaml.safe_load(open(args.cfg, "r"))
    cfg_enc = cfg_models["encoder"]
    cfg_agg = cfg_models["aggregation"]

    # Ajustar in_s_channels dinámicamente igual que en entrenamiento
    n_thr_channels = 3 if use_one_hot else 1
    in_s_channels = n_thr_channels + (1 if use_time else 0)
    if cfg_enc["in_s_channels"] != in_s_channels:
        wprint(
            f"Ajustando encoder in_s_channels: {cfg_enc['in_s_channels']} → {in_s_channels} "
            f"(use_one_hot={use_one_hot}, use_time={use_time})"
        )
        cfg_enc["in_s_channels"] = in_s_channels

    preprocessing_cfg = {
        "use_scalar": use_scalar,
        "use_one_hot": use_one_hot,
        "use_time": use_time,
        "use_energy": True,
        "use_log": use_log,
        "z_norm": z_norm,
        "z_norm_yaml_path": z_norm_path,
    }
    filters_cfg = cfg_models.get("filters", {})

    # ---- Cargar dataset completo (sin split) ----
    import h5py as _h5py
    if len(data_paths) == 1:
        path = data_paths[0]
        ext = os.path.splitext(path)[1].lower()
        if ext in [".h5", ".hdf5"]:
            with _h5py.File(path, "r") as probe:
                is_flat = "offsets" in probe
        elif ext == ".npz":
            probe = np.load(path, allow_pickle=False)
            is_flat = "offsets" in probe
        else:
            is_flat = False

        if is_flat:
            dataset = FlatSDHCALDataset(path, preprocessing_cfg=preprocessing_cfg, filters=filters_cfg)
        else:
            dataset = HitsDataset(data_paths, mode=mode)
    else:
        dataset = HitsDataset(data_paths, mode=mode)

    loggers["io"].info(f"Dataset cargado: {dataset.len()} eventos (100 % del input)")

    # Patch testbeam threshold labelling: swap thr1 ↔ thr2 before any processing
    if inverse_n1_n2 and isinstance(dataset, FlatSDHCALDataset):
        dataset._thr1, dataset._thr2 = dataset._thr2, dataset._thr1
        # Keep _thr consistent: remap 1→2 and 2→1 in place
        thr = dataset._thr
        mask1 = thr == 1
        mask2 = thr == 2
        thr[mask1] = 2
        thr[mask2] = 1
        loggers["io"].info("inverse_n1_n2=True: threshold labels 1 and 2 have been swapped.")

    eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    if z_norm:
        if z_norm_path is not None:
            with open(z_norm_path, "r") as f:
                stats_raw = yaml.safe_load(f)
            # The YAML has structure {"norm_type": ..., "stats": {...}};
            # _apply_preprocessing_inplace expects only the inner stats dict.
            stats = stats_raw.get("stats", stats_raw) if isinstance(stats_raw, dict) else stats_raw
        else:
            raise ValueError("If using z norm you must provide the yaml z norm file")
    else:
        stats = None

    # Apply preprocessing (normalization + log-energy) to the dataset in-place.
    # FlatSDHCALDataset.__init__ intentionally skips this step (it's designed to be
    # called by make_pf_splits after the train/val split). For evaluation we do it here.
    if isinstance(dataset, FlatSDHCALDataset):
        norm_type = "z_norm" if z_norm else None
        dataset._apply_preprocessing_inplace(stats or {}, norm_type, preprocessing_cfg)
    module = LightningGATrRegressor(
        cfg_enc=cfg_enc,
        cfg_agg=cfg_agg,
        class_weights=getattr(dataset, "weights", None),
        use_scalar=use_scalar,
        use_one_hot=use_one_hot,
        use_time=use_time,
        use_log=use_log,
        z_norm=z_norm,
        stats=stats,
        learning_rate=cfg_models.get("optimizer", {}).get("lr", 1e-3),
        max_epochs=1,
        plot_every=0,
        output_path=output_dir,
        scheduler_cfg=cfg_models.get("scheduler", {"type": "cosine"}),
        optimizer_cfg=cfg_models.get("optimizer", {"weight_decay": 0.0001}),
        debug_cfg=cfg_models.get("debug", {"debug_grad_step": 0, "debug_event_step": 0}),
    ).to(device)

    _load_checkpoint_weights(module, checkpoint_path, device)
    module.eval()
    loggers["io"].info(f"Checkpoint cargado: {checkpoint_path}")

    # ---- Inferencia (single-GPU manual o multi-GPU con Trainer.predict) ----
    use_trainer = (
        (isinstance(devices_cfg, int) and devices_cfg > 1)
        or (isinstance(devices_cfg, list) and len(devices_cfg) > 1)
    )

    loggers["io"].info("Evaluando sobre el dataset completo...")

    if use_trainer:
        # ======== Multi-GPU con Lightning Trainer.predict() ========
        loggers["io"].info(f"Modo multi-GPU activado ({devices_cfg} dispositivos).")
        predict_wrapper = _PredictWrapper(
            module, use_scalar, use_one_hot, use_time, use_log, z_norm, stats
        )
        trainer = L.Trainer(
            accelerator=accelerator,
            devices=devices_cfg,
            strategy=strategy,
            logger=False,
            enable_progress_bar=True,
        )
        raw_predictions = trainer.predict(predict_wrapper, dataloaders=eval_loader)
        # raw_predictions es una lista de dicts {"y_pred": ..., "y_true": ...}
        y_true_all, y_pred_all = [], []
        for batch_out in raw_predictions:
            yt = batch_out["y_true"].detach().cpu().numpy().reshape(-1)
            yp = batch_out["y_pred"].detach().cpu().numpy().reshape(-1)
            if use_log:
                yt = np.exp(yt)
                yp = np.exp(yp)
            y_true_all.append(yt)
            y_pred_all.append(yp)

        # Con DDP cada rank tiene solo su shard (~1/N del dataset).
        # Reunimos todos los shards en rank 0 antes de calcular métricas y guardar.
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            local_data = (np.concatenate(y_true_all), np.concatenate(y_pred_all))
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local_data)
            if trainer.is_global_zero:
                y_true_all = [np.concatenate([g[0] for g in gathered])]
                y_pred_all = [np.concatenate([g[1] for g in gathered])]
            else:
                # Los ranks no-cero no necesitan continuar
                return
    else:
        # ======== Single-GPU: bucle manual (más simple, sin overhead) ========
        y_true_all, y_pred_all = [], []
        with torch.inference_mode():
            for batch in tqdm(eval_loader):
                mv_v, mv_s, scalars, batch_idx_t, y_true, extra_global = _prepare_batch(
                    batch, device, use_scalar, use_one_hot, use_time, use_log, z_norm, stats)
                y_pred = module.model(mv_v, mv_s, scalars, batch_idx_t,
                                      extra_global_features=extra_global).squeeze(-1)

                y_true_np = y_true.detach().cpu().numpy().reshape(-1)
                y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)

                if use_log:
                    y_true_np = np.exp(y_true_np)
                    y_pred_np = np.exp(y_pred_np)

                y_true_all.append(y_true_np)
                y_pred_all.append(y_pred_np)

    if len(y_true_all) == 0:
        raise RuntimeError("No se obtuvieron muestras para evaluar.")

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    target_col = "energy"
    pred_df = pl.DataFrame({
        target_col: y_true_all,
        "E_reco": y_pred_all,
        "set": ["all"] * len(y_true_all),
    })

    all_err = metrics(pred_df, target_col)
    by_energy_all = summarize_by_energy(pred_df, target_col)

    summary = {
        "method": "GATrRegressor",
        "checkpoint_path": checkpoint_path,
        "relative_error": {
            "all_mean_abs": all_err,
            "by_energy": {"all": by_energy_all},
        },
        "n": {"all": int(len(pred_df))},
    }

    # Guardado de resultados
    pred_path = os.path.join(output_dir, "predictions_all.parquet")
    summary_path = os.path.join(output_dir, "summary_all.json")
    pred_df.write_parquet(pred_path)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    loggers["processing"].info(f"[GATr][EVAL] All-data relative absolute error (MAE): {all_err:.6f}")
    loggers["io"].info(f"Predicciones guardadas en: {pred_path}")
    loggers["io"].info(f"Resumen guardado en: {summary_path}")

    if do_plot:
        fitter_proxy = _FitterProxy(output_dir=output_dir, target_col=target_col, loggers=loggers)
        plot_results(fitter_proxy, pred_df)
        loggers["io"].info(f"Plots guardados en: {os.path.join(output_dir, 'Images')}")


if __name__ == "__main__":
    main()