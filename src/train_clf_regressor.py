import os
from src.utils.logging import wprint

import torch
import yaml

from models.gatr_clf_regressor import GATrClassifierRegressor
from utils.datasets import SDHCALRegressorDataModule
from utils.lightning_clf_trainer import LightningGATrClfRegressor


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

try:
    import wandb
except ImportError:
    wandb = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def test_forward_on_gpu(model, cfg_enc, device):
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere GPU para el modo --test")
    model = model.to(device)
    model.eval()

    n = 16
    mv_v_part = torch.randn(n, 3, device=device)
    mv_s_part = torch.zeros(n, 1, device=device)
    scalars = torch.randn(n, cfg_enc["in_s_channels"], device=device)
    batch_idx = torch.zeros(n, dtype=torch.long, device=device)

    with torch.no_grad():
        energy_pred, class_logits = model(mv_v_part, mv_s_part, scalars, batch_idx)

    wprint(f"Test OK: energy_pred shape={energy_pred.shape}, class_logits shape={class_logits.shape}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Training script for GATrClassifierRegressor (PyTorch Lightning)"
    )
    # ---- Data ----
    parser.add_argument("--data_paths", nargs="+", help="File list of dataset paths (HDF5 or NPZ).")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio (0-1)")
    parser.add_argument("--mode", choices=["memory", "lazy"], default="lazy", help="Data loading mode")

    # ---- Features ----
    parser.add_argument("--use_scalar",  action="store_true", help="Include scalar features (k) as model input")
    parser.add_argument("--use_one_hot", action="store_true", help="Use one-hot encoding for threshold input")
    parser.add_argument("--use_time",    action="store_true", help="Include hit time as extra scalar input")
    parser.add_argument("--use_energy",  action="store_true", help="Kept for CLI compatibility")

    # ---- Normalisation ----
    parser.add_argument("--z_norm", action="store_true", help="[Alias] Equivalente a --norm z_norm")
    parser.add_argument(
        "--norm", choices=["z_norm", "minmax"], default=None,
        help="Normalisation type for x, y, z (and k/thr/time if applicable).",
    )
    parser.add_argument(
        "--norm_yaml", type=str, default=None,
        help="Path to normalisation stats YAML. Auto-computed if not provided.",
    )

    # ---- Loss ----
    parser.add_argument("--no_weighted_loss", action="store_true",
                        help="Disable energy-distribution-weighted loss")

    # ---- Training ----
    parser.add_argument("--test",       action="store_true", help="Run a forward pass test on GPU and exit")
    parser.add_argument("--epochs",     type=int,   default=1,    help="Training epochs")
    parser.add_argument("--batch_size", type=int,   default=3,    help="Per-device batch size")
    parser.add_argument("--use_log",    action="store_true",      help="Use log(E) as training target")
    parser.add_argument("--plot_every", type=int,   default=0,    help="Plot regression every N epochs (0=auto)")
    parser.add_argument("--cfg", "-c",  type=str,
                        default="config/model_cfg_clf_regressor.yml", help="YAML config path")
    parser.add_argument("--save-every", type=int,   default=500,
                        help="Save checkpoint every N train steps (0 disable)")
    parser.add_argument("-o", "--out",  type=str,   default="./trained_clf_regressor")
    parser.add_argument("--resume",     type=str,   default=None, help="Path to resume checkpoint")

    # ---- Hardware ----
    parser.add_argument("--gpu",         type=int,  default=None, help="GPU id for single-GPU runs")
    parser.add_argument("--gpu_ids",     nargs="+", type=int,    default=None,
                        help="Explicit GPU ids, e.g. --gpu_ids 0 1")
    parser.add_argument("--devices",     type=int,  default=4,   help="Number of devices (default: 4 GPUs)")
    parser.add_argument("--accelerator", type=str,  default="gpu")
    parser.add_argument("--strategy",    type=str,  default="ddp")
    parser.add_argument("--train-num-workers", type=int, default=4)
    parser.add_argument("--val-num-workers",   type=int, default=2)
    parser.add_argument("--lr",   type=float, default=1e-3)
    parser.add_argument("--seed", type=int,   default=42)

    # ---- Classification / binning (new) ----
    parser.add_argument("--n_bins",          type=int,   default=None,
                        help="Number of energy bins for classification (overrides YAML)")
    parser.add_argument("--energy_min_gev",  type=float, default=None,
                        help="Left edge of energy range in GeV (overrides YAML)")
    parser.add_argument("--energy_max_gev",  type=float, default=None,
                        help="Right edge of energy range in GeV (overrides YAML)")
    parser.add_argument("--alpha_cls",       type=float, default=None,
                        help="Classification loss weight (overrides YAML)")
    parser.add_argument("--alpha_lb",        type=float, default=None,
                        help="Load-balancing loss weight (overrides YAML)")

    return parser.parse_args()


def _resolve_trainer_runtime(args):
    accelerator = args.accelerator
    strategy = args.strategy
    devices = args.devices

    if accelerator == "gpu":
        available = torch.cuda.device_count()
        if available == 0:
            wprint("No hay GPUs disponibles. Se cambia a CPU.")
            accelerator = "cpu"
            devices = 1
            strategy = "auto"
        elif args.gpu_ids:
            invalid = [gid for gid in args.gpu_ids if gid < 0 or gid >= available]
            if invalid:
                raise ValueError(f"GPU ids invalidas {invalid}; GPUs disponibles: 0..{available - 1}")
            devices = args.gpu_ids
            strategy = "ddp" if len(devices) > 1 else "auto"
        elif args.gpu is not None and args.devices == 1:
            devices = [args.gpu]
            strategy = "auto"
        else:
            if isinstance(devices, int) and devices > available:
                wprint(f"Solicitadas {devices} GPUs, pero solo hay {available}. Se usaran {available}.")
                devices = available
            if isinstance(devices, int) and devices <= 1:
                strategy = "auto"
            elif strategy == "auto":
                strategy = "ddp"
    elif strategy == "ddp" and isinstance(devices, int) and devices <= 1:
        strategy = "auto"

    if strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=False)

    return accelerator, devices, strategy


def main():
    args = parse_args()
    L.pytorch.seed_everything(args.seed, workers=True)

    cfg_models = yaml.safe_load(open(args.cfg, "r"))
    cfg_enc = cfg_models["encoder"]
    cfg_agg = cfg_models["aggregation"]

    # Adjust in_s_channels dynamically based on active flags
    n_thr_channels = 3 if args.use_one_hot else 1
    in_s_channels = n_thr_channels + (1 if args.use_time else 0)
    if cfg_enc["in_s_channels"] != in_s_channels:
        wprint(
            f"Ajustando encoder in_s_channels: {cfg_enc['in_s_channels']} → {in_s_channels} "
            f"(use_one_hot={args.use_one_hot}, use_time={args.use_time})"
        )
        cfg_enc["in_s_channels"] = in_s_channels

    # Resolve clf hyperparameters: CLI > YAML > default
    clf_cfg = cfg_models.get("clf", {})
    n_bins       = args.n_bins         if args.n_bins         is not None else clf_cfg.get("n_bins", 10)
    energy_min   = args.energy_min_gev if args.energy_min_gev is not None else clf_cfg.get("energy_min_gev", 5.0)
    energy_max   = args.energy_max_gev if args.energy_max_gev is not None else clf_cfg.get("energy_max_gev", 100.0)
    alpha_cls    = args.alpha_cls      if args.alpha_cls      is not None else clf_cfg.get("alpha_cls", 0.1)
    alpha_lb     = args.alpha_lb       if args.alpha_lb       is not None else clf_cfg.get("alpha_lb", 0.05)
    label_smoothing = clf_cfg.get("label_smoothing", 0.0)

    phase1_epochs    = cfg_models.get("training", {}).get("phase1_epochs", 20)
    reg_ramp_epochs  = cfg_models.get("training", {}).get("reg_ramp_epochs", 10)
    use_gumbel       = clf_cfg.get("gumbel_softmax", False)
    tau_start        = clf_cfg.get("tau_start", 1.0)
    tau_end          = clf_cfg.get("tau_end", 0.1)
    tau_anneal_epochs = clf_cfg.get("tau_anneal_epochs", 50)

    # Inject clf settings into cfg_agg so the model can read them
    cfg_agg["clf"] = clf_cfg

    output_path = args.out
    os.makedirs(output_path, exist_ok=True)
    checkpoint_path = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    base_model = GATrClassifierRegressor(cfg_enc=cfg_enc, cfg_agg=cfg_agg, n_bins=n_bins)
    if args.test:
        if args.gpu is not None and torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_forward_on_gpu(base_model, cfg_enc, device)
        return

    data_paths = args.data_paths if args.data_paths else [
        "/nfs/cms/arqolmo/SDHCAL_Energy/GATrAutoencoder/flat_all.npz",
    ]
    norm_type = args.norm or ("z_norm" if args.z_norm else None)
    preprocessing_cfg = {
        "use_scalar":     args.use_scalar,
        "use_one_hot":    args.use_one_hot,
        "use_time":       args.use_time,
        "use_energy":     True,
        "use_log":        args.use_log,
        "norm_type":      norm_type,
        "norm_yaml_path": args.norm_yaml,
        "z_norm":         args.z_norm,
    }
    filters_cfg = cfg_models.get("filters", {})
    wprint(f"Cargando datos desde: {data_paths} con val_ratio={args.val_ratio} y mode={args.mode}")

    datamodule = SDHCALRegressorDataModule(
        data_paths=data_paths,
        val_ratio=args.val_ratio,
        mode=args.mode,
        preprocessing_cfg=preprocessing_cfg,
        filters_cfg=filters_cfg,
        batch_size=args.batch_size,
        train_num_workers=args.train_num_workers,
        val_num_workers=args.val_num_workers,
        seed=args.seed,
        use_weighted_loss=not args.no_weighted_loss,
    )
    datamodule.setup()
    class_weights = datamodule.class_weights

    plot_every = args.plot_every if args.plot_every > 0 else max(1, args.epochs // 10)
    scheduler_cfg = cfg_models.get("scheduler", {"type": "cosine"})
    optimizer_cfg = cfg_models.get("optimizer", {"weight_decay": 0.0001})
    debug_cfg = cfg_models.get("debug", {"debug_grad_step": 500, "debug_event_step": 500})

    module = LightningGATrClfRegressor(
        cfg_enc=cfg_enc,
        cfg_agg=cfg_agg,
        class_weights=class_weights,
        use_scalar=args.use_scalar,
        use_one_hot=args.use_one_hot,
        use_time=args.use_time,
        use_log=args.use_log,
        z_norm=args.z_norm,
        stats=None,
        learning_rate=optimizer_cfg.get("lr", args.lr),
        max_epochs=args.epochs,
        plot_every=plot_every,
        output_path=output_path,
        n_bins=n_bins,
        energy_min_gev=energy_min,
        energy_max_gev=energy_max,
        alpha_cls=alpha_cls,
        alpha_lb=alpha_lb,
        label_smoothing=label_smoothing,
        phase1_epochs=phase1_epochs,
        use_gumbel=use_gumbel,
        tau_start=tau_start,
        tau_end=tau_end,
        tau_anneal_epochs=tau_anneal_epochs,
        scheduler_cfg=scheduler_cfg,
        optimizer_cfg=optimizer_cfg,
        debug_cfg=debug_cfg,
        reg_ramp_epochs=reg_ramp_epochs,
    )

    logger = None
    if wandb is not None:
        logger = WandbLogger(
            project="gatr-clf-regressor",
            save_dir=output_path,
            config={
                "scheduler": scheduler_cfg,
                "args": vars(args),
                "class_weights": class_weights,
                "model_cfg": cfg_models,
                "n_bins": n_bins,
                "energy_min_gev": energy_min,
                "energy_max_gev": energy_max,
                "alpha_cls": alpha_cls,
                "alpha_lb": alpha_lb,
                "phase1_epochs": phase1_epochs,
                "use_gumbel": use_gumbel,
            },
        )
        logger.experiment.log({"model/params": sum(p.numel() for p in module.parameters())})

    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    early_stopping_patience = cfg_models.get("training", {}).get("early_stopping_patience", 0)
    if early_stopping_patience > 0:
        callbacks.append(EarlyStopping(
            monitor="loss/val",
            patience=early_stopping_patience,
            mode="min",
            verbose=True,
        ))
    if args.save_every > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_path,
                filename="gatr_clf_regressor_epoch{epoch:02d}_step{step}",
                every_n_train_steps=args.save_every,
                save_top_k=-1,
                save_weights_only=False,
            )
        )
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="gatr_clf_regressor_best_epoch{epoch:02d}_lossval{loss/val:.6f}",
            monitor="loss/val",
            mode="min",
            save_top_k=2,
            save_weights_only=False,
        )
    )

    accelerator, devices, strategy = _resolve_trainer_runtime(args)
    wprint(f"Trainer config -> accelerator={accelerator}, devices={devices}, strategy={strategy}")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        num_sanity_val_steps=1,
        sync_batchnorm=bool(isinstance(devices, int) and devices > 1 and accelerator == "gpu"),
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)

    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
