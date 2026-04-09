import os

import torch
import yaml

from models.gatr_ar_autoencoder import GATrARAutoencoder
from utils.datasets import SDHCALRegressorDataModule
from utils.lightning_ar_trainer import LightningGATrARAutoencoder
from utils.logging import wprint

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
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


def test_forward_on_gpu(model, cfg_enc, d_step, device):
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere GPU para el modo --test")
    model = model.to(device)
    model.eval()

    B, N, n_steps = 2, 16, 4
    mv_v_part = torch.randn(B * N, 3, device=device)
    mv_s_part = torch.zeros(B * N, 1, device=device)
    scalars   = torch.randn(B * N, cfg_enc["in_s_channels"], device=device)
    batch_idx = torch.repeat_interleave(torch.arange(B, device=device), N)

    with torch.no_grad():
        out = model(mv_v_part, mv_s_part, scalars, batch_idx, n_steps=n_steps)

    wprint(f"log_n_pred shape: {out['log_n_pred'].shape}")
    wprint(f"gen_xyz shape:    {out['gen_xyz'].shape}")
    wprint(f"gen_k shape:      {out['gen_k'].shape}")
    wprint(f"gen_thr shape:    {out['gen_thr'].shape}")
    if out.get("embed_mu") is not None:
        wprint(f"embed_mu shape:   {out['embed_mu'].shape}")
    wprint("Test OK: forward ejecutado en GPU")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Training script for GATrARAutoencoder (PyTorch Lightning)"
    )
    parser.add_argument("--data_paths", nargs="+", help="Dataset paths (HDF5 or NPZ).")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--mode", choices=["memory", "lazy"], default="lazy")
    parser.add_argument("--use_scalar",  action="store_true", help="Include depth k as model input")
    parser.add_argument("--use_one_hot", action="store_true", help="One-hot encode thresholds")
    parser.add_argument("--use_time",    action="store_true", help="Include hit time")
    parser.add_argument(
        "--norm",
        choices=["z_norm", "minmax"],
        default=None,
    )
    parser.add_argument("--norm_yaml", type=str, default=None)
    parser.add_argument("--z_norm", action="store_true", help="[Alias] z_norm normalisation")
    parser.add_argument("--cfg", "-c", type=str,
                        default="config/model_cfg_ar_autoencoder.yml")
    parser.add_argument("--epochs",     type=int,   default=1)
    parser.add_argument("--batch_size", type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("-o", "--out",  type=str,   default="./trained_ar_autoencoder")
    parser.add_argument("--resume",     type=str,   default=None)
    parser.add_argument("--save-every", type=int,   default=500)
    parser.add_argument("--gpu",        type=int,   default=None)
    parser.add_argument("--gpu_ids",    nargs="+",  type=int, default=None)
    parser.add_argument("--devices",    type=int,   default=4)
    parser.add_argument("--accelerator", type=str,  default="gpu")
    parser.add_argument("--strategy",   type=str,   default="ddp")
    parser.add_argument("--train-num-workers", type=int, default=4)
    parser.add_argument("--val-num-workers",   type=int, default=2)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--test",       action="store_true",
                        help="Run a forward pass test on GPU and exit")
    return parser.parse_args()


def _resolve_trainer_runtime(args):
    """Identical to train_regressor.py helper."""
    accelerator = args.accelerator
    strategy    = args.strategy
    devices     = args.devices

    if accelerator == "gpu":
        available = torch.cuda.device_count()
        if available == 0:
            wprint("No hay GPUs disponibles. Se cambia a CPU.")
            accelerator = "cpu"
            devices     = 1
            strategy    = "auto"
        elif args.gpu_ids:
            invalid = [g for g in args.gpu_ids if g < 0 or g >= available]
            if invalid:
                raise ValueError(
                    f"GPU ids invalidas {invalid}; disponibles: 0..{available - 1}"
                )
            devices  = args.gpu_ids
            strategy = "ddp" if len(devices) > 1 else "auto"
        elif args.gpu is not None and args.devices == 1:
            devices  = [args.gpu]
            strategy = "auto"
        else:
            if isinstance(devices, int) and devices > available:
                wprint(f"Solicitadas {devices} GPUs; disponibles {available}.")
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
    cfg_enc    = cfg_models["encoder"]
    cfg_dec    = cfg_models["decoder"]
    cfg_agg    = cfg_models["aggregation"]
    cfg_n_head = cfg_models.get("n_head", {})
    cfg_vae    = cfg_models.get("vae", {"use_vae": False})
    cfg_train  = cfg_models.get("training", {})

    # Adjust in_s_channels based on flags (same logic as train_regressor.py)
    n_thr_channels = 3 if args.use_one_hot else 1
    in_s_channels  = n_thr_channels + (1 if args.use_time else 0)
    if cfg_enc["in_s_channels"] != in_s_channels:
        wprint(
            f"Ajustando encoder in_s_channels: {cfg_enc['in_s_channels']} → {in_s_channels}"
        )
        cfg_enc["in_s_channels"] = in_s_channels

    output_path    = args.out
    checkpoint_path = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Build base model for --test
    base_model = GATrARAutoencoder(
        cfg_enc=cfg_enc,
        cfg_dec=cfg_dec,
        cfg_agg=cfg_agg,
        cfg_n_head=cfg_n_head,
        d=cfg_train.get("d_step", 4),
        use_vae=cfg_vae.get("use_vae", False),
        event_embed_dim=cfg_vae.get("event_embed_dim", 32),
    )

    if args.test:
        device = (
            torch.device(f"cuda:{args.gpu}")
            if args.gpu is not None and torch.cuda.is_available()
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        test_forward_on_gpu(base_model, cfg_enc, cfg_train.get("d_step", 4), device)
        return

    data_paths = args.data_paths or [
        "/nfs/cms/arqolmo/SDHCAL_Energy/GATrAutoencoder/flat_all.npz",
    ]
    norm_type = args.norm or ("z_norm" if args.z_norm else None)
    preprocessing_cfg = {
        "use_scalar":     args.use_scalar,
        "use_one_hot":    args.use_one_hot,
        "use_time":       args.use_time,
        "use_energy":     False,   # AR autoencoder does not use energy
        "use_log":        False,
        "norm_type":      norm_type,
        "norm_yaml_path": args.norm_yaml,
        "z_norm":         args.z_norm,
    }
    filters_cfg = cfg_models.get("filters", {})
    wprint(f"Cargando datos desde: {data_paths}")

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
        use_weighted_loss=False,   # not used here
    )
    datamodule.setup()

    scheduler_cfg  = cfg_models.get("scheduler",  {"type": "cosine"})
    optimizer_cfg  = cfg_models.get("optimizer",  {"weight_decay": 1e-4})
    debug_cfg      = cfg_models.get("debug",      {})
    hungarian_weights = cfg_train.get("hungarian_weights", {})

    module = LightningGATrARAutoencoder(
        cfg_enc=cfg_enc,
        cfg_dec=cfg_dec,
        cfg_agg=cfg_agg,
        cfg_n_head=cfg_n_head,
        cfg_vae=cfg_vae,
        pretrain_n_epochs=cfg_train.get("pretrain_n_epochs", 10),
        lambda_hungarian=cfg_train.get("lambda_hungarian", 1.0),
        lambda_n=cfg_train.get("lambda_n", 0.1),
        d_step=cfg_train.get("d_step", 4),
        scheduled_sampling_T=cfg_train.get("scheduled_sampling_T", 0),
        hungarian_weights=hungarian_weights,
        use_scalar=args.use_scalar,
        use_one_hot=args.use_one_hot,
        use_time=args.use_time,
        learning_rate=optimizer_cfg.get("lr", args.lr),
        max_epochs=args.epochs,
        output_path=output_path,
        scheduler_cfg=scheduler_cfg,
        optimizer_cfg=optimizer_cfg,
        debug_cfg=debug_cfg,
    )

    logger = None
    if wandb is not None:
        logger = WandbLogger(
            project="gatr-ar-autoencoder",
            save_dir=output_path,
            config={
                "scheduler":  scheduler_cfg,
                "args":       vars(args),
                "model_cfg":  cfg_models,
            },
        )
        logger.experiment.log(
            {"model/params": sum(p.numel() for p in module.parameters())}
        )

    callbacks = [LearningRateMonitor(logging_interval="step")]
    if args.save_every > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_path,
                filename="gatr_ar_epoch{epoch:02d}_step{step}",
                every_n_train_steps=args.save_every,
                save_top_k=-1,
                save_weights_only=False,
            )
        )
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="gatr_ar_best_epoch{epoch:02d}_lossval{loss/val:.6f}",
            monitor="loss/val",
            mode="min",
            save_top_k=2,
            save_weights_only=False,
        )
    )

    accelerator, devices, strategy = _resolve_trainer_runtime(args)
    wprint(f"Trainer: accelerator={accelerator}, devices={devices}, strategy={strategy}")

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
        sync_batchnorm=bool(
            isinstance(devices, int) and devices > 1 and accelerator == "gpu"
        ),
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)

    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
