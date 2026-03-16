import os
import random
import time

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from utils.datasets import make_pf_splits
from models.gatr_autoencoder import GATrAutoencoder


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None
import yaml

from utils.batch_utils import build_batch


def reconstruction_loss(outputs, mv_v_part, mv_s_part, scalars, use_scalar=False):
        point_rec = outputs["point_rec"]
        scalar_rec = outputs["scalar_rec"]
        s_rec = outputs["s_rec"]

        loss_xyz = nn.functional.mse_loss(point_rec, mv_v_part)
        if use_scalar:
            loss_depth = nn.functional.mse_loss(scalar_rec, mv_s_part)
        else:
            loss_depth = 0.0
        
        scalar_loss = nn.functional.mse_loss(s_rec, scalars)

        return loss_xyz + loss_depth + scalar_loss


def _plot_event_projections(xyz, xyz_rec):
        if plt is None:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].scatter(xyz[:, 0], xyz[:, 2], s=3, alpha=0.6, label="orig")
        axes[0].scatter(xyz_rec[:, 0], xyz_rec[:, 2], s=3, alpha=0.6, label="rec")
        axes[0].set_title("XZ")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("z")
        axes[0].legend()

        axes[1].scatter(xyz[:, 1], xyz[:, 2], s=3, alpha=0.6, label="orig")
        axes[1].scatter(xyz_rec[:, 1], xyz_rec[:, 2], s=3, alpha=0.6, label="rec")
        axes[1].set_title("YZ")
        axes[1].set_xlabel("y")
        axes[1].set_ylabel("z")
        axes[1].legend()

        fig.tight_layout()
        return fig


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
            outputs = model(mv_v_part, mv_s_part, scalars, batch_idx)

        try:
            from torchviz import make_dot

            dot = make_dot(outputs["point_rec"].sum(), params=dict(model.named_parameters()))
            dot.format = "png"
            dot.render("gatr_autoencoder_graph", cleanup=True)
            print("Gráfico guardado en gatr_autoencoder_graph.png")
        except Exception:
            print("No se pudo dibujar la red (torchviz no disponible).")

        print("Test OK: forward ejecutado en GPU")


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, val_loss, args):
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["global_step"], ckpt.get("val_loss", float("inf"))


class TopKCheckpointManager:
    """Mantiene los top-k checkpoints por val_loss (menor es mejor)."""

    def __init__(self, ckpt_dir, top_k=2):
        self.ckpt_dir = ckpt_dir
        self.top_k = top_k
        self._entries = []  # lista de (val_loss, path)

    def update(self, val_loss, model, optimizer, scheduler, epoch, global_step, args):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f"checkpoint_best_epoch{epoch:04d}_loss{val_loss:.6f}.pt")
        save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, val_loss, args)
        self._entries.append((val_loss, path))
        # Ordenar por loss ascendente y eliminar los peores
        self._entries.sort(key=lambda x: x[0])
        while len(self._entries) > self.top_k:
            _, old_path = self._entries.pop()   # el peor (mayor loss)
            if os.path.exists(old_path):
                os.remove(old_path)
            _log(f"  [ckpt] Eliminado checkpoint antiguo: {os.path.basename(old_path)}")
        _log(f"  [ckpt] Top-{self.top_k} guardados: {[os.path.basename(e[1]) for e in self._entries]}")


def build_scheduler(optimizer, args, total_steps_per_epoch):
    """Construye el scheduler según los argumentos CLI.

    Soporta:
    - interval: 'epoch' o 'step'
    - scheduler_type: 'cosine' o 'step'
    - warmup opcional (warmup_pct > 0)
    """
    interval = args.sched_interval
    scheduler_type = args.sched_type
    warmup_pct = args.warmup_pct
    warmup_start_factor = args.warmup_start_factor

    if interval == "step":
        total_steps = args.epochs * total_steps_per_epoch
        warmup_steps = int(total_steps * warmup_pct)
        if scheduler_type == "cosine":
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=args.lr_min,
            )
        else:
            main_sched = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.decay_steps,
                gamma=args.decay_rate,
            )
        if warmup_steps > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[warmup_steps],
            )
        else:
            sched = main_sched
    else:
        warmup_epochs = int(args.epochs * warmup_pct)
        if scheduler_type == "cosine":
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - warmup_epochs),
                eta_min=args.lr_min,
            )
        else:
            main_sched = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.decay_steps,
                gamma=args.decay_rate,
            )
        if warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[warmup_epochs],
            )
        else:
            sched = main_sched

    return sched, interval


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Entrena un autoencoder con GATr para datos de SDHCAL.")
    parser.add_argument("--data_paths", nargs="+", help="Lista de archivos .npz para entrenamiento/validación")
    parser.add_argument("--val_ratio", type=float, help="Fracción de datos para validación")
    parser.add_argument("--mode", choices=["memory", "lazy"], help="Modo de carga de datos")
    parser.add_argument("--use_scalar", action="store_true", help="Si se incluye la coordenada escalar (profundidad) en el modelo")
    parser.add_argument("--use_one_hot", action="store_true", help="Si se usa one-hot encoding para las clases de thr en lugar de un solo valor escalar")
    parser.add_argument("--use_energy", action="store_true", help="Si se incluye la energía como parte de la entrada/salida (requiere modificar el modelo y los datos)")
    parser.add_argument("--z_norm", action="store_true", help="Si se aplica normalización z-score a las coordenadas espaciales y otras características usando estadísticas precomputadas")
    parser.add_argument("--test", action="store_true", help="Ejecuta un forward de prueba en GPU y dibuja la red")
    parser.add_argument("--epochs", type=int, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, help="Tamaño de batch para entrenamiento y validación")
    parser.add_argument("--cfg", "-c", type=str, default="config/model_cfg.yml", help="Archivo YAML con la configuración del modelo")
    # Optimizer
    parser.add_argument("--lr", type=float, help="Learning rate inicial")
    parser.add_argument("--weight_decay", type=float, help="Weight decay del optimizador AdamW")
    # Scheduler
    parser.add_argument("--sched_type", choices=["cosine", "step", "none"], help="Tipo de scheduler de LR")
    parser.add_argument("--sched_interval", choices=["epoch", "step"], help="Intervalo de actualización del scheduler")
    parser.add_argument("--warmup_pct", type=float, help="Fracción del total de steps/épocas dedicada al warmup (0 = sin warmup)")
    parser.add_argument("--warmup_start_factor", type=float, help="Factor de LR inicial del warmup (LR_inicio = lr * factor)")
    parser.add_argument("--lr_min", type=float, help="LR mínimo para CosineAnnealingLR")
    parser.add_argument("--decay_steps", type=int, help="step_size de StepLR (en épocas o steps según --sched_interval)")
    parser.add_argument("--decay_rate", type=float, help="Gamma de StepLR")
    parser.add_argument("--plot_every", type=int, help="Genera plots de reconstrucción cada N batches de train (0 = desactivado)")
    # Checkpoints
    parser.add_argument("--checkpoint_dir", type=str, help="Directorio donde guardar los checkpoints")
    parser.add_argument("--ckpt_save_every", type=int, help="Guarda checkpoint_latest cada N batches (0 = desactivado)")
    parser.add_argument("--resume", type=str, default=None, help="Ruta a un checkpoint desde el que reanudar el entrenamiento")

    # Paso 1: parsear solo --cfg para saber qué YAML cargar
    pre_args, _ = parser.parse_known_args()
    cfg_path = pre_args.cfg

    # Paso 2: cargar defaults de la sección 'training' del YAML
    yaml_training = {}
    try:
        with open(cfg_path, "r") as f:
            yaml_all = yaml.safe_load(f)
        yaml_training = yaml_all.get("training", {})
    except FileNotFoundError:
        pass  # sin YAML, se usan los hardcoded defaults de abajo

    # Hardcoded defaults (mínima prioridad)
    hardcoded = {
        "val_ratio": 0.2, "mode": "lazy", "epochs": 1, "batch_size": 3,
        "lr": 1e-3, "weight_decay": 1e-4,
        "sched_type": "none", "sched_interval": "epoch",
        "warmup_pct": 0.0, "warmup_start_factor": 0.1,
        "lr_min": 1e-6, "decay_steps": 30, "decay_rate": 0.5,
        "plot_every": 0,
        "checkpoint_dir": "checkpoints",
        "ckpt_save_every": 50,
    }
    hardcoded.update(yaml_training)   # YAML sobreescribe hardcoded
    parser.set_defaults(**hardcoded)  # CLI sobreescribirá estos defaults

    return parser.parse_args()

def main():
    args = parse_args()
    
    cfg_models = yaml.safe_load(open(args.cfg, "r"))
    cfg_enc = cfg_models["encoder"]
    cfg_dec = cfg_models["decoder"] 
    cfg_agg = cfg_models["aggregation"]

    # Ensure output dec is compatible with encoder output for reconstruction
    if cfg_dec["out_s_channels"] != cfg_enc["in_s_channels"]:
        print(f"Warning: Adjusting decoder out_s_channels from {cfg_dec['out_s_channels']} to match encoder in_s_channels {cfg_enc['in_s_channels']} for reconstruction.")
        cfg_dec["out_s_channels"] = cfg_enc["in_s_channels"]

    _log(f"Dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log("Construyendo modelo...")
    model = GATrAutoencoder(cfg_enc=cfg_enc, cfg_agg=cfg_agg, cfg_dec=cfg_dec, latent_s_channels=2)
    model.to(device)
    _log(f"Modelo listo ({sum(p.numel() for p in model.parameters()):,} parámetros)")

    if args.test:
        test_forward_on_gpu(model, cfg_enc, device)
        return
    data_paths = args.data_paths if args.data_paths else [
        "../data/Datos/El_50GeV.npz_flat.npz",
    ]
    val_ratio = args.val_ratio
    mode = args.mode
    print(f"Cargando datos desde: {data_paths} con val_ratio={val_ratio} y mode={mode}")
    t0 = time.time()
    train_dataset, val_dataset = make_pf_splits(
        data_paths, val_ratio=val_ratio, mode=mode
    )
    _log(f"Datasets listos en {time.time()-t0:.1f}s — train={len(train_dataset)}, val={len(val_dataset)}")

    _log("Construyendo DataLoaders...")
    t0 = time.time()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    _log(f"DataLoaders listos en {time.time()-t0:.1f}s — {len(train_loader)} train batches, {len(val_loader)} val batches")

    if wandb is not None:
        _log("Iniciando wandb.init()...")
        t0 = time.time()
        wandb.init(
            project="gatr-autoencoder",
            config={
                "cfg_enc": cfg_enc,
                "cfg_dec": cfg_dec,
                "cfg_agg": cfg_agg,
                "latent_s_channels": 2,
                "batch_size": args.batch_size,
                "val_ratio": val_ratio,
                "mode": mode,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "sched_type": args.sched_type,
                "sched_interval": args.sched_interval,
                "warmup_pct": args.warmup_pct,
                "warmup_start_factor": args.warmup_start_factor,
                "lr_min": args.lr_min,
                "decay_steps": args.decay_steps,
                "decay_rate": args.decay_rate,
            },
        )
        _log(f"wandb.init() completado en {time.time()-t0:.1f}s")
        _log("Llamando wandb.watch()...")
        t0 = time.time()
        wandb.watch(model, log="all", log_freq=100)
        _log(f"wandb.watch() completado en {time.time()-t0:.1f}s")
        wandb.define_metric("global_step")
        wandb.define_metric("loss/train_batch", step_metric="global_step")
        wandb.define_metric("train/reco_projections", step_metric="global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("loss/train", step_metric="epoch")
        wandb.define_metric("loss/val", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("relative_err/*", step_metric="epoch")
        wandb.define_metric("reco/*", step_metric="epoch")
        wandb.log({"model/params": sum(p.numel() for p in model.parameters())})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_scalar = args.use_scalar  # Si queremos que el modelo intente reconstruir la coordenada escalar (profundidad)
    use_one_hot = args.use_one_hot # Si queremos usar one-hot encoding para las clases de thr en lugar de un solo valor escalar
    use_energy = args.use_energy # Si queremos incluir la energía como parte de la entrada/salida (requiere modificar el modelo y los datos)

    scheduler, sched_interval = None, None
    if args.sched_type != "none":
        scheduler, sched_interval = build_scheduler(optimizer, args, len(train_loader))
        _log(f"Scheduler: type={args.sched_type}, interval={args.sched_interval}, warmup_pct={args.warmup_pct}")

    # Checkpoint manager (top-2 por val loss)
    ckpt_manager = TopKCheckpointManager(ckpt_dir=args.checkpoint_dir, top_k=2)
    ckpt_latest_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")

    # Reanudar desde checkpoint si se especifica
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        _log(f"Reanudando desde checkpoint: {args.resume}")
        start_epoch, global_step, _ = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch += 1  # continuamos en la siguiente época
        _log(f"  Reanudado en epoch={start_epoch}, global_step={global_step}")

    for epoch in range(start_epoch, args.epochs):
        _log(f"=== Epoch {epoch+1}/{args.epochs} — iniciando entrenamiento ===")
        model.train()
        total_loss = 0.0
        t_epoch = time.time()
        for batch_num, batch in enumerate(train_loader):
            if batch_num == 0:
                _log(f"  Primer batch recibido (batch_num=0)")
            if batch_num < 3 or batch_num % 50 == 0:
                _log(f"  [train] batch {batch_num+1}/{len(train_loader)}")
            data = build_batch(batch, use_scalar=use_scalar, use_one_hot=use_one_hot, use_energy=use_energy, z_norm=args.z_norm)
            mv_v_part = data["mv_v_part"].to(device)
            mv_s_part = data["mv_s_part"].to(device)
            scalars = data["scalars"].to(device)
            batch_idx = data["batch_idx"].to(device)

            outputs = model(mv_v_part, mv_s_part, scalars, batch_idx)
            loss = reconstruction_loss(outputs, mv_v_part,
                                       mv_s_part, scalars,
                                       use_scalar=use_scalar,
                                       )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if scheduler is not None and sched_interval == "step":
                scheduler.step()

            if args.ckpt_save_every > 0 and global_step % args.ckpt_save_every == 0:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                save_checkpoint(ckpt_latest_path, model, optimizer, scheduler, epoch, global_step, float("inf"), args)
                _log(f"  [ckpt] checkpoint_latest guardado (step={global_step})")

            if wandb is not None:
                log_dict = {"loss/train_batch": loss.item(), "global_step": global_step}
                if plt is not None and args.plot_every > 0 and global_step % args.plot_every == 0:
                    point_rec_cpu = outputs["point_rec"].detach().cpu()
                    mv_v_cpu = mv_v_part.detach().cpu()
                    batch_idx_cpu = batch_idx.detach().cpu()
                    ev = random.choice(batch_idx_cpu.unique().tolist())
                    mask = batch_idx_cpu == ev
                    fig = _plot_event_projections(mv_v_cpu[mask].numpy(), point_rec_cpu[mask].numpy())
                    if fig is not None:
                        log_dict["train/reco_projections"] = wandb.Image(fig)
                        plt.close(fig)
                wandb.log(log_dict)
            global_step += 1

        if scheduler is not None and sched_interval == "epoch":
            scheduler.step()

        avg_loss = total_loss / max(1, len(train_loader))
        current_lr = optimizer.param_groups[0]["lr"]
        _log(f"  Entrenamiento epoch {epoch+1} completado en {time.time()-t_epoch:.1f}s — avg_train_loss={avg_loss:.6f} — lr={current_lr:.2e}")

        _log(f"  Iniciando validación epoch {epoch+1}...")
        model.eval()
        val_loss = 0.0
        first_val_batch = None
        first_val_outputs = None
        t_val = time.time()
        with torch.no_grad():
            for val_batch_num, batch in enumerate(val_loader):
                if val_batch_num == 0:
                    _log(f"  Primer val batch recibido")
                data = build_batch(batch, use_scalar=use_scalar, use_one_hot=use_one_hot, use_energy=use_energy, z_norm=args.z_norm)
                mv_v_part = data["mv_v_part"].to(device)
                mv_s_part = data["mv_s_part"].to(device)
                scalars = data["scalars"].to(device)
                batch_idx = data["batch_idx"].to(device)

                outputs = model(mv_v_part, mv_s_part, scalars, batch_idx)
                loss = reconstruction_loss(outputs,
                                           mv_v_part,
                                           mv_s_part,
                                           scalars,
                                           use_scalar=use_scalar,
                                           )

                val_loss += loss.item()

                if first_val_batch is None:
                    first_val_batch = {
                        "mv_v_part": mv_v_part.detach().cpu(),
                        "mv_s_part": mv_s_part.detach().cpu(),
                        "scalars": scalars.detach().cpu(),
                        "batch_idx": batch_idx.detach().cpu(),
                    }
                    first_val_outputs = {
                        "point_rec": outputs["point_rec"].detach().cpu(),
                        "scalar_rec": outputs["scalar_rec"].detach().cpu(),
                        "s_rec": outputs["s_rec"].detach().cpu(),
                    }

        avg_val_loss = val_loss / max(1, len(val_loader))
        _log(f"  Validación epoch {epoch+1} completada en {time.time()-t_val:.1f}s — avg_val_loss={avg_val_loss:.6f}")
        print(
            f"Epoch {epoch+1}: train_loss={avg_loss:.6f} val_loss={avg_val_loss:.6f}"
        )
        ckpt_manager.update(avg_val_loss, model, optimizer, scheduler, epoch, global_step, args)

        if wandb is not None:
            wandb.log({"loss/train": avg_loss, "loss/val": avg_val_loss, "lr": current_lr, "epoch": epoch})

            if first_val_batch is not None and first_val_outputs is not None:
                mv_v_part = first_val_batch["mv_v_part"]
                mv_s_part = first_val_batch["mv_s_part"]
                scalars = first_val_batch["scalars"]
                batch_idx = first_val_batch["batch_idx"]

                point_rec = first_val_outputs["point_rec"]
                scalar_rec = first_val_outputs["scalar_rec"]
                s_rec = first_val_outputs["s_rec"]
                # Calculamos el error como reco-true/true
                r_err_x = ((point_rec[:, 0]- mv_v_part[:,0])/mv_v_part[:,0].abs().clamp(min=1e-6)).numpy()
                r_err_y = ((point_rec[:, 1]- mv_v_part[:,1])/mv_v_part[:,1].abs().clamp(min=1e-6)).numpy()
                r_err_z = ((point_rec[:, 2]- mv_v_part[:,2])/mv_v_part[:,2].abs().clamp(min=1e-6)).numpy()
                r_err_depth = ((scalar_rec[:, 0]- mv_s_part[:,0])/mv_s_part[:,0].abs().clamp(min=1e-6)).numpy()
                r_err_thr = ((s_rec - scalars)/scalars.abs().clamp(min=1e-6)).numpy().reshape(-1)
                wandb_log_dict = {
                    "loss/train": avg_loss,
                    "loss/val": avg_val_loss,
                    "relative_err/x": wandb.Histogram(r_err_x),
                    "relative_err/y": wandb.Histogram(r_err_y),
                    "relative_err/z": wandb.Histogram(r_err_z),
                    "relative_err/depth": wandb.Histogram(r_err_depth),
                    "relative_err/thr": wandb.Histogram(r_err_thr),
                }
                if not use_scalar:
                    del wandb_log_dict["relative_err/depth"]
                wandb_log_dict["epoch"] = epoch
                wandb.log(wandb_log_dict)

                if plt is not None:
                    event_ids = batch_idx.unique().tolist()
                    if event_ids:
                        ev = random.choice(event_ids)
                        mask = batch_idx == ev
                        fig = _plot_event_projections(
                            mv_v_part[mask].numpy(), point_rec[mask].numpy()
                        )
                        if fig is not None:
                            wandb.log({"reco/projections": wandb.Image(fig), "epoch": epoch})
                            plt.close(fig)


if __name__ == "__main__":
    main()
