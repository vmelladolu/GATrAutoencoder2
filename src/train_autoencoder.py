import random

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from datasets import make_pf_splits
from models.gatr_autoencoder import GATrAutoencoder

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None
import yaml

from utils.data_utils import build_batch


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
            print("Gráfico guardado en gatr_autoencoder_graph.png")
        except Exception:
            print("No se pudo dibujar la red (torchviz no disponible).")

        print("Test OK: forward ejecutado en GPU")

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Entrena un autoencoder con GATr para datos de SDHCAL.")
    parser.add_argument("--data_paths", nargs="+", help="Lista de archivos .npz para entrenamiento/validación")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fracción de datos para validación")
    parser.add_argument("--mode", choices=["memory", "lazy"], default="lazy", help="Modo de carga de datos")
    parser.add_argument("--use_scalar", action="store_true", help="Si se incluye la coordenada escalar (profundidad) en el modelo")
    parser.add_argument("--use_one_hot", action="store_true", help="Si se usa one-hot encoding para las clases de thr en lugar de un solo valor escalar")
    parser.add_argument("--use_energy", action="store_true", help="Si se incluye la energía como parte de la entrada/salida (requiere modificar el modelo y los datos)")
    parser.add_argument("--z_norm", action="store_true", help="Si se aplica normalización z-score a las coordenadas espaciales y otras características usando estadísticas precomputadas")
    parser.add_argument("--test", action="store_true", help="Ejecuta un forward de prueba en GPU y dibuja la red")
    parser.add_argument("--epochs", type=int, default=1, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=3, help="Tamaño de batch para entrenamiento y validación")
    parser.add_argument("--cfg", "-c", type=str, default="config/model_cfg.yml", help="Archivo YAML con la configuración del modelo")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATrAutoencoder(cfg_enc=cfg_enc, cfg_agg=cfg_agg, cfg_dec=cfg_dec, latent_s_channels=2)
    model.to(device)

    if args.test:
        test_forward_on_gpu(model, cfg_enc, device)
        return
    data_paths = args.data_paths if args.data_paths else [
        "../data/piones_npz/Pi_30GeV_714562_715747_716264_716308.npz",
    ]
    val_ratio = args.val_ratio
    mode = args.mode
    print(f"Cargando datos desde: {data_paths} con val_ratio={val_ratio} y mode={mode}")
    train_dataset, val_dataset = make_pf_splits(
        data_paths, val_ratio=val_ratio, mode=mode
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if wandb is not None:
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
            },
        )
        wandb.watch(model, log="all", log_freq=100)
        wandb.log({"model/params": sum(p.numel() for p in model.parameters())})

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_scalar = args.use_scalar  # Si queremos que el modelo intente reconstruir la coordenada escalar (profundidad)
    use_one_hot = args.use_one_hot # Si queremos usar one-hot encoding para las clases de thr en lugar de un solo valor escalar
    use_energy = args.use_energy # Si queremos incluir la energía como parte de la entrada/salida (requiere modificar el modelo y los datos)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
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

        avg_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        first_val_batch = None
        first_val_outputs = None
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
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
        print(
            f"Epoch {epoch+1}: train_loss={avg_loss:.6f} val_loss={avg_val_loss:.6f}"
        )

        if wandb is not None:
            wandb.log({"loss/train": avg_loss, "loss/val": avg_val_loss}, step=epoch)

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
                wandb.log(wandb_log_dict, step=epoch)

                if plt is not None:
                    event_ids = batch_idx.unique().tolist()
                    if event_ids:
                        ev = random.choice(event_ids)
                        mask = batch_idx == ev
                        fig = _plot_event_projections(
                            mv_v_part[mask].numpy(), point_rec[mask].numpy()
                        )
                        if fig is not None:
                            wandb.log({"reco/projections": wandb.Image(fig)}, step=epoch)
                            plt.close(fig)


if __name__ == "__main__":
    main()
