import random

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from scipy.stats import norm
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datasets import make_pf_splits
from models.gatr_regressor import GATrRegressor
from utils.batch_utils import build_batch


# ============================================================
# Funciones de debug para wandb
# ============================================================

def _log_gradient_stats(model, step):
    """Loguea norma de gradientes por capa y global para detectar vanishing/exploding."""
    if wandb is None:
        return
    total_norm = 0.0
    grad_log = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            # Agrupar por módulo principal para no saturar wandb
            short_name = name.replace(".", "/")
            grad_log[f"grads/{short_name}"] = param_norm
    total_norm = total_norm ** 0.5
    grad_log["grads/total_norm"] = total_norm
    wandb.log(grad_log, step=step)


def _log_prediction_debug(outputs, target, step):
    """Loguea estadísticas de predicciones vs target para detectar colapso a media."""
    if wandb is None:
        return
    pred = outputs.detach().cpu()
    tgt = target.detach().cpu().view(-1)
    wandb.log({
        "debug/pred_mean": pred.mean().item(),
        "debug/pred_std": pred.std().item(),
        "debug/pred_min": pred.min().item(),
        "debug/pred_max": pred.max().item(),
        "debug/target_mean": tgt.mean().item(),
        "debug/target_std": tgt.std().item(),
        "debug/target_min": tgt.min().item(),
        "debug/target_max": tgt.max().item(),
        "debug/pred_target_corr": float(torch.corrcoef(torch.stack([pred.view(-1), tgt]))[0, 1]),
    }, step=step)


def _log_aggregation_debug(model, mv_v_part, mv_s_part, scalars, batch_idx, step):
    """Loguea estadísticas del latent space post-agregación."""
    if wandb is None:
        return
    with torch.no_grad():
        mv_latent, s_latent, _, _ = model.encode(mv_v_part, mv_s_part, scalars, batch_idx)
        from torch_scatter import scatter_mean as _sm
        mv_agg = _sm(mv_latent.squeeze(1), batch_idx, dim=0)
        s_agg = _sm(s_latent, batch_idx, dim=0)
        agg = torch.cat([mv_agg, s_agg], dim=-1)
        wandb.log({
            "debug/latent_agg_mean": agg.mean().item(),
            "debug/latent_agg_std": agg.std().item(),
            "debug/latent_agg_max": agg.max().item(),
            "debug/latent_agg_min": agg.min().item(),
            "debug/latent_mv_std": mv_agg.std().item(),
            "debug/latent_s_std": s_agg.std().item(),
            "debug/latent_agg_dim_std_mean": agg.std(dim=0).mean().item(),
            # Cuántas dimensiones están "muertas" (std < 1e-5)
            "debug/latent_dead_dims": int((agg.std(dim=0) < 1e-5).sum().item()),
        }, step=step)


def _log_event_display(mv_v_part, batch_idx, outputs, logenergy, use_log, step):
    """Plotea un evento aleatorio del batch: proyecciones XY e YZ con E_true y E_pred."""
    if plt is None or wandb is None:
        return
    pos = mv_v_part.detach().cpu()
    bidx = batch_idx.detach().cpu()
    pred = outputs.detach().cpu()
    tgt = logenergy.detach().cpu().view(-1)

    # Elegir un evento aleatorio del batch
    unique_events = bidx.unique()
    evt = unique_events[torch.randint(len(unique_events), (1,)).item()].item()
    mask = bidx == evt

    x = pos[mask, 0].numpy()
    y = pos[mask, 1].numpy()
    z = pos[mask, 2].numpy()

    e_pred_val = float(pred[evt])
    e_true_val = float(tgt[evt])

    # Convertir a energía lineal si se usó log
    if use_log:
        e_pred_lin = np.exp(e_pred_val)
        e_true_lin = np.exp(e_true_val)
    else:
        e_pred_lin = e_pred_val
        e_true_lin = e_true_val

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Evento (nhits={mask.sum().item()})  |  "
        f"$E_{{true}}$ = {e_true_lin:.2f} GeV   $E_{{pred}}$ = {e_pred_lin:.2f} GeV",
        fontsize=13,
    )

    # Proyección XY
    ax1.scatter(x, y, s=4, alpha=0.7, c="steelblue")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Proyección XY")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Proyección YZ
    ax2.scatter(y, z, s=4, alpha=0.7, c="darkorange")
    ax2.set_xlabel("Y")
    ax2.set_ylabel("Z")
    ax2.set_title("Proyección YZ")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    wandb.log({"debug/event_display": wandb.Image(fig)}, step=step)
    plt.close(fig)


try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None
import yaml

z_norm_yaml_path = "/nfs/cms/arqolmo/SDHCAL_Energy/GATrAutoencoder/stats_z_norm.yaml"  # Ruta al archivo YAML con las estadísticas para normalización z-score

def reconstruction_loss(outputs, logenergy):
        E_pred = outputs
        # (E_true - E_pred)**2 / E_true
        # E_pred is (N). Ensure logenergy is also (N) and not (N, 1)
        logenergy = logenergy.view(-1)
        # loss = torch.mean((logenergy - E_pred.squeeze())**2 / (logenergy.abs() + 1e-6))
        # MSE loss
        loss = nn.MSELoss()(E_pred.squeeze(), logenergy) 
        return loss


def _log_regression_plots(E_true, E_reco, step=None):
    """Crea y sube a wandb los gráficos de regresión de energía (mismo estilo que MLPFit)."""
    if plt is None or wandb is None:
        return

    import pandas as pd
    df_plot = pd.DataFrame({"E_true": E_true, "E_reco": E_reco})

    # Agrupar por energía discreta o por bins
    if df_plot["E_true"].nunique() < 30:
        grouped = df_plot.groupby("E_true")["E_reco"]
        E_true_vals = grouped.mean().index.to_numpy()
        E_reco_mean = grouped.mean().to_numpy()
        E_reco_std = grouped.std().to_numpy()
    else:
        n_bins = 30
        bins = np.linspace(df_plot["E_true"].min(), df_plot["E_true"].max(), n_bins + 1)
        df_plot["E_bin"] = np.digitize(df_plot["E_true"], bins)
        grouped = df_plot.groupby("E_bin")
        E_true_vals = grouped["E_true"].mean().to_numpy()
        E_reco_mean = grouped["E_reco"].mean().to_numpy()
        E_reco_std = grouped["E_reco"].std().to_numpy()

    # --- Plot 1: E_reco vs E_true con banda de error ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(df_plot["E_true"], df_plot["E_reco"], alpha=0.9, s=8, color="gray",
                label="Individual Predictions")
    ax1.plot(E_true_vals, E_reco_mean, color="blue", label=r"$E_{reco}$ Mean")
    ax1.fill_between(E_true_vals, E_reco_mean - E_reco_std, E_reco_mean + E_reco_std,
                     color="lightblue", alpha=0.4, label=r"±1$\sigma$")
    min_E, max_E = df_plot["E_true"].min(), df_plot["E_true"].max()
    ax1.plot([min_E, max_E], [min_E, max_E], color="red", linestyle="--",
             label=r"$E_{reco} = E_{true}$")
    ax1.set_xlabel(r"$E_{true}$ [GeV]")
    ax1.set_ylabel(r"$E_{reco}$ [GeV]")
    ax1.set_title(r"$E_{reco}$ vs $E_{true}$ (GATr Regressor)")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.7)
    fig1.tight_layout()

    # --- Plot 2: Distribución del error relativo ---
    rel_error = (df_plot["E_reco"] - df_plot["E_true"]) / df_plot["E_true"]
    mean_err = float(np.mean(rel_error))
    std_err = float(np.std(rel_error))

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(rel_error, bins=50, color="steelblue", alpha=0.75, edgecolor="black",
             label=r"$(E_{reco} - E_{true}) / E_{true}$")
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Error = 0")
    ax2.axvline(mean_err, color="green", linestyle="-.", linewidth=1.2,
                label=f"Media = {mean_err:.3f}")
    ax2.axvline(mean_err + std_err, color="gray", linestyle=":", linewidth=1,
                label=f"±1σ = {std_err:.3f}")
    ax2.axvline(mean_err - std_err, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel(r"Error relativo $(E_{reco} - E_{true}) / E_{true}$")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title("Distribución del error relativo (GATr Regressor)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)
    fig2.tight_layout()

    # --- Plot 3: Resolución vs 1/sqrt(E) ---
    mean_vals, std_vals = [], []
    if df_plot["E_true"].nunique() < 30:
        grouped_res = df_plot.groupby("E_true")
        E_true_res = []
        n_energy_values = len(grouped_res)
        nrows = int(np.ceil(n_energy_values / 3))
        ncols = min(3, n_energy_values)
        fig_dist = plt.figure(figsize=(ncols * 5, nrows * 4))
        for i, (E_val, group) in enumerate(grouped_res):
            plt.subplot(nrows, ncols, i + 1) 
            E_true_res.append(E_val)
            rel_res = (group["E_reco"] - E_val) / E_val
            mu, sigma = norm.fit(rel_res)
            plt.hist(rel_res, bins=20, alpha=0.6, density=True)
            x = np.linspace(rel_res.min(), rel_res.max(), 200)
            plt.plot(x, norm.pdf(x, mu, sigma), 'r-')
            plt.title(f"E_true = {E_val:.2f}")
            plt.grid(True, linestyle=":", alpha=0.7)
            plt.xlabel("Relative Residuals")
            plt.ylabel("Count")
            mean_vals.append(mu)
            std_vals.append(sigma)
        fig_dist.tight_layout()
        E_true_res = np.array(E_true_res)
    else:
        fig_dist = None
        n_bins_res = 30
        bins_res = np.linspace(df_plot["E_true"].min(), df_plot["E_true"].max(), n_bins_res + 1)
        df_plot["E_bin_res"] = np.digitize(df_plot["E_true"], bins_res)
        grouped_res = df_plot.groupby("E_bin_res")
        E_true_res = []
        for _, group in grouped_res:
            E_true_res.append(group["E_true"].mean())
            rel_res = (group["E_reco"] - group["E_true"]) / group["E_true"]
            mu, sigma = norm.fit(rel_res)
            mean_vals.append(mu)
            std_vals.append(sigma)
        E_true_res = np.array(E_true_res)

    mean_vals = np.array(mean_vals)
    std_vals = np.array(std_vals)
    ratio = std_vals / np.abs(mean_vals).clip(min=1e-6)
    # inv_sqrt_E = 1.0 / np.sqrt(E_true_res)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.scatter(E_true_res, ratio, s=50, color="purple", alpha=0.8,
                label=r"$\sigma(E)/\mu(E)$")
    ax3.plot(E_true_res, ratio, color="purple", alpha=0.6)
    ax3.set_xlabel(r"$E_{true}$")
    ax3.set_ylabel(r"$\sigma(E_{reco}) / \mu(E_{reco})$")
    ax3.set_title(r"Resolution vs E$ (GATr Regressor)")
    ax3.grid(True, linestyle=":", alpha=0.7)
    ax3.legend()
    fig3.tight_layout()

    # --- Plot 4: Boxplot de E_reco por cada E_true ---
    unique_energies = sorted(df_plot["E_true"].unique())
    box_data = [df_plot.loc[df_plot["E_true"] == e, "E_reco"].values for e in unique_energies]

    fig4, ax4 = plt.subplots(figsize=(max(8, len(unique_energies) * 0.8), 6))
    bp = ax4.boxplot(box_data, positions=range(len(unique_energies)), widths=0.6,
                     patch_artist=True, showfliers=True,
                     flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax4.set_xticks(range(len(unique_energies)))
    ax4.set_xticklabels([f"{e:.0f}" for e in unique_energies], rotation=45)
    ax4.plot(range(len(unique_energies)), unique_energies, "r--",
             label=r"$E_{reco} = E_{true}$")
    ax4.set_xlabel(r"$E_{true}$ [GeV]")
    ax4.set_ylabel(r"$E_{reco}$ [GeV]")
    ax4.set_title(r"Distribution of $E_{reco}$ per $E_{true}$ (GATr Regressor)")
    ax4.legend()
    ax4.grid(True, linestyle=":", alpha=0.7)
    fig4.tight_layout()

    # Subir a wandb
    log_dict = {
        "plots/E_reco_vs_E_true": wandb.Image(fig1),
        "plots/relative_error_hist": wandb.Image(fig2),
        "plots/resolution_vs_inv_sqrt_E": wandb.Image(fig3),
        "plots/boxplot_E_reco_per_E_true": wandb.Image(fig4),
        "plots/relative_residuals_distribution": wandb.Image(fig_dist),
        "metrics/rel_error_mean": mean_err,
        "metrics/rel_error_std": std_err,
        "metrics/MAE_rel": float(np.mean(np.abs(rel_error))),
    }
    wandb.log(log_dict, step=step)

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig_dist)


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

            dot = make_dot(outputs.sum(), params=dict(model.named_parameters()))
            dot.format = "png"
            dot.render("gatr_autoencoder_graph", cleanup=True)
            print("Gráfico guardado en gatr_autoencoder_graph.png")
        except Exception:
            print("No se pudo dibujar la red (torchviz no disponible).")

        print("Test OK: forward ejecutado en GPU")

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Training script for GATrRegressor")
    parser.add_argument("--data_paths", nargs="+", help="File list of dataset paths (HDF5 or NPZ).")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio (0-1)")
    parser.add_argument("--mode", choices=["memory", "lazy"], default="lazy", help="Data loading mode: memory for loading all data into RAM, lazy for on-the-fly loading (recomendado para datasets grandes)")
    parser.add_argument("--use_scalar", action="store_true", help="Wheter to include scalar features (like layer number) as input to the model")
    parser.add_argument("--use_one_hot", action="store_true", help="Wheter to include one-hot encoding of categorical features (like particle type) as input to the model")
    parser.add_argument("--use_energy", action="store_true", help="Wheter to use energy as objective (always True for regressor, but kept for consistency with autoencoder script)")
    parser.add_argument("--z_norm", action="store_true", help="Whether to apply z-score normalization to the input features using precomputed stats from a YAML file")
    parser.add_argument("--test", action="store_true", help="Run a forward pass test on GPU and save the model graph, then exit")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size for training and validation")
    parser.add_argument("--use_log", action="store_true", help="Wheter to use log(E) instead of E for regression")
    parser.add_argument("--plot_every", type=int, default=0, help="Whether to plot regression graphs every N epochs (0 = only at the end)")
    parser.add_argument("--cfg", "-c", type=str, default="config/model_cfg_regressor.yml", help="YAML config file for model architecture and training hyperparameters")
    parser.add_argument("--save-every", type=int, default=10000, help="Save a checkpoint every N training steps (0 to disable)")
    parser.add_argument("--o", "--out", type=str, default="./trained_model")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    cfg_models = yaml.safe_load(open(args.cfg, "r"))
    cfg_enc = cfg_models["encoder"]
    cfg_dec = cfg_models["decoder"] 
    cfg_agg = cfg_models["aggregation"]
    
    output_path = args.o
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    checkpoint_path = os.path.join(output_path, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    # Ensure output dec is compatible with encoder output for reconstruction
    if cfg_dec["out_s_channels"] != cfg_enc["in_s_channels"]:
        print(f"Warning: Adjusting decoder out_s_channels from {cfg_dec['out_s_channels']} to match encoder in_s_channels {cfg_enc['in_s_channels']} for reconstruction.")
        cfg_dec["out_s_channels"] = cfg_enc["in_s_channels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATrRegressor(cfg_enc=cfg_enc, cfg_agg=cfg_agg)
    model.to(device)

    if args.test:
        test_forward_on_gpu(model, cfg_enc, device)
        return
    data_paths = args.data_paths if args.data_paths else [
        "/nfs/cms/arqolmo/SDHCAL_Energy/GATrAutoencoder/flat_all.npz",
    ]
    
    preprocessing_cfg = {
            "use_scalar": args.use_scalar,
            "use_one_hot": args.use_one_hot,
            "use_energy": True,  # Siempre True para regresor de energía
            "use_log": args.use_log,
            "z_norm": args.z_norm,
            "z_norm_yaml_path": z_norm_yaml_path,
        }
    val_ratio = args.val_ratio
    mode = args.mode
    print(f"Cargando datos desde: {data_paths} con val_ratio={val_ratio} y mode={mode}")
    filters_cfg = cfg_models.get("filters", {})
    train_dataset, val_dataset = make_pf_splits(
        data_paths, val_ratio=val_ratio, mode=mode, preprocessing_cfg=preprocessing_cfg, filters=filters_cfg
    )

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, persistent_workers=True)

    if wandb is not None:
        wandb.init(
            project="gatr-regressor",
            config={
                "cfg_enc": cfg_enc,
                "cfg_agg": cfg_agg,
                "args": vars(args),
                # "batch_size": args.batch_size,
                # "val_ratio": val_ratio,
                # "mode": mode,
                # "epochs": args.epochs,
            },
        )
        wandb.watch(model, log="all", log_freq=100)
        wandb.log({"model/params": sum(p.numel() for p in model.parameters())})

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_scalar = args.use_scalar
    use_one_hot = args.use_one_hot
    use_energy = True
    use_log = args.use_log
    global_step = 0
    plot_every = args.plot_every if args.plot_every > 0 else max(1, args.epochs // 10)
    if args.z_norm:
        with open(z_norm_yaml_path, "r") as f:
            print("Loading z-score normalization stats from", z_norm_yaml_path)
            stats = yaml.safe_load(f)
    else:
        stats = None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    
    start_epoch = 0
    if args.resume is not None:
        print(f"Loading checkpoints from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)

        print(f"Resuming from epoch {start_epoch}, global_step {global_step}")
    
    for epoch in range(start_epoch, args.epochs):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            data = build_batch(batch,
                               use_scalar=use_scalar,
                               use_one_hot=use_one_hot,
                               use_log=use_log,
                               use_energy=use_energy,
                               z_norm=args.z_norm, stats=stats)
            mv_v_part = data["mv_v_part"].to(device)
            mv_s_part = data["mv_s_part"].to(device)
            scalars = data["scalars"].to(device)
            batch_idx = data["batch_idx"].to(device)

            outputs = model(mv_v_part, mv_s_part, scalars, batch_idx)
            loss = reconstruction_loss(outputs, data["logenergy"].to(device))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # ---- Debug: gradientes y predicciones cada 50 batches ----
            if global_step % 50 == 0:
                _log_gradient_stats(model, global_step)
                _log_prediction_debug(outputs, data["logenergy"].to(device), global_step)
                _log_aggregation_debug(model, mv_v_part, mv_s_part, scalars, batch_idx, global_step)

            # ---- Debug: event display cada 1000 batches ----
            if global_step % 1000 == 0:
                _log_event_display(mv_v_part, batch_idx, outputs, data["logenergy"].to(device), use_log, global_step)

            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            total_loss += loss.item()
            global_step += 1
            
            if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
                checkpoint_file = f"gatr_regressor_epoch{epoch+1}_step{global_step}.pt"
                checkpoint_path_full = os.path.join(checkpoint_path, checkpoint_file)
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, checkpoint_path_full)

                # torch.save(model.state_dict(), checkpoint_path_full)
                print(f"Checkpoint guardado en {checkpoint_path_full}")
            # Log por batch (train)
            if wandb is not None:
                wandb.log({"loss/train_batch": loss.item()}, step=global_step)

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        all_log_E_true = []
        all_log_E_pred = []
        with torch.no_grad():
            for batch in val_loader:
                data = build_batch(batch,
                                   use_scalar=use_scalar,
                                   use_one_hot=use_one_hot,
                                   use_energy=use_energy,
                                   use_log=use_log,
                                   z_norm=args.z_norm,
                                   stats=stats)
                mv_v_part = data["mv_v_part"].to(device)
                mv_s_part = data["mv_s_part"].to(device)
                scalars_v = data["scalars"].to(device)
                batch_idx_v = data["batch_idx"].to(device)

                outputs = model(mv_v_part, mv_s_part, scalars_v, batch_idx_v)
                loss = reconstruction_loss(outputs, data["logenergy"].to(device))
                val_loss += loss.item()

                # Acumular predicciones para gráficos
                all_log_E_true.append(data["logenergy"].view(-1).numpy())
                all_log_E_pred.append(outputs.squeeze().cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.6f} val_loss={avg_val_loss:.6f}")

        # Log por epoch (val + train promedio) usando global_step
        if wandb is not None:
            wandb.log({
                "loss/train_epoch": avg_loss,
                "loss/val": avg_val_loss,
            }, step=global_step)

        # Gráficos de regresión periódicos y al final
        if (epoch + 1) % plot_every == 0 or epoch == args.epochs - 1:
            all_log_E_true_np = np.concatenate(all_log_E_true)
            all_log_E_pred_np = np.concatenate(all_log_E_pred)
            # Convertir de log-energía a energía real si se usó la transformación logarítmica
            if use_log:
                E_true = np.exp(all_log_E_true_np)
                E_reco = np.exp(all_log_E_pred_np)
            else:
                E_true = all_log_E_true_np
                E_reco = all_log_E_pred_np
            _log_regression_plots(E_true, E_reco, step=global_step)
            
        # clean cache
        torch.cuda.empty_cache()

    if wandb is not None:
        wandb.finish()
    
    # Guardar modelo final
    final_model_path = os.path.join(output_path, "gatr_regressor_final.pt")
    # torch.save(model.state_dict(), final_model_path)
    torch.save({
    "epoch": epoch,
    "global_step": global_step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    }, final_model_path)

    print(f"Modelo final guardado en {final_model_path}")


if __name__ == "__main__":
    main()
