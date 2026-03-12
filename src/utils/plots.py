import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from scipy.stats import norm


def _log_event_display(mv_v_part, batch_idx, outputs, logenergy, use_log, step=None):
    if plt is None or wandb is None:
        return
    pos = mv_v_part.detach().cpu()
    bidx = batch_idx.detach().cpu()
    pred = outputs.detach().cpu()
    tgt = logenergy.detach().cpu().view(-1)

    unique_events = bidx.unique()
    evt = unique_events[torch.randint(len(unique_events), (1,)).item()].item()
    mask = bidx == evt

    x = pos[mask, 0].numpy()
    y = pos[mask, 1].numpy()
    z = pos[mask, 2].numpy()

    e_pred_val = float(pred[evt])
    e_true_val = float(tgt[evt])

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
    ax1.scatter(x, y, s=4, alpha=0.7, c="steelblue")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Proyeccion XY")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.scatter(y, z, s=4, alpha=0.7, c="darkorange")
    ax2.set_xlabel("Y")
    ax2.set_ylabel("Z")
    ax2.set_title("Proyeccion YZ")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    wandb.log({"debug/event_display": wandb.Image(fig)})
    plt.close(fig)


def _log_regression_plots(E_true, E_reco, step=None):
    if plt is None or wandb is None:
        return
    import pandas as pd

    df_plot = pd.DataFrame({"E_true": E_true, "E_reco": E_reco})

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

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(df_plot["E_true"], df_plot["E_reco"], alpha=0.9, s=8, color="gray", label="Individual Predictions")
    ax1.plot(E_true_vals, E_reco_mean, color="blue", label=r"$E_{reco}$ Mean")
    ax1.fill_between(E_true_vals, E_reco_mean - E_reco_std, E_reco_mean + E_reco_std, color="lightblue", alpha=0.4, label=r"$\pm1\sigma$")
    min_e, max_e = df_plot["E_true"].min(), df_plot["E_true"].max()
    ax1.plot([min_e, max_e], [min_e, max_e], color="red", linestyle="--", label=r"$E_{reco}=E_{true}$")
    ax1.set_xlabel(r"$E_{true}$ [GeV]")
    ax1.set_ylabel(r"$E_{reco}$ [GeV]")
    ax1.set_title(r"$E_{reco}$ vs $E_{true}$ (GATr Regressor)")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.7)
    fig1.tight_layout()

    rel_error = (df_plot["E_reco"] - df_plot["E_true"]) / df_plot["E_true"]
    mean_err = float(np.mean(rel_error))
    std_err = float(np.std(rel_error))

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(rel_error, bins=50, color="steelblue", alpha=0.75, edgecolor="black", label=r"$(E_{reco}-E_{true})/E_{true}$")
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Error = 0")
    ax2.axvline(mean_err, color="green", linestyle="-.", linewidth=1.2, label=f"Media = {mean_err:.3f}")
    ax2.axvline(mean_err + std_err, color="gray", linestyle=":", linewidth=1, label=f"$\pm1\sigma$ = {std_err:.3f}")
    ax2.axvline(mean_err - std_err, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel(r"Error relativo $(E_{reco}-E_{true})/E_{true}$")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title("Distribucion del error relativo")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)
    fig2.tight_layout()

    mean_vals, std_vals = [], []
    if df_plot["E_true"].nunique() < 30:
        grouped_res = df_plot.groupby("E_true")
        e_true_res = []
        n_energy_values = len(grouped_res)
        nrows = int(np.ceil(n_energy_values / 3))
        ncols = min(3, n_energy_values)
        fig_dist = plt.figure(figsize=(ncols * 5, nrows * 4))
        for i, (e_val, group) in enumerate(grouped_res):
            plt.subplot(nrows, ncols, i + 1)
            e_true_res.append(e_val)
            rel_res = (group["E_reco"] - e_val) / e_val
            mu, sigma = norm.fit(rel_res)
            plt.hist(rel_res, bins=20, alpha=0.6, density=True)
            x = np.linspace(rel_res.min(), rel_res.max(), 200)
            plt.plot(x, norm.pdf(x, mu, sigma), "r-")
            plt.title(f"E_true = {e_val:.2f}")
            plt.grid(True, linestyle=":", alpha=0.7)
            plt.xlabel("Relative Residuals")
            plt.ylabel("Count")
            mean_vals.append(mu)
            std_vals.append(sigma)
        fig_dist.tight_layout()
        e_true_res = np.array(e_true_res)
    else:
        fig_dist = None
        n_bins_res = 30
        bins_res = np.linspace(df_plot["E_true"].min(), df_plot["E_true"].max(), n_bins_res + 1)
        df_plot["E_bin_res"] = np.digitize(df_plot["E_true"], bins_res)
        grouped_res = df_plot.groupby("E_bin_res")
        e_true_res = []
        for _, group in grouped_res:
            e_true_res.append(group["E_true"].mean())
            rel_res = (group["E_reco"] - group["E_true"]) / group["E_true"]
            mu, sigma = norm.fit(rel_res)
            mean_vals.append(mu)
            std_vals.append(sigma)
        e_true_res = np.array(e_true_res)

    mean_vals = np.array(mean_vals)
    std_vals = np.array(std_vals)
    ratio = std_vals / np.abs(mean_vals).clip(min=1e-6)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.scatter(e_true_res, ratio, s=50, color="purple", alpha=0.8, label=r"$\sigma(E)/\mu(E)$")
    ax3.plot(e_true_res, ratio, color="purple", alpha=0.6)
    ax3.set_xlabel(r"$E_{true}$")
    ax3.set_ylabel(r"$\sigma(E_{reco})/\mu(E_{reco})$")
    ax3.set_title("Resolution vs E")
    ax3.grid(True, linestyle=":", alpha=0.7)
    ax3.legend()
    fig3.tight_layout()

    unique_energies = sorted(df_plot["E_true"].unique())
    is_continuous = len(unique_energies) > 30

    if is_continuous:
        n_box_bins = 20
        box_edges = np.linspace(df_plot["E_true"].min(), df_plot["E_true"].max(), n_box_bins + 1)
        box_centers = 0.5 * (box_edges[:-1] + box_edges[1:])
        df_plot["_box_bin"] = np.digitize(df_plot["E_true"], box_edges).clip(1, n_box_bins)
        box_data = []
        box_labels = []
        box_ref = []
        for b in range(1, n_box_bins + 1):
            vals = df_plot.loc[df_plot["_box_bin"] == b, "E_reco"].values
            if len(vals) > 0:
                box_data.append(vals)
                box_labels.append(f"{box_centers[b-1]:.0f}")
                box_ref.append(box_centers[b-1])
    else:
        box_data = [df_plot.loc[df_plot["E_true"] == e, "E_reco"].values for e in unique_energies]
        box_labels = [f"{e:.0f}" for e in unique_energies]
        box_ref = list(unique_energies)

    fig4, ax4 = plt.subplots(figsize=(max(8, len(box_data) * 0.8), 6))
    bp = ax4.boxplot(
        box_data,
        positions=range(len(box_data)),
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax4.set_xticks(range(len(box_data)))
    ax4.set_xticklabels(box_labels, rotation=45)
    ax4.plot(range(len(box_ref)), box_ref, "r--", label=r"$E_{reco}=E_{true}$")
    ax4.set_xlabel(r"$E_{true}$ [GeV]")
    ax4.set_ylabel(r"$E_{reco}$ [GeV]")
    ax4.set_title(r"Distribution of $E_{reco}$ per $E_{true}$")
    ax4.legend()
    ax4.grid(True, linestyle=":", alpha=0.7)
    fig4.tight_layout()

    log_dict = {
        "plots/E_reco_vs_E_true": wandb.Image(fig1),
        "plots/relative_error_hist": wandb.Image(fig2),
        "plots/resolution_vs_E": wandb.Image(fig3),
        "plots/boxplot_E_reco_per_E_true": wandb.Image(fig4),
        "metrics/rel_error_mean": mean_err,
        "metrics/rel_error_std": std_err,
        "metrics/MAE_rel": float(np.mean(np.abs(rel_error))),
    }
    if fig_dist is not None:
        log_dict["plots/relative_residuals_distribution"] = wandb.Image(fig_dist)
    wandb.log(log_dict)

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    if fig_dist is not None:
        plt.close(fig_dist)
