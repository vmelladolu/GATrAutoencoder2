import numpy as np
import polars as pl
from typing import Dict, Any
import os
import wandb
import matplotlib.pyplot as plt
from scipy.stats import norm


def mean_abs_rel_error(E_pred: np.ndarray, E_true: np.ndarray, eps: float = 1e-12) -> float:
    denom = np.where(np.abs(E_true) < eps, eps, E_true)
    return float(np.mean((E_pred - E_true)**2 / denom))

def metrics(df: pl.DataFrame, target_col: str, pred_col: str="E_reco") -> float:
    if len(df) == 0:
        return float("nan")
    y = df[target_col].to_numpy()
    yhat = df[pred_col].to_numpy()
    return mean_abs_rel_error(yhat, y)
  
def summarize_by_energy(df: pl.DataFrame, target_col: str, pred_col: str = "E_reco") -> Dict[Any, Dict[str, Any]]:
    """
    Devuelve un dict por cada valor de energía real (asumimos que target suele ser discreto),
    con error relativo cuadrático medio (E_reco - E_true)^2 / E_true y n.
    """
    # Cálculo en numpy para velocidad
    y = df[target_col].to_numpy()
    yhat = df[pred_col].to_numpy()
    rel = (yhat - y)**2 / np.where(np.abs(y) < 1e-12, 1e-12, y)
    # Agrupar por valores únicos de y (si es continuo con muchos valores, esto será grande)
    summary = {}
    for val in np.unique(y):
        mask = (y == val)
        if not np.any(mask):
            continue
        val = round(float(val),2)
        summary[val] = {
            "mean_abs_rel_error": float(np.mean(rel[mask])),
            "count": int(np.sum(mask)),
        }
    return summary




def plot_results(fitter, df: pl.DataFrame):
        # Convertir a pandas para facilitar agrupaciones
        df_plot = df.select([fitter.target_col, "E_reco"]).to_pandas()
        
        # Agrupar por valor de energía (si discreta) o por bins (si continua)
        if df_plot[fitter.target_col].nunique() < 30:
            grouped = df_plot.groupby(fitter.target_col)["E_reco"]
            E_true_vals = grouped.mean().index.to_numpy()
            E_reco_mean = grouped.mean().to_numpy()
            E_reco_std = grouped.std().to_numpy()
        else:
            n_bins = 30
            bins = np.linspace(df_plot[fitter.target_col].min(), df_plot[fitter.target_col].max(), n_bins + 1)
            df_plot["E_bin"] = np.digitize(df_plot[fitter.target_col], bins)
            grouped = df_plot.groupby("E_bin")
            E_true_vals = grouped[fitter.target_col].mean().to_numpy()
            E_reco_mean = grouped["E_reco"].mean().to_numpy()
            E_reco_std = grouped["E_reco"].std().to_numpy()

        # --- Plot 1: E_reco vs E_true con banda de error ---
        plt.figure(figsize=(8, 6))

        plt.scatter(
            df_plot[fitter.target_col],
            df_plot["E_reco"],
            alpha=0.9,
            s=8,
            color="gray",
            label="Individual Predictions"
        )

        plt.plot(E_true_vals, E_reco_mean, color="blue", label=r"$E_{reco} Mean$")

        plt.fill_between(
            E_true_vals,
            E_reco_mean - E_reco_std,
            E_reco_mean + E_reco_std,
            color="lightblue",
            alpha=0.4,
            label=r"±1$\sigma$"
        )

        min_E = df_plot[fitter.target_col].min()
        max_E = df_plot[fitter.target_col].max()
        plt.plot([min_E, max_E], [min_E, max_E], color="red", linestyle="--", label="E_reco = E_true")

        plt.xlabel(r"$E_{true}$")
        plt.ylabel(r"$E_{reco}$")
        plt.title(r"$E_{reco}$ vs $E_{true}$ (MLP Fit)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()

        output_fig_path = fitter.cfg.get("output_fig_path")
        if not output_fig_path:
            output_fig_path = fitter.cfg.get("output_results_dir")
            output_fig_path = os.path.join(output_fig_path, "Images")
        os.makedirs(output_fig_path, exist_ok=True)
        plt.savefig(os.path.join(output_fig_path, "E_reco_vs_E_true_mlp_w_error.png"), dpi=200)
        plt.close()

        # Save raw data for potential future analysis
        df_plot.to_csv(os.path.join(output_fig_path, "E_reco_vs_E_true_data.csv"), index=False)
        
        # --- Plot 2: Distribución del error relativo ---
        rel_error = (df_plot["E_reco"] - df_plot[fitter.target_col]) / df_plot[fitter.target_col]

        plt.figure(figsize=(8, 5))
        plt.hist(
            rel_error,
            bins=50,
            color="steelblue",
            alpha=0.75,
            edgecolor="black",
            label=r"($E_{reco} - E_{true}) / E_{true}$"
        )

        mean_err = np.mean(rel_error)
        std_err = np.std(rel_error)

        plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Error = 0")
        plt.axvline(mean_err, color="green", linestyle="-.", linewidth=1.2, label=f"Media = {mean_err:.3f}")
        plt.axvline(mean_err + std_err, color="gray", linestyle=":", linewidth=1, label=f"±1σ = {std_err:.3f}")
        plt.axvline(mean_err - std_err, color="gray", linestyle=":", linewidth=1)

        plt.xlabel(r"Error relativo  ($E_{reco} - E_{true}) / E_{true}$")
        plt.ylabel("Frecuencia")
        plt.title("Distribución del error relativo (MLP Fit)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        plt.savefig(os.path.join(output_fig_path, "relative_error_hist_mlp.png"), dpi=200)
        plt.close()

        # --- Plot 3: Resolución vs 1/sqrt(E) ---
        df_ratio = df.select([fitter.target_col, "E_reco"]).to_pandas()

        mean_vals = []
        std_vals = []

        if df_ratio[fitter.target_col].nunique() < 30:
            grouped = df_ratio.groupby(fitter.target_col)
            n_energy_values = len(grouped)
            nrows = int(np.ceil(n_energy_values / 3))
            ncols = min(3, n_energy_values)
            fig_dist = plt.figure(figsize=(ncols * 5, nrows * 4))
            E_true_vals = []
            E_pred_means = []
            
            for i,(E_true, group) in enumerate(grouped):
                E_true_vals.append(E_true)
                E_pred_means.append(group["E_reco"].mean())

                plt.subplot(nrows, ncols, i+1)
                # Residual relativo
                rel_res = (group["E_reco"] - E_true) / E_true

                # Ajuste a normal
                mu, sigma = norm.fit(rel_res)
                plt.hist(rel_res, bins=30, density=True, alpha=0.6, color="steelblue", edgecolor="black")
                x = np.linspace(rel_res.min(), rel_res.max(), 100)
                plt.plot(x, norm.pdf(x, mu, sigma), color="red", label=f"Fit Normal\nμ={mu:.3f}\nσ={sigma:.3f}")
                plt.xlabel(r"Residual relativo ($E_{reco} - E_{true}) / E_{true}$)")
                plt.ylabel("Densidad")
                plt.title(f"Distribución del residual relativo\npara $E_{{true}}$ = {E_true:.1f} GeV")
                plt.legend()
                mean_vals.append(mu)    
                std_vals.append(sigma)
            fig_dist.tight_layout()
            fig_dist.savefig(os.path.join(output_fig_path, "relative_residuals_by_energy_mlp.png"), dpi=200)
            plt.close(fig_dist)
            E_true_vals = np.array(E_true_vals)
            mean_vals = np.array(mean_vals)
            std_vals = np.array(std_vals)

        else:
            n_bins = 30
            bins = np.linspace(
                df_ratio[fitter.target_col].min(),
                df_ratio[fitter.target_col].max(),
                n_bins + 1
            )

            df_ratio["E_bin"] = np.digitize(df_ratio[fitter.target_col], bins)
            grouped = df_ratio.groupby("E_bin")

            E_true_vals = []
            E_pred_means = []
            for _, group in grouped:
                E_true_mean = group[fitter.target_col].mean()
                E_true_vals.append(E_true_mean)

                rel_res = (group["E_reco"] - group[fitter.target_col]) / group[fitter.target_col]
                E_pred_means.append(group["E_reco"].mean())
                mu, sigma = norm.fit(rel_res)

                mean_vals.append(mu)
                std_vals.append(sigma)

        E_true_vals = np.array(E_true_vals)
        mean_vals = np.array(mean_vals)
        std_vals = np.array(std_vals)

        ratio = std_vals / np.abs(E_pred_means)
        # ratio = std_vals

        plt.figure(figsize=(8, 6))
        plt.scatter(E_true_vals, ratio, s=50, color="blue", alpha=0.8, label=r"$\sigma(E_{reco})/E_{true}$")
        # plt.scatter(E_true_vals, ratio, s=50, color="purple", alpha=0.8, label=r"$\sigma(E)/\mu(E)$")
        plt.plot(E_true_vals, ratio, color="purple", alpha=0.6)
        plt.xlabel(r"$E_{true}$ [GeV]")
        plt.ylabel(r"$\sigma(E_{reco}) / E_{true}$")
        # plt.ylabel(r"$\sigma(E_{reco}) / \mu(E_{reco})$")
        plt.title("Std/E vs $E_{true}$")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_path, "std_vs_E_true_mlp.png"), dpi=200)
        plt.close()

        # --- Plot 4: Boxplot de E_reco por cada E_true ---
        unique_energies = sorted(df_plot[fitter.target_col].unique())
        box_data = [df_plot.loc[df_plot[fitter.target_col] == e, "E_reco"].values for e in unique_energies]

        fig_box, ax_box = plt.subplots(figsize=(max(8, len(unique_energies) * 0.8), 6))
        bp = ax_box.boxplot(box_data, positions=range(len(unique_energies)), widths=0.6,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(marker='.', markersize=2, alpha=0.3))
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax_box.set_xticks(range(len(unique_energies)))
        ax_box.set_xticklabels([f"{e:.0f}" for e in unique_energies], rotation=45)
        ax_box.plot(range(len(unique_energies)), unique_energies, 'r--', label=r"$E_{reco} = E_{true}$")
        ax_box.set_xlabel(r"$E_{true}$ [GeV]")
        ax_box.set_ylabel(r"$E_{reco}$ [GeV]")
        ax_box.set_title(r"Distribution of $E_{reco}$ per $E_{true}$ (MLP Fit)")
        ax_box.legend()
        ax_box.grid(True, linestyle=":", alpha=0.7)
        fig_box.tight_layout()
        fig_box.savefig(os.path.join(output_fig_path, "boxplot_E_reco_per_E_true_mlp.png"), dpi=200)
        plt.close(fig_box)
        
        if hasattr(fitter, 'use_wandb') and fitter.use_wandb and not fitter.cfg.get("trained", False):
            wandb.log({
                "E_reco_vs_E_true": wandb.Image(os.path.join(output_fig_path, "E_reco_vs_E_true_mlp_w_error.png")),
                "relative_error_hist": wandb.Image(os.path.join(output_fig_path, "relative_error_hist_mlp.png")),
                "resolution_vs_inv_sqrt_E": wandb.Image(os.path.join(output_fig_path, "std_vs_E_true_mlp.png")),
                "boxplot_E_reco_per_E_true": wandb.Image(os.path.join(output_fig_path, "boxplot_E_reco_per_E_true_mlp.png")),
                "relative_residuals_by_energy": wandb.Image(os.path.join(output_fig_path, "relative_residuals_by_energy_mlp.png")),
            })

        fitter.loggers["io"].info(f"Plots saved to in {output_fig_path}")