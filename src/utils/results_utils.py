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
  
def summarize_by_energy(df: pl.DataFrame, target_col: str, pred_col: str = "E_reco",
                        n_bins: int = 30) -> Dict[Any, Dict[str, Any]]:
    """
    Devuelve un dict por cada valor de energía real.
    - Si el target es discreto (≤ n_bins valores únicos): agrupa por valor exacto.
    - Si es continuo: agrupa en n_bins bins equiespaciados y usa el centro del bin como clave.
    Métrica por grupo: error relativo cuadrático medio (E_reco - E_true)^2 / E_true.
    """
    y = df[target_col].to_numpy()
    yhat = df[pred_col].to_numpy()
    rel = (yhat - y)**2 / np.where(np.abs(y) < 1e-12, 1e-12, y)

    unique_vals = np.unique(y)
    summary = {}

    if len(unique_vals) <= n_bins:
        for val in unique_vals:
            mask = (y == val)
            if not np.any(mask):
                continue
            key = round(float(val), 2)
            summary[key] = {
                "mean_abs_rel_error": float(np.mean(rel[mask])),
                "count": int(np.sum(mask)),
            }
    else:
        bins = np.linspace(y.min(), y.max(), n_bins + 1)
        bin_idx = np.digitize(y, bins)
        for b in range(1, n_bins + 1):
            mask = (bin_idx == b)
            if not np.any(mask):
                continue
            center = round(float(0.5 * (bins[b - 1] + bins[b])), 2)
            summary[center] = {
                "mean_abs_rel_error": float(np.mean(rel[mask])),
                "count": int(np.sum(mask)),
                "E_true_mean": round(float(y[mask].mean()), 4),
                "E_true_range": [round(float(y[mask].min()), 4), round(float(y[mask].max()), 4)],
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
                rel_res_finite = rel_res[np.isfinite(rel_res)]
                if len(rel_res_finite) < 2:
                    mean_vals.append(float("nan"))
                    std_vals.append(float("nan"))
                    continue
                mu, sigma = norm.fit(rel_res_finite)

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
        _MAX_BOX_BINS = 30
        n_unique_box = df_plot[fitter.target_col].nunique()
        if n_unique_box <= _MAX_BOX_BINS:
            unique_energies = sorted(df_plot[fitter.target_col].unique())
            box_data = [df_plot.loc[df_plot[fitter.target_col] == e, "E_reco"].values for e in unique_energies]
            box_labels = [f"{e:.0f}" for e in unique_energies]
        else:
            n_bins = _MAX_BOX_BINS
            bins = np.linspace(df_plot[fitter.target_col].min(), df_plot[fitter.target_col].max(), n_bins + 1)
            bin_idx = np.digitize(df_plot[fitter.target_col], bins)
            unique_energies, box_data, box_labels = [], [], []
            for b in range(1, n_bins + 1):
                mask = bin_idx == b
                if mask.sum() == 0:
                    continue
                center = 0.5 * (bins[b - 1] + bins[b])
                unique_energies.append(center)
                box_data.append(df_plot.loc[mask, "E_reco"].values)
                box_labels.append(f"{center:.0f}")

        fig_box, ax_box = plt.subplots(figsize=(max(8, len(unique_energies) * 0.8), 6))
        bp = ax_box.boxplot(box_data, positions=range(len(unique_energies)), widths=0.6,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(marker='.', markersize=2, alpha=0.3))
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax_box.set_xticks(range(len(unique_energies)))
        ax_box.set_xticklabels(box_labels, rotation=45)
        ax_box.plot(range(len(unique_energies)), unique_energies, 'r--', label=r"$E_{reco} = E_{true}$")
        ax_box.set_xlabel(r"$E_{true}$ [GeV]")
        ax_box.set_ylabel(r"$E_{reco}$ [GeV]")
        ax_box.set_title(r"Distribution of $E_{reco}$ per $E_{true}$ (MLP Fit)")
        ax_box.legend()
        ax_box.grid(True, linestyle=":", alpha=0.7)
        fig_box.tight_layout()
        fig_box.savefig(os.path.join(output_fig_path, "boxplot_E_reco_per_E_true_mlp.png"), dpi=200)
        plt.close(fig_box)
        
        # --- Plot 5: σ(E_pred)/μ(E_pred) vs E_true (Gaussian fit to raw predictions) ---
        df_gauss = df.select([fitter.target_col, "E_reco"]).to_pandas()

        gauss_mu_vals = []
        gauss_sigma_vals = []
        gauss_E_true_vals = []

        if df_gauss[fitter.target_col].nunique() < 30:
            grouped_gauss = df_gauss.groupby(fitter.target_col)
            for E_true, group in grouped_gauss:
                e_reco_finite = group["E_reco"].values[np.isfinite(group["E_reco"].values)]
                if len(e_reco_finite) < 2:
                    continue
                gauss_E_true_vals.append(E_true)
                mu_pred, sigma_pred = norm.fit(e_reco_finite)
                gauss_mu_vals.append(mu_pred)
                gauss_sigma_vals.append(sigma_pred)
        else:
            n_bins = 30
            bins_gauss = np.linspace(
                df_gauss[fitter.target_col].min(),
                df_gauss[fitter.target_col].max(),
                n_bins + 1
            )
            df_gauss["E_bin"] = np.digitize(df_gauss[fitter.target_col], bins_gauss)
            grouped_gauss = df_gauss.groupby("E_bin")
            for _, group in grouped_gauss:
                e_reco_finite = group["E_reco"].values[np.isfinite(group["E_reco"].values)]
                if len(e_reco_finite) < 2:
                    continue
                gauss_E_true_vals.append(group[fitter.target_col].mean())
                mu_pred, sigma_pred = norm.fit(e_reco_finite)
                gauss_mu_vals.append(mu_pred)
                gauss_sigma_vals.append(sigma_pred)

        gauss_E_true_vals = np.array(gauss_E_true_vals)
        gauss_mu_vals = np.array(gauss_mu_vals)
        gauss_sigma_vals = np.array(gauss_sigma_vals)
        resolution_gauss = gauss_sigma_vals / np.abs(gauss_mu_vals)

        plt.figure(figsize=(8, 6))
        plt.scatter(gauss_E_true_vals, resolution_gauss, s=50, color="darkorange", alpha=0.8,
                    label=r"$\sigma(E_{pred}) / \mu(E_{pred})$")
        plt.plot(gauss_E_true_vals, resolution_gauss, color="coral", alpha=0.6)
        plt.xlabel(r"$E_{true}$ [GeV]")
        plt.ylabel(r"$\sigma(E_{pred}) / \mu(E_{pred})$")
        plt.title(r"$\sigma(E_{pred}) / \mu(E_{pred})$ vs $E_{true}$ (Gaussian fit)")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_path, "resolution_gauss_fit_vs_E_true.png"), dpi=200)
        plt.close()

        # --- Plot 6 & 7: Resolution & Linearity with double Gaussian fit (paper method) ---
        df_dg = df.select([fitter.target_col, "E_reco"]).to_pandas()

        dg_E_true_vals = []
        dg_mu2_vals = []
        dg_sigma2_vals = []

        def _double_gauss_fit(e_reco_arr):
            e_reco_arr = e_reco_arr[np.isfinite(e_reco_arr)]
            if len(e_reco_arr) < 3:
                return float("nan"), float("nan")
            mu1, sigma1 = norm.fit(e_reco_arr)
            lo, hi = mu1 - 1.5 * sigma1, mu1 + 1.5 * sigma1
            filtered = e_reco_arr[(e_reco_arr >= lo) & (e_reco_arr <= hi)]
            if len(filtered) < 3:
                return mu1, sigma1
            mu2, sigma2 = norm.fit(filtered)
            return mu2, sigma2

        if df_dg[fitter.target_col].nunique() < 30:
            grouped_dg = df_dg.groupby(fitter.target_col)
            for E_true, group in grouped_dg:
                mu2, sigma2 = _double_gauss_fit(group["E_reco"].values)
                dg_E_true_vals.append(E_true)
                dg_mu2_vals.append(mu2)
                dg_sigma2_vals.append(sigma2)
        else:
            n_bins = 30
            bins_dg = np.linspace(
                df_dg[fitter.target_col].min(),
                df_dg[fitter.target_col].max(),
                n_bins + 1
            )
            df_dg["E_bin"] = np.digitize(df_dg[fitter.target_col], bins_dg)
            grouped_dg = df_dg.groupby("E_bin")
            for _, group in grouped_dg:
                mu2, sigma2 = _double_gauss_fit(group["E_reco"].values)
                dg_E_true_vals.append(group[fitter.target_col].mean())
                dg_mu2_vals.append(mu2)
                dg_sigma2_vals.append(sigma2)

        dg_E_true_vals = np.array(dg_E_true_vals)
        dg_mu2_vals = np.array(dg_mu2_vals)
        dg_sigma2_vals = np.array(dg_sigma2_vals)

        # --- Plot 6: Resolution R = sigma2 / mu2 vs E_true ---
        resolution_dg = dg_sigma2_vals / np.abs(dg_mu2_vals)

        plt.figure(figsize=(8, 6))
        plt.scatter(dg_E_true_vals, resolution_dg, s=50, color="teal", alpha=0.8,
                    label=r"$R = \sigma_{E_{reco}} / E_{reco}$")
        plt.plot(dg_E_true_vals, resolution_dg, color="darkslategray", alpha=0.6)
        plt.xlabel(r"$E_{mc}$ [GeV]")
        plt.ylabel(r"$R = \sigma_{E_{reco}} / E_{reco}$")
        plt.title(r"Resolution $R = \sigma_{E_{reco}} / E_{reco}$ vs $E_{mc}$ (double Gaussian fit)")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_path, "resolution_double_gauss_vs_E_true.png"), dpi=200)
        plt.close()

        # --- Plot 7: Linearity dE/E = (mu2 - E_mc) / E_mc vs E_true ---
        linearity = (dg_mu2_vals - dg_E_true_vals) / dg_E_true_vals

        plt.figure(figsize=(8, 6))
        plt.scatter(dg_E_true_vals, linearity, s=50, color="crimson", alpha=0.8,
                    label=r"$\Delta E / E_{mc}$")
        plt.plot(dg_E_true_vals, linearity, color="firebrick", alpha=0.6)
        plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        plt.xlabel(r"$E_{mc}$ [GeV]")
        plt.ylabel(r"$\Delta E / E_{mc} = (E_{reco} - E_{mc}) / E_{mc}$")
        plt.title(r"Linearity $\Delta E / E_{mc}$ vs $E_{mc}$ (double Gaussian fit)")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_path, "linearity_double_gauss_vs_E_true.png"), dpi=200)
        plt.close()

        if hasattr(fitter, 'use_wandb') and fitter.use_wandb and not fitter.cfg.get("trained", False):
            wandb.log({
                "E_reco_vs_E_true": wandb.Image(os.path.join(output_fig_path, "E_reco_vs_E_true_mlp_w_error.png")),
                "relative_error_hist": wandb.Image(os.path.join(output_fig_path, "relative_error_hist_mlp.png")),
                "resolution_vs_inv_sqrt_E": wandb.Image(os.path.join(output_fig_path, "std_vs_E_true_mlp.png")),
                "boxplot_E_reco_per_E_true": wandb.Image(os.path.join(output_fig_path, "boxplot_E_reco_per_E_true_mlp.png")),
                "relative_residuals_by_energy": wandb.Image(os.path.join(output_fig_path, "relative_residuals_by_energy_mlp.png")),
                "resolution_gauss_fit": wandb.Image(os.path.join(output_fig_path, "resolution_gauss_fit_vs_E_true.png")),
                "resolution_double_gauss": wandb.Image(os.path.join(output_fig_path, "resolution_double_gauss_vs_E_true.png")),
                "linearity_double_gauss": wandb.Image(os.path.join(output_fig_path, "linearity_double_gauss_vs_E_true.png")),
            })

        fitter.loggers["io"].info(f"Plots saved to in {output_fig_path}")