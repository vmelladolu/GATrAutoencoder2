#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_KEYS = ["nHits_total", "nHits_1", "nHits_2", "nHits_3"]

COLUMN_ALIASES = {
    "nHits_total": ["nHits_total", "nhits", "n_hits_total", "nHits", "Nhits"],
    "nHits_1": ["nHits_1", "nhits_1", "nHits1"],
    "nHits_2": ["nHits_2", "nhits_2", "nHits2"],
    "nHits_3": ["nHits_3", "nhits_3", "nHits3"],
}

XMIN = {
    "nHits_total": 0,
    "nHits_1": 0,
    "nHits_2": 0,
    "nHits_3": 0,
}

XMAX = {
    "nHits_total": 145,
    "nHits_1": 145,
    "nHits_2": 60,
    "nHits_3": 35,
}

YMAX = {
    "nHits_total": 0.046,
    "nHits_1": 0.052,
    "nHits_2": 0.11,
    "nHits_3": 0.42,
}

NBINS = {
    "nHits_total": 42,
    "nHits_1": 42,
    "nHits_2": 32,
    "nHits_3": 24,
}


def read_csv_any(path):
    return pd.read_csv(path)


def standardize_columns(df):
    df = df.copy()
    rename = {}
    present = set(df.columns)
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in present:
                rename[alias] = target
                break
    return df.rename(columns=rename)


def get_vals(df, key, xmax):
    if key not in df.columns:
        return np.array([])
    vals = pd.to_numeric(df[key], errors="coerce").dropna().to_numpy()
    vals = vals[(vals >= 0) & (vals <= xmax)]
    return vals


def stats_text(values, label):
    mean = np.mean(values) if len(values) else np.nan
    std = np.std(values, ddof=1) if len(values) > 1 else np.nan
    n = len(values)
    return f"{label}\nN = {n}\nMean = {mean:.2f}\nStd = {std:.2f}"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "grid.color": "#b0b0b0",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.28,
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "legend.fontsize": 12,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.titlesize": 19,
    })


def style_axis(ax, key):
    ax.set_title(key, pad=12)
    ax.set_xlabel(key)
    ax.set_ylabel("Density")
    ax.set_xlim(XMIN[key], XMAX[key])
    ax.set_ylim(0, YMAX[key])
    ax.set_axisbelow(True)
    ax.grid(True, which="major")

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4.5,
        width=1,
        colors="0.15",
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)


def add_statboxes(ax, tb_vals, sim_vals):
    ax.text(
        0.72, 0.72,
        stats_text(tb_vals, "TB data"),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="0.45",
            alpha=0.97,
        ),
        zorder=10,
    )

    ax.text(
        0.72, 0.47,
        stats_text(sim_vals, "Simulation"),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox=dict(
            boxstyle="round",
            facecolor="#d8e8f5",
            edgecolor="#87aeca",
            alpha=0.97,
        ),
        zorder=10,
    )


def make_bins(key):
    return np.linspace(XMIN[key], XMAX[key], NBINS[key] + 1)


def plot_panel(ax, tb_df, sim_df, key):
    tb_vals = get_vals(tb_df, key, XMAX[key])
    sim_vals = get_vals(sim_df, key, XMAX[key])

    if len(tb_vals) == 0 and len(sim_vals) == 0:
        style_axis(ax, key)
        return

    bins = make_bins(key)

    if len(sim_vals):
        ax.hist(
            sim_vals,
            bins=bins,
            density=True,
            alpha=0.72,
            color="#8fb3d1",
            edgecolor="#8fb3d1",
            linewidth=0.20,
            label="Simulation",
        )

    if len(tb_vals):
        ax.hist(
            tb_vals,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color="black",
            label="TB data",
        )

    style_axis(ax, key)

    handles, labels = ax.get_legend_handles_labels()
    if "TB data" in labels and "Simulation" in labels:
        order = [labels.index("TB data"), labels.index("Simulation")]
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]

    ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.8",
        borderpad=0.40,
        handlelength=1.8,
        labelspacing=0.35,
    )

    add_statboxes(ax, tb_vals, sim_vals)


def build_summary(tb_df, sim_df):
    rows = []
    for key in PLOT_KEYS:
        tb_vals = get_vals(tb_df, key, XMAX[key])
        sim_vals = get_vals(sim_df, key, XMAX[key])

        rows.append({
            "feature": key,
            "tb_n": len(tb_vals),
            "tb_mean": float(np.mean(tb_vals)) if len(tb_vals) else np.nan,
            "tb_std": float(np.std(tb_vals, ddof=1)) if len(tb_vals) > 1 else np.nan,
            "sim_n": len(sim_vals),
            "sim_mean": float(np.mean(sim_vals)) if len(sim_vals) else np.nan,
            "sim_std": float(np.std(sim_vals, ddof=1)) if len(sim_vals) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def plot_comparison(tb_df, sim_df, output_png, title=None):
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
    axes = axes.ravel()

    for ax, key in zip(axes, PLOT_KEYS):
        plot_panel(ax, tb_df, sim_df, key)

    if title:
        fig.suptitle(title, fontsize=19, y=0.97)

    plt.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.09,
        top=0.90,
        wspace=0.24,
        hspace=0.38,
    )

    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Compara nHits entre test beam y simulación a partir de dos CSV."
    )
    ap.add_argument("--tb-csv", required=True)
    ap.add_argument("--sim-csv", required=True)
    ap.add_argument("--output-dir", default="plots_muon_nhits")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tb_df = standardize_columns(read_csv_any(args.tb_csv))
    sim_df = standardize_columns(read_csv_any(args.sim_csv))

    summary = build_summary(tb_df, sim_df)
    summary.to_csv(output_dir / "summary_nhits.csv", index=False)

    plot_comparison(
        tb_df,
        sim_df,
        output_dir / "nhits_comparison.png",
        title=args.title,
    )


if __name__ == "__main__":
    main()
