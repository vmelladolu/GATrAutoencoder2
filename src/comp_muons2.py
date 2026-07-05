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
    vals = pd.to_numeric(df[key], errors="coerce").dropna().to_numpy()
    vals = vals[(vals >= 0) & (vals <= xmax)]
    return vals


def stats_text(values, label):
    mean = np.mean(values) if len(values) else np.nan
    std = np.std(values, ddof=1) if len(values) > 1 else np.nan
    n = len(values)
    return f"{label}\nN = {n}\nMean = {mean:.2f}\nStd = {std:.2f}"


def add_statboxes(ax, tb_vals, sim_vals):
    ax.text(
        0.04, 0.96, stats_text(tb_vals, "TB data"),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.45", alpha=0.95),
        zorder=10,
    )

    ax.text(
        0.04, 0.73, stats_text(sim_vals, "Simulation"),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="#d8e8f5", edgecolor="#87aeca", alpha=0.95),
        zorder=10,
    )


def setup_style():
    plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#E5E5E5",
        "savefig.facecolor": "white",
        "axes.edgecolor": "0.25",
        "axes.linewidth": 1.0,
        "grid.color": "#c6c6c6",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.55,
        "font.size": 11,
        "axes.titlesize": 17,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
    })


def style_axis(ax, key):
    ax.set_title(key, pad=8)
    ax.set_xlabel(key)
    ax.set_ylabel("Density")
    ax.set_xlim(0, XMAX[key])
    ax.set_ylim(0, YMAX[key])
    ax.set_axisbelow(True)
    ax.grid(True, which="major")

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4,
        width=1,
        colors="0.2",
        labelsize=11,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.25")
        spine.set_linewidth(1.0)


def plot_panel(ax, tb_df, sim_df, key):
    xmax = XMAX[key]

    tb_vals = get_vals(tb_df, key, xmax)
    sim_vals = get_vals(sim_df, key, xmax)

    if len(tb_vals) == 0 and len(sim_vals) == 0:
        style_axis(ax, key)
        return

    combined = (
        np.concatenate([tb_vals, sim_vals])
        if len(tb_vals) and len(sim_vals)
        else (tb_vals if len(tb_vals) else sim_vals)
    )

    lo = float(np.min(combined))
    hi = float(np.max(combined))

    if lo == hi:
        bins = 20
    else:
        bins = np.linspace(lo, hi, 40)

    if len(sim_vals):
        ax.hist(
            sim_vals,
            bins=bins,
            density=True,
            alpha=0.75,
            color="#8fb3d1",
            label="Simulation",
        )

    if len(tb_vals):
        ax.hist(
            tb_vals,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.7,
            color="black",
            label="TB data",
        )

    style_axis(ax, key)
    add_statboxes(ax, tb_vals, sim_vals)

    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("TB data"), labels.index("Simulation")] if "TB data" in labels and "Simulation" in labels else range(len(labels))
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=0.9,
        facecolor="#f0f0f0",
        edgecolor="0.75",
        borderpad=0.3,
        handlelength=1.6,
        labelspacing=0.3,
    )


def build_summary(tb_df, sim_df):
    rows = []
    for key in PLOT_KEYS:
        xmax = XMAX[key]
        tb_vals = get_vals(tb_df, key, xmax)
        sim_vals = get_vals(sim_df, key, xmax)

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


def plot_comparison(tb_df, sim_df, output_png, title=None, header_text=None):
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, key in zip(axes, PLOT_KEYS):
        plot_panel(ax, tb_df, sim_df, key)

    if title:
        fig.suptitle(title, fontsize=20, y=0.975)

    if header_text:
        fig.text(0.5, 0.94, header_text, ha="center", va="bottom", fontsize=11)

    plt.subplots_adjust(left=0.08, right=0.985, bottom=0.085, top=0.91, wspace=0.15, hspace=0.28)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Compara nHits entre test beam y simulación a partir de dos CSV usando la misma lógica de histogramas del script base."
    )
    ap.add_argument("--tb-csv", required=True)
    ap.add_argument("--sim-csv", required=True)
    ap.add_argument("--output-dir", default="plots_muon_nhits")
    ap.add_argument("--title", default=None)
    ap.add_argument(
        "--header-text",
        default="Best parameters: thr=[0.165, 7.795, 19.795] pC  eff=0.940  pm=6.841  pw=0.500",
    )
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
        header_text=args.header_text,
    )


if __name__ == "__main__":
    main()
