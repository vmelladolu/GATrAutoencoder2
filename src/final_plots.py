#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_LABEL_ENERGIES = [20.0, 50.0, 80.0]
SIM_WINDOWS = {
    20.0: (18.0, 22.0),
    50.0: (48.0, 52.0),
    80.0: (78.0, 82.0),
}

PARTICLE_TITLE = {
    "electron": "Electrons",
    "pion": "Pions",
}

COLUMN_ALIASES = {
    "nHits_total": ["nHits_total", "nhits", "n_hits_total", "nHits", "Nhits"],
    "nHits_1": ["nHits_1", "nhits_1", "nHits1"],
    "nHits_2": ["nHits_2", "nhits_2", "nHits2"],
    "nHits_3": ["nHits_3", "nhits_3", "nHits3"],
    "energy": ["energy", "beam_energy"],
}

ENERGY_KEYS = ["energy", "energies", "beam_energy", "true_energy", "E", "Energy"]


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


def infer_energy_from_path(path):
    s = Path(path).stem.lower()
    for e in [20, 50, 80]:
        if str(e) in s:
            return float(e)
    return None


def load_nhits_from_h5(h5file):
    with h5py.File(h5file, "r") as f:
        offsets = f["offsets"][:]
    return np.diff(offsets)


def load_energy_from_h5(h5file):
    with h5py.File(h5file, "r") as f:
        for key in ENERGY_KEYS:
            if key in f:
                arr = np.asarray(f[key][:]).reshape(-1)
                return arr.astype(np.float32)
    return None


def build_event_grpc_masks(h5_path, n_first=3, n_last=5, max_empty=2):
    with h5py.File(h5_path, "r") as f:
        offsets = f["offsets"][:]
        k_hits = f["k"][:]

    n_events = len(offsets) - 1
    first_signal = np.zeros(n_events, dtype=bool)
    last_signal = np.zeros(n_events, dtype=bool)
    complete_event = np.zeros(n_events, dtype=bool)
    unique_track = np.zeros(n_events, dtype=bool)

    for ev in range(n_events):
        start = offsets[ev]
        stop = offsets[ev + 1]
        ks = np.unique(k_hits[start:stop])
        if ks.size == 0:
            continue

        first_signal[ev] = np.any(ks < n_first)
        last_signal[ev] = np.any(ks >= (ks.max() - n_last + 1))

        sorted_ks = np.sort(ks)
        gaps = np.diff(sorted_ks)
        max_gap = gaps.max() if len(gaps) else 0
        complete_event[ev] = max_gap <= (max_empty + 1)
        unique_track[ev] = True

    return first_signal, last_signal, complete_event, unique_track


def load_k_per_event(h5file):
    with h5py.File(h5file, "r") as f:
        offsets = f["offsets"][:]
        k_hits = f["k"][:]

    n_events = len(offsets) - 1
    K_event = np.zeros(n_events, dtype=np.int32)

    for ev in range(n_events):
        start = offsets[ev]
        stop = offsets[ev + 1]
        if stop <= start:
            K_event[ev] = 0
            continue
        K_event[ev] = len(np.unique(k_hits[start:stop]))

    return K_event


def load_event_xyzk(h5file):
    with h5py.File(h5file, "r") as f:
        offsets = f["offsets"][:]
        x_hits = f["x"][:]
        y_hits = f["y"][:]
        k_hits = f["k"][:]
        thr = f["thr"][:] if "thr" in f else None

    return offsets, x_hits, y_hits, k_hits, thr


def compute_second_max_hits_per_layer(offsets, k_hits):
    n_events = len(offsets) - 1
    second_max_hits = np.zeros(n_events, dtype=np.int32)

    for ev in range(n_events):
        start = offsets[ev]
        stop = offsets[ev + 1]
        ks = k_hits[start:stop]
        if len(ks) == 0:
            continue

        _, counts = np.unique(ks, return_counts=True)
        counts_sorted = np.sort(counts)[::-1]
        second_max_hits[ev] = counts_sorted[1] if len(counts_sorted) >= 2 else counts_sorted[0]

    return second_max_hits


def compute_longitudinal_per_event(offsets, k_hits, n_first_layers=14):
    n_events = len(offsets) - 1
    longitudinal = np.zeros(n_events, dtype=np.float32)
    nhits_first = np.zeros(n_events, dtype=np.int32)

    for ev in range(n_events):
        start = offsets[ev]
        stop = offsets[ev + 1]
        ks = k_hits[start:stop]
        totalhits = len(ks)
        if totalhits == 0:
            continue

        hits_first = np.sum(ks < n_first_layers)
        nhits_first[ev] = hits_first
        longitudinal[ev] = hits_first / totalhits

    return longitudinal, nhits_first


def compute_lateral_per_event(offsets, x_hits, y_hits, k_hits, layers_for_axis=10, radius=13):
    n_events = len(offsets) - 1
    lateral = np.zeros(n_events, dtype=np.float32)
    lateral_hits = np.zeros(n_events, dtype=np.int32)
    axis_x = np.full(n_events, np.nan, dtype=np.float32)
    axis_y = np.full(n_events, np.nan, dtype=np.float32)
    half_window = radius / 2.0

    for ev in range(n_events):
        start = offsets[ev]
        stop = offsets[ev + 1]
        xs = x_hits[start:stop]
        ys = y_hits[start:stop]
        ks = k_hits[start:stop]
        totalhits = len(xs)

        if totalhits == 0:
            continue

        mask_axis = ks < layers_for_axis
        if not np.any(mask_axis):
            continue

        x_mean = np.mean(xs[mask_axis])
        y_mean = np.mean(ys[mask_axis])
        axis_x[ev] = x_mean
        axis_y[ev] = y_mean

        mask_radius = (
            (xs >= x_mean - half_window)
            & (xs <= x_mean + half_window)
            & (ys >= y_mean - half_window)
            & (ys <= y_mean + half_window)
        )

        nhitsinradius = np.sum(mask_radius)
        lateral_hits[ev] = nhitsinradius
        lateral[ev] = nhitsinradius / totalhits

    return lateral, lateral_hits, axis_x, axis_y


def compute_threshold_hits_per_event(offsets, thr):
    n_events = len(offsets) - 1
    n1 = np.zeros(n_events, dtype=np.int32)
    n2 = np.zeros(n_events, dtype=np.int32)
    n3 = np.zeros(n_events, dtype=np.int32)

    if thr is None:
        return n1, n2, n3

    for ev in range(n_events):
        start = offsets[ev]
        stop = offsets[ev + 1]
        vals = thr[start:stop]
        if len(vals) == 0:
            continue

        n1[ev] = np.sum(vals == 1)
        n2[ev] = np.sum(vals == 2)
        n3[ev] = np.sum(vals == 3)

    return n1, n2, n3


def add_reconstructed_features(csv_path, h5_path, forced_energy=None):
    df = read_csv_any(csv_path)
    df = standardize_columns(df)

    nhits = load_nhits_from_h5(h5_path)
    energy_h5 = load_energy_from_h5(h5_path)
    K_event = load_k_per_event(h5_path)
    first_signal, last_signal, complete_event, unique_track = build_event_grpc_masks(h5_path)
    offsets, x_hits, y_hits, k_hits, thr = load_event_xyzk(h5_path)
    second_max_hits = compute_second_max_hits_per_layer(offsets, k_hits)
    longitudinal_14, nhits_first14 = compute_longitudinal_per_event(offsets, k_hits, n_first_layers=14)
    lateral_x13, lateral_hits_x13, axis_x, axis_y = compute_lateral_per_event(
        offsets, x_hits, y_hits, k_hits, layers_for_axis=10, radius=13
    )
    n1, n2, n3 = compute_threshold_hits_per_event(offsets, thr)

    arrays = [nhits, K_event, first_signal, second_max_hits, longitudinal_14, lateral_x13, n1, n2, n3]
    if energy_h5 is not None:
        arrays.append(energy_h5)

    n = min([len(df)] + [len(a) for a in arrays])

    df = df.iloc[:n].copy()
    df["event_id"] = np.arange(n, dtype=int)
    df["nHits_total"] = nhits[:n]
    df["K_event"] = K_event[:n]
    df["second_max_hits"] = second_max_hits[:n]
    df["nhits_first14"] = nhits_first14[:n]
    df["longitudinal"] = longitudinal_14[:n]
    df["lateral_hits_x13"] = lateral_hits_x13[:n]
    df["lateral"] = lateral_x13[:n]
    df["axis_x"] = axis_x[:n]
    df["axis_y"] = axis_y[:n]
    df["first_signal"] = first_signal[:n]
    df["last_signal"] = last_signal[:n]
    df["complete_event"] = complete_event[:n]
    df["unique_track"] = unique_track[:n]
    df["nHits_1"] = n1[:n]
    df["nHits_2"] = n2[:n]
    df["nHits_3"] = n3[:n]
    df["density"] = df["nHits_total"] / pd.Series(df["K_event"]).replace(0, np.nan)

    if forced_energy is not None:
        df["energy"] = float(forced_energy)
    elif energy_h5 is not None:
        df["energy"] = energy_h5[:n]
    elif "energy" not in df.columns:
        inferred = infer_energy_from_path(csv_path)
        if inferred is not None:
            df["energy"] = inferred

    return df


def filter_reconstructed(df, min_density=0, min_energy=5):
    df = df.copy()

    if "density" in df.columns and min_density > 0:
        df = df[df["density"] >= min_density]

    if "energy" in df.columns:
        df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
        df = df[df["energy"] >= min_energy]

    return df


def assign_window_label(energies, windows):
    labels = pd.Series([None] * len(energies), index=energies.index, dtype="object")
    for label, (emin, emax) in windows.items():
        mask = (energies >= emin) & (energies <= emax)
        labels.loc[mask] = float(label)
    return labels


def compute_stats_binned(df, cols, windows, energy_col="energy"):
    if energy_col not in df.columns:
        raise KeyError(f"No se encontró la columna {energy_col} para binning por energía")

    df = df.copy()
    df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")
    df["energy_bin"] = assign_window_label(df[energy_col], windows)

    mean_dict = {col: {} for col in cols}
    std_dict = {col: {} for col in cols}

    for energy in windows.keys():
        sub = df[df["energy_bin"] == float(energy)]
        for col in cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            mean_dict[col][float(energy)] = float(vals.mean()) if len(vals) else np.nan
            std_dict[col][float(energy)] = float(vals.std()) if len(vals) else np.nan

    return mean_dict, std_dict, df


def build_summary_table(tb_energy_bins, tb_mean, tb_std, sim_mean, sim_std, cols):
    rows = []
    for energy in tb_energy_bins:
        row = {"energy": energy}
        for col in cols:
            row[f"{col}_mean_tb"] = tb_mean[col].get(energy, np.nan)
            row[f"{col}_std_tb"] = tb_std[col].get(energy, np.nan)
            row[f"{col}_mean_sim"] = sim_mean[col].get(energy, np.nan)
            row[f"{col}_std_sim"] = sim_std[col].get(energy, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def relative_diff(tb, sim):
    out = []
    for a, b in zip(tb, sim):
        if pd.isna(a) or pd.isna(b) or b == 0:
            out.append(np.nan)
        else:
            out.append((a - b) / b)
    return out


def get_vals(summary_df, energies, tb_col, sim_col):
    tb_vals = []
    sim_vals = []

    for e in energies:
        row = summary_df[summary_df["energy"] == e]
        tb_vals.append(row[tb_col].iloc[0] if len(row) else np.nan)
        sim_vals.append(row[sim_col].iloc[0] if len(row) else np.nan)

    return tb_vals, sim_vals


def save_mean_hits_plot(summary_df, particle, output_dir):
    energies = [e for e in PLOT_LABEL_ENERGIES if e in summary_df["energy"].tolist()]
    hit_defs = [
        ("nHits_total", "nHits_total_mean_tb", "nHits_total_mean_sim", "blue"),
        ("nHits_1", "nHits_1_mean_tb", "nHits_1_mean_sim", "green"),
        ("nHits_2", "nHits_2_mean_tb", "nHits_2_mean_sim", "red"),
        ("nHits_3", "nHits_3_mean_tb", "nHits_3_mean_sim", "orange"),
    ]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    for i, (label, tb_col, sim_col, color) in enumerate(hit_defs):
        tb_vals, sim_vals = get_vals(summary_df, energies, tb_col, sim_col)
        legend_tb = "Test Beam" if i == 0 else None
        legend_sim = "Simulation" if i == 0 else None
        ax1.plot(energies, tb_vals, marker="o", color=color, label=legend_tb)
        ax1.plot(energies, sim_vals, marker="x", linestyle="--", color=color, label=legend_sim)
        ax2.plot(energies, relative_diff(tb_vals, sim_vals), marker="o", color=color, label=label)

    ax1.set_title(f"Mean Hits vs Energy for {PARTICLE_TITLE[particle]}")
    ax1.set_ylabel("Mean Hits")
    ax1.set_xticks(PLOT_LABEL_ENERGIES)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Energy (GeV)")
    ax2.set_ylabel("Relative difference\n(TB - Sim) / Sim")
    ax2.set_xticks(PLOT_LABEL_ENERGIES)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"mean_hits_vs_energy_{particle}.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_single_nhit_plot(summary_df, particle, hit_key, output_dir):
    energies = [e for e in PLOT_LABEL_ENERGIES if e in summary_df["energy"].tolist()]
    tb_col = f"{hit_key}_mean_tb"
    sim_col = f"{hit_key}_mean_sim"
    tb_vals, sim_vals = get_vals(summary_df, energies, tb_col, sim_col)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(energies, tb_vals, marker="o", color="blue", label="Test Beam")
    ax1.plot(energies, sim_vals, marker="x", linestyle="--", color="blue", label="Simulation")
    ax1.set_title(f"Mean {hit_key} vs Energy for {PARTICLE_TITLE[particle]}")
    ax1.set_ylabel(f"Mean {hit_key}")
    ax1.set_xticks(PLOT_LABEL_ENERGIES)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(energies, relative_diff(tb_vals, sim_vals), marker="o", color="blue")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Energy (GeV)")
    ax2.set_ylabel("Relative difference\n(TB - Sim) / Sim")
    ax2.set_xticks(PLOT_LABEL_ENERGIES)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"mean_{hit_key}_{particle}.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_feature_plot(summary_df, particle, feature, title, output_dir):
    energies = [e for e in PLOT_LABEL_ENERGIES if e in summary_df["energy"].tolist()]
    tb_col = f"{feature}_mean_tb"
    sim_col = f"{feature}_mean_sim"
    tb_vals, sim_vals = get_vals(summary_df, energies, tb_col, sim_col)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(energies, tb_vals, marker="o", color="blue", label="Test Beam")
    ax1.plot(energies, sim_vals, marker="x", linestyle="--", color="blue", label="Simulation")
    ax1.set_title(f"Mean {title} vs Energy for {PARTICLE_TITLE[particle]}")
    ax1.set_ylabel(f"Mean {title}")
    ax1.set_xticks(PLOT_LABEL_ENERGIES)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(energies, relative_diff(tb_vals, sim_vals), marker="o", color="blue")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Energy (GeV)")
    ax2.set_ylabel("Relative difference\n(TB - Sim) / Sim")
    ax2.set_xticks(PLOT_LABEL_ENERGIES)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"mean_{feature}_{particle}.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_hist_plot_for_key(df_tb_binned, df_sim_binned, particle, key, output_dir):
    energies = PLOT_LABEL_ENERGIES
    fig, axes = plt.subplots(1, len(energies), figsize=(5 * len(energies), 4), squeeze=False)
    axes = axes[0]

    for ax, e in zip(axes, energies):
        tb_vals = pd.to_numeric(
            df_tb_binned[df_tb_binned["energy_bin"] == float(e)][key], errors="coerce"
        ).dropna().to_numpy()
        sim_vals = pd.to_numeric(
            df_sim_binned[df_sim_binned["energy_bin"] == float(e)][key], errors="coerce"
        ).dropna().to_numpy()

        if len(tb_vals) == 0 and len(sim_vals) == 0:
            ax.set_title(f"Energy: {e:.1f} GeV\n(no data)")
            continue

        combined = (
            np.concatenate([tb_vals, sim_vals])
            if len(tb_vals) and len(sim_vals)
            else (tb_vals if len(tb_vals) else sim_vals)
        )

        lo, hi = float(np.min(combined)), float(np.max(combined))
        bins = np.linspace(lo, hi, 40) if lo < hi else 20

        if len(tb_vals):
            ax.hist(tb_vals, bins=bins, density=True, alpha=0.75, color="blue", label="Test Beam")

        if len(sim_vals):
            ax.hist(
                sim_vals,
                bins=bins,
                density=True,
                histtype="step",
                linestyle="--",
                linewidth=1.6,
                color="gray",
                label="Simulation",
            )

        ax.set_title(f"Energy: {e:.1f} GeV")
        ax.set_xlabel(key)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"hist_{key}_{particle}.png", dpi=150, bbox_inches="tight")
    plt.close()


def load_triplet(csv_paths, h5_paths, min_density=0):
    parts = []

    for csv_path, h5_path in zip(csv_paths, h5_paths):
        energy = infer_energy_from_path(csv_path)
        if energy is None:
            energy = infer_energy_from_path(h5_path)
        if energy is None:
            raise ValueError(f"No puedo inferir si {csv_path} / {h5_path} es 20, 50 o 80 GeV")

        df = add_reconstructed_features(csv_path, h5_path, forced_energy=energy)
        df = filter_reconstructed(df, min_density=min_density)
        parts.append(df)

    return pd.concat(parts, ignore_index=True)


def process_particle(tb_csvs, tb_h5s, sim_csv, sim_h5, particle, output_dir):
    tb_df = load_triplet(tb_csvs, tb_h5s, min_density=7)
    sim_df_full = add_reconstructed_features(sim_csv, sim_h5, forced_energy=None)
    sim_df_full = filter_reconstructed(sim_df_full, min_density=0)

    cols = ["nHits_total", "nHits_1", "nHits_2", "nHits_3", "lateral", "longitudinal"]

    tb_mean, tb_std, tb_binned = compute_stats_binned(tb_df, cols, SIM_WINDOWS)
    sim_mean, sim_std, sim_binned = compute_stats_binned(sim_df_full, cols, SIM_WINDOWS)

    summary = build_summary_table(PLOT_LABEL_ENERGIES, tb_mean, tb_std, sim_mean, sim_std, cols)
    summary.to_csv(Path(output_dir) / f"summary_{particle}.csv", index=False)
    tb_binned.to_csv(Path(output_dir) / f"tb_reconstructed_{particle}.csv", index=False)
    sim_df_full.to_csv(Path(output_dir) / f"sim_reconstructed_{particle}.csv", index=False)

    counts = []
    for e, (emin, emax) in SIM_WINDOWS.items():
        tb_n = int((tb_binned["energy_bin"] == float(e)).sum())
        sim_n = int((sim_binned["energy_bin"] == float(e)).sum())
        counts.append(
            {
                "energy_label": e,
                "window_min": emin,
                "window_max": emax,
                "tb_events_in_window": tb_n,
                "sim_events_in_window": sim_n,
            }
        )
    pd.DataFrame(counts).to_csv(Path(output_dir) / f"counts_{particle}.csv", index=False)

    save_mean_hits_plot(summary, particle, output_dir)
    save_single_nhit_plot(summary, particle, "nHits_1", output_dir)
    save_single_nhit_plot(summary, particle, "nHits_2", output_dir)
    save_single_nhit_plot(summary, particle, "nHits_3", output_dir)

    save_hist_plot_for_key(tb_binned, sim_binned, particle, "nHits_total", output_dir)
    save_hist_plot_for_key(tb_binned, sim_binned, particle, "nHits_1", output_dir)
    save_hist_plot_for_key(tb_binned, sim_binned, particle, "nHits_2", output_dir)
    save_hist_plot_for_key(tb_binned, sim_binned, particle, "nHits_3", output_dir)

    save_feature_plot(summary, particle, "lateral", "Lateral", output_dir)
    save_feature_plot(summary, particle, "longitudinal", "Longitudinal", output_dir)


def main():
    ap = argparse.ArgumentParser(
        description="Reconstruye variables físicas de TB y simulación desde CSV+H5 y compara ambos usando ventanas de energía 18-22, 48-52 y 78-82 GeV."
    )

    ap.add_argument("--tb-electron-csv", nargs=3, required=True, metavar=("E20CSV", "E50CSV", "E80CSV"))
    ap.add_argument("--tb-electron-h5", nargs=3, required=True, metavar=("E20H5", "E50H5", "E80H5"))
    ap.add_argument("--tb-pion-csv", nargs=3, required=True, metavar=("P20CSV", "P50CSV", "P80CSV"))
    ap.add_argument("--tb-pion-h5", nargs=3, required=True, metavar=("P20H5", "P50H5", "P80H5"))
    ap.add_argument("--sim-electron-csv", required=True)
    ap.add_argument("--sim-electron-h5", required=True)
    ap.add_argument("--sim-pion-csv", required=True)
    ap.add_argument("--sim-pion-h5", required=True)
    ap.add_argument("--output-dir", default="plots_tb_vs_sim_windows")

    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    process_particle(
        args.tb_electron_csv,
        args.tb_electron_h5,
        args.sim_electron_csv,
        args.sim_electron_h5,
        "electron",
        output_dir,
    )

    process_particle(
        args.tb_pion_csv,
        args.tb_pion_h5,
        args.sim_pion_csv,
        args.sim_pion_h5,
        "pion",
        output_dir,
    )

    print("[DONE] Output in", output_dir)


if __name__ == "__main__":
    main()
