import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

OUTDIR = "raw_nhits_and_eventdisplays"
os.makedirs(OUTDIR, exist_ok=True)

H5_FILES = {
    "electron": {
        20: "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam2/electrones_20_test.h5",
        50: "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam2/electrones_50_test.h5",
        80: "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam2/electrones_80_test.h5",
    },
    "pion": {
        20: "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam2/piones_20_test.h5",
        50: "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam2/piones_50_test.h5",
        80: "/home/vmellado/FQM378/vmellado/GATrEnv/data/testbeam2/piones_80_test.h5",
    }
}

ENERGIES = [20, 50, 80]
BINS = 120
XLIM = (0, 1700)
N_EVENTS_PER_FIG = 5

PEAK_WINDOWS = {
    "noise": (0, 30),
    "muon": (45, 110),

    "electron_20": (160, 260),
    "electron_50": (300, 470),
    "electron_80": (480, 700),

    "pion_20": (220, 420),
    "pion_50": (620, 900),
    "pion_80": (950, 1300),
}

# ==========================================================
# LOADERS
# ==========================================================

def load_offsets(h5file):
    with h5py.File(h5file, "r") as f:
        return f["offsets"][:]

def load_nhits(h5file):
    offsets = load_offsets(h5file)
    return np.diff(offsets)

def load_event(h5file, event_id):
    with h5py.File(h5file, "r") as f:
        x = f["x"][:]
        y = f["y"][:]
        z = f["z"][:]
        offsets = f["offsets"][:]

        start = offsets[event_id]
        end = offsets[event_id + 1]

        evt_x = x[start:end]
        evt_y = y[start:end]
        evt_z = z[start:end]

    return evt_x, evt_y, evt_z

# ==========================================================
# LAYER MAPPING
# ==========================================================

def z_to_layer_indices(evt_z):
    unique_z = np.unique(evt_z)
    unique_z = np.sort(unique_z)
    z_to_layer = {zv: i + 1 for i, zv in enumerate(unique_z)}
    layers = np.array([z_to_layer[v] for v in evt_z])
    return layers

def get_layer_ticks(max_layer):
    if max_layer <= 6:
        return list(range(1, max_layer + 1))
    elif max_layer <= 12:
        return list(range(1, max_layer + 1, 2))
    elif max_layer <= 24:
        return list(range(1, max_layer + 1, 4))
    else:
        return list(range(1, max_layer + 1, 6))

# ==========================================================
# RAW NHITS PLOTS
# ==========================================================

def plot_nhits_overlay_all_energies(outdir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, energy in zip(axes, ENERGIES):
        nh_e = load_nhits(H5_FILES["electron"][energy])
        nh_p = load_nhits(H5_FILES["pion"][energy])

        ax.hist(
            nh_e,
            bins=BINS,
            range=XLIM,
            histtype="step",
            linewidth=2.2,
            color="crimson",
            label=f"electron dataset",
            log=True
        )

        ax.hist(
            nh_p,
            bins=BINS,
            range=XLIM,
            histtype="step",
            linewidth=2.2,
            color="royalblue",
            label=f"pion dataset",
            log=True
        )

        ax.set_title(f"Raw Nhits distribution - {energy} GeV")
        ax.set_xlabel("Nhits")
        ax.set_xlim(*XLIM)
        ax.grid(alpha=0.25)
        ax.legend()

    axes[0].set_ylabel("Events")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "raw_nhits_overlay_20_50_80.png"), dpi=300)
    plt.close()

def plot_nhits_single_energy(energy, outdir):
    nh_e = load_nhits(H5_FILES["electron"][energy])
    nh_p = load_nhits(H5_FILES["pion"][energy])

    plt.figure(figsize=(10, 7))

    plt.hist(
        nh_e,
        bins=BINS,
        range=XLIM,
        histtype="step",
        linewidth=2.3,
        color="crimson",
        label=f"electron {energy} GeV",
        log=True
    )

    plt.hist(
        nh_p,
        bins=BINS,
        range=XLIM,
        histtype="step",
        linewidth=2.3,
        color="royalblue",
        label=f"pion {energy} GeV",
        log=True
    )

    plt.xlabel("Nhits")
    plt.ylabel("Events")
    plt.title(f"Raw Nhits distribution - electron vs pion - {energy} GeV")
    plt.xlim(*XLIM)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"raw_nhits_overlay_{energy}GeV.png"), dpi=300)
    plt.close()

# ==========================================================
# EVENT SELECTION
# ==========================================================

def pick_multiple_events(h5file, nhit_min, nhit_max, n=5):
    nhits = load_nhits(h5file)
    ids = np.where((nhits >= nhit_min) & (nhits < nhit_max))[0]

    if len(ids) == 0:
        return []

    if len(ids) <= n:
        return [(int(i), int(nhits[i])) for i in ids]

    idx = np.linspace(0, len(ids) - 1, n).astype(int)
    chosen = ids[idx]

    return [(int(i), int(nhits[i])) for i in chosen]

# ==========================================================
# EVENT DISPLAY GALLERY
# ==========================================================

def plot_event_gallery(h5file, event_list, main_title, outfile):
    if len(event_list) == 0:
        print(f"No events found for {outfile}")
        return

    nrows = len(event_list)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.3 * nrows))

    if nrows == 1:
        axes = np.array([axes])

    for i, (event_id, nh) in enumerate(event_list):
        evt_x, evt_y, evt_z = load_event(h5file, event_id)
        layers = z_to_layer_indices(evt_z)

        max_layer = int(layers.max())
        xticks = get_layer_ticks(max_layer)

        ax1 = axes[i, 0]
        ax1.scatter(layers, evt_x, s=5, alpha=0.75)
        ax1.set_xlabel("Number of layer")
        ax1.set_ylabel("X [cm]")
        ax1.set_title(f"Event {event_id}, Number of hits = {nh} - XZ", fontsize=10)
        ax1.set_xlim(0.5, max_layer + 0.5)
        ax1.set_xticks(xticks)
        ax1.grid(alpha=0.2)

        ax2 = axes[i, 1]
        ax2.scatter(layers, evt_y, s=5, alpha=0.75)
        ax2.set_xlabel("Number of layer")
        ax2.set_ylabel("Y [cm]")
        ax2.set_title(f"Event {event_id}, Number of hits = {nh} - YZ", fontsize=10)
        ax2.set_xlim(0.5, max_layer + 0.5)
        ax2.set_xticks(xticks)
        ax2.grid(alpha=0.2)

    fig.suptitle(main_title, fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

# ==========================================================
# GALLERIES
# ==========================================================

def generate_event_galleries(outdir):
    events = pick_multiple_events(
        H5_FILES["electron"][80],
        *PEAK_WINDOWS["noise"],
        n=N_EVENTS_PER_FIG
    )
    plot_event_gallery(
        H5_FILES["electron"][80],
        events,
        "Peak 1: noise-like events",
        os.path.join(outdir, "peak1_noise_like_gallery.png")
    )

    events = pick_multiple_events(
        H5_FILES["pion"][80],
        *PEAK_WINDOWS["noise"],
        n=N_EVENTS_PER_FIG
    )
    plot_event_gallery(
        H5_FILES["pion"][80],
        events,
        "Peak 1: cosmic-like events",
        os.path.join(outdir, "peak1_cosmic_like_gallery.png")
    )

    events = pick_multiple_events(
        H5_FILES["pion"][80],
        *PEAK_WINDOWS["muon"],
        n=N_EVENTS_PER_FIG
    )
    h5_mu = H5_FILES["pion"][80]

    if len(events) == 0:
        events = pick_multiple_events(
            H5_FILES["electron"][80],
            *PEAK_WINDOWS["muon"],
            n=N_EVENTS_PER_FIG
        )
        h5_mu = H5_FILES["electron"][80]

    plot_event_gallery(
        h5_mu,
        events,
        "Peak 2: muon-like events",
        os.path.join(outdir, "peak2_muon_like_gallery.png")
    )

    for energy in ENERGIES:
        key = f"electron_{energy}"
        events = pick_multiple_events(
            H5_FILES["electron"][energy],
            *PEAK_WINDOWS[key],
            n=N_EVENTS_PER_FIG
        )
        plot_event_gallery(
            H5_FILES["electron"][energy],
            events,
            f"Electron shower events - {energy} GeV",
            os.path.join(outdir, f"electron_{energy}_shower_gallery.png")
        )

    for energy in ENERGIES:
        key = f"pion_{energy}"
        events = pick_multiple_events(
            H5_FILES["pion"][energy],
            *PEAK_WINDOWS[key],
            n=N_EVENTS_PER_FIG
        )
        plot_event_gallery(
            H5_FILES["pion"][energy],
            events,
            f"Pion shower events - {energy} GeV",
            os.path.join(outdir, f"pion_{energy}_shower_gallery.png")
        )

# ==========================================================
# MAIN
# ==========================================================

def main():
    print("===================================")
    print("RAW NHITS DISTRIBUTIONS")
    print("===================================")

    plot_nhits_overlay_all_energies(OUTDIR)

    for energy in ENERGIES:
        plot_nhits_single_energy(energy, OUTDIR)

    print("===================================")
    print("EVENT DISPLAY GALLERIES")
    print("===================================")

    generate_event_galleries(OUTDIR)

    print("===================================")
    print("ALL OUTPUTS SAVED")
    print("===================================")
    print(f"Saved in: {OUTDIR}")

if __name__ == "__main__":
    main()
