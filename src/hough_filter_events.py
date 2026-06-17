import os
import argparse
import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Hough cleaning preserving original HDF5 schema")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pad-x", type=float, default=1.0)
    p.add_argument("--pad-y", type=float, default=1.0)
    p.add_argument("--pad-z", type=float, default=1.0)
    p.add_argument("--z-as-layer", action="store_true")
    p.add_argument("--theta-bins", type=int, default=360)
    p.add_argument("--rho-bins", type=int, default=800)
    p.add_argument("--band-pads", type=int, default=2)
    p.add_argument("--min-hits", type=int, default=4)
    p.add_argument("--min-occupied", type=int, default=3)
    p.add_argument("--min-votes", type=int, default=4)
    p.add_argument("--keep-fallback", choices=["original", "empty"], default="original")
    p.add_argument("--debug-every", type=int, default=1000)
    return p.parse_args()


def copy_attrs(src_obj, dst_obj):
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v


def classify_datasets(f, offsets):
    n_events = len(offsets) - 1
    n_hits_total = int(offsets[-1])

    hit_keys = []
    event_keys = []
    other_keys = []

    for key in f.keys():
        obj = f[key]
        if not isinstance(obj, h5py.Dataset):
            other_keys.append(key)
            continue

        if key == "offsets":
            continue

        shape = obj.shape
        if len(shape) == 0:
            other_keys.append(key)
            continue

        if shape[0] == n_hits_total:
            hit_keys.append(key)
        elif shape[0] == n_events:
            event_keys.append(key)
        else:
            other_keys.append(key)

    return hit_keys, event_keys, other_keys


def choose_coordinate_keys(hit_keys):
    if {"x", "y", "k"}.issubset(hit_keys):
        return ("x", "y", "k")
    if {"x", "y", "z"}.issubset(hit_keys):
        return ("x", "y", "z")
    if {"i", "j", "k"}.issubset(hit_keys):
        return ("i", "j", "k")
    raise KeyError("No encuentro un triplete de coordenadas compatible entre los datasets por-hit")


def quantize_hits(hits, pad_x, pad_y, pad_z, z_as_layer):
    x_idx = np.rint(hits[:, 0] / pad_x).astype(np.int32)
    y_idx = np.rint(hits[:, 1] / pad_y).astype(np.int32)
    z_idx = np.rint(hits[:, 2]).astype(np.int32) if z_as_layer else np.rint(hits[:, 2] / pad_z).astype(np.int32)
    return x_idx, y_idx, z_idx


def unique_occupied_pairs(a_idx, z_idx):
    pairs = np.column_stack([a_idx, z_idx])
    uniq = np.unique(pairs, axis=0)
    return uniq[:, 0], uniq[:, 1]


def hough_line_from_points(a, z, theta_bins=360, rho_bins=800):
    if len(a) < 2:
        return None

    a = a.astype(np.float64)
    z = z.astype(np.float64)

    thetas = np.linspace(-np.pi / 2, np.pi / 2, theta_bins, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    rho_max = np.sqrt(np.max(a * a + z * z)) + 1e-6
    rhos = np.linspace(-rho_max, rho_max, rho_bins)
    acc = np.zeros((rho_bins, theta_bins), dtype=np.int32)

    dr = (rhos[-1] - rhos[0]) / (rho_bins - 1)

    for ai, zi in zip(a, z):
        rho_vals = ai * cos_t + zi * sin_t
        rho_idx = np.rint((rho_vals - rhos[0]) / dr).astype(int)
        rho_idx = np.clip(rho_idx, 0, rho_bins - 1)
        acc[rho_idx, np.arange(theta_bins)] += 1

    best_rho_idx, best_theta_idx = np.unravel_index(np.argmax(acc), acc.shape)

    return {
        "rho": float(rhos[best_rho_idx]),
        "theta": float(thetas[best_theta_idx]),
        "votes": int(acc[best_rho_idx, best_theta_idx]),
    }


def point_line_distance_2d(a, z, rho, theta):
    return np.abs(a * np.cos(theta) + z * np.sin(theta) - rho)


def clean_event_with_hough(coord_hits, args):
    n_hits = len(coord_hits)

    if n_hits < args.min_hits:
        if args.keep_fallback == "empty":
            return np.zeros(n_hits, dtype=bool), {
                "reason": "few_hits",
                "votes_xz": 0,
                "votes_yz": 0,
                "kept_frac": 0.0,
            }
        return np.ones(n_hits, dtype=bool), {
            "reason": "few_hits",
            "votes_xz": 0,
            "votes_yz": 0,
            "kept_frac": 1.0,
        }

    x_idx, y_idx, z_idx = quantize_hits(coord_hits, args.pad_x, args.pad_y, args.pad_z, args.z_as_layer)

    x_occ, z_occ_x = unique_occupied_pairs(x_idx, z_idx)
    y_occ, z_occ_y = unique_occupied_pairs(y_idx, z_idx)

    if len(x_occ) < args.min_occupied or len(y_occ) < args.min_occupied:
        if args.keep_fallback == "empty":
            return np.zeros(n_hits, dtype=bool), {
                "reason": "few_occupied",
                "votes_xz": 0,
                "votes_yz": 0,
                "kept_frac": 0.0,
            }
        return np.ones(n_hits, dtype=bool), {
            "reason": "few_occupied",
            "votes_xz": 0,
            "votes_yz": 0,
            "kept_frac": 1.0,
        }

    model_xz = hough_line_from_points(x_occ, z_occ_x, args.theta_bins, args.rho_bins)
    model_yz = hough_line_from_points(y_occ, z_occ_y, args.theta_bins, args.rho_bins)

    vx = 0 if model_xz is None else model_xz["votes"]
    vy = 0 if model_yz is None else model_yz["votes"]

    if model_xz is None and model_yz is None:
        if args.keep_fallback == "empty":
            return np.zeros(n_hits, dtype=bool), {
                "reason": "no_line",
                "votes_xz": 0,
                "votes_yz": 0,
                "kept_frac": 0.0,
            }
        return np.ones(n_hits, dtype=bool), {
            "reason": "no_line",
            "votes_xz": 0,
            "votes_yz": 0,
            "kept_frac": 1.0,
        }

    if model_xz is None or model_yz is None or vx < args.min_votes or vy < args.min_votes:
        if args.keep_fallback == "empty":
            return np.zeros(n_hits, dtype=bool), {
                "reason": "weak_line",
                "votes_xz": vx,
                "votes_yz": vy,
                "kept_frac": 0.0,
            }
        return np.ones(n_hits, dtype=bool), {
            "reason": "weak_line",
            "votes_xz": vx,
            "votes_yz": vy,
            "kept_frac": 1.0,
        }

    dist_xz = point_line_distance_2d(
        x_idx.astype(np.float64), z_idx.astype(np.float64),
        model_xz["rho"], model_xz["theta"]
    )
    dist_yz = point_line_distance_2d(
        y_idx.astype(np.float64), z_idx.astype(np.float64),
        model_yz["rho"], model_yz["theta"]
    )

    mask = (dist_xz <= args.band_pads) & (dist_yz <= args.band_pads)

    if mask.sum() == 0:
        if args.keep_fallback == "empty":
            return np.zeros(n_hits, dtype=bool), {
                "reason": "empty_band",
                "votes_xz": vx,
                "votes_yz": vy,
                "kept_frac": 0.0,
            }
        return np.ones(n_hits, dtype=bool), {
            "reason": "empty_band",
            "votes_xz": vx,
            "votes_yz": vy,
            "kept_frac": 1.0,
        }

    return mask, {
        "reason": "ok",
        "votes_xz": vx,
        "votes_yz": vy,
        "kept_frac": float(mask.mean()),
    }


def main():
    args = parse_args()

    print(f"Cargando: {args.input}")

    with h5py.File(args.input, "r") as fin:
        input_keys = list(fin.keys())
        print("Keys:", input_keys)

        offsets = fin["offsets"][:]
        copy_attrs(fin, fin)

        hit_keys, event_keys, other_keys = classify_datasets(fin, offsets)
        coord_keys = choose_coordinate_keys(hit_keys)

        print("Datasets por-hit:", hit_keys)
        print("Datasets por-evento:", event_keys)
        if other_keys:
            print("Otros objetos/datasets no estándar:", other_keys)

        n_events = len(offsets) - 1
        n_hits_total = int(offsets[-1])

        hit_data = {k: fin[k][:] for k in hit_keys}
        event_data = {k: fin[k][:] for k in event_keys}

        filtered_hit_chunks = {k: [] for k in hit_keys}
        new_offsets = np.zeros(n_events + 1, dtype=np.int64)

        kept_fracs = []
        reason_counts = {}

        cx, cy, cz = coord_keys

        for i in range(n_events):
            a = int(offsets[i])
            b = int(offsets[i + 1])

            coord_hits = np.column_stack([
                hit_data[cx][a:b],
                hit_data[cy][a:b],
                hit_data[cz][a:b],
            ])

            mask, info = clean_event_with_hough(coord_hits, args)

            for k in hit_keys:
                filtered_hit_chunks[k].append(hit_data[k][a:b][mask])

            new_offsets[i + 1] = new_offsets[i] + int(mask.sum())

            kept_fracs.append(info["kept_frac"])
            reason_counts[info["reason"]] = reason_counts.get(info["reason"], 0) + 1

            if args.debug_every > 0 and i % args.debug_every == 0:
                print(
                    f"evt={i} hits={b-a} kept={int(mask.sum())} "
                    f"reason={info['reason']} votes_xz={info['votes_xz']} "
                    f"votes_yz={info['votes_yz']} kept_frac={info['kept_frac']:.3f}"
                )

    print(f"Guardando: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with h5py.File(args.input, "r") as fin, h5py.File(args.output, "w") as fout:
        copy_attrs(fin, fout)

        for key in fin.keys():
            obj = fin[key]

            if not isinstance(obj, h5py.Dataset):
                fin.copy(obj, fout, name=key)
                continue

            if key == "offsets":
                ds = fout.create_dataset("offsets", data=new_offsets)
                copy_attrs(obj, ds)

            elif key in hit_keys:
                if len(filtered_hit_chunks[key]) > 0:
                    arr = np.concatenate(filtered_hit_chunks[key], axis=0)
                else:
                    arr = np.empty((0,), dtype=obj.dtype)

                ds = fout.create_dataset(key, data=arr, dtype=obj.dtype)
                copy_attrs(obj, ds)

            elif key in event_keys:
                ds = fout.create_dataset(key, data=event_data[key], dtype=obj.dtype)
                copy_attrs(obj, ds)

            else:
                fin.copy(obj, fout, name=key)

    kept_fracs = np.array(kept_fracs, dtype=float)
    print("\nResumen")
    print(f"  Eventos procesados: {n_events}")
    print(f"  Hits totales originales: {n_hits_total}")
    print(f"  Hits totales filtrados: {int(new_offsets[-1])}")
    print(f"  Fracción media conservada: {kept_fracs.mean():.3f}")
    print(f"  Fracción mediana conservada: {np.median(kept_fracs):.3f}")
    print(f"  reasons: {reason_counts}")


if __name__ == "__main__":
    main()
