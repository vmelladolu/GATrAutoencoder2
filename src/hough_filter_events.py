import os
import argparse
import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Hough cleaning on SDHCAL pads')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--pad-x', type=float, default=1.0)
    p.add_argument('--pad-y', type=float, default=1.0)
    p.add_argument('--pad-z', type=float, default=1.0)
    p.add_argument('--z-as-layer', action='store_true')
    p.add_argument('--theta-bins', type=int, default=360)
    p.add_argument('--rho-bins', type=int, default=800)
    p.add_argument('--band-pads', type=int, default=2)
    p.add_argument('--min-hits', type=int, default=4)
    p.add_argument('--min-occupied', type=int, default=3)
    p.add_argument('--min-votes', type=int, default=4)
    p.add_argument('--keep-fallback', choices=['original', 'empty'], default='original')
    p.add_argument('--debug-every', type=int, default=1000)
    return p.parse_args()


def load_events(path):
    with h5py.File(path, 'r') as f:
        print('Keys:', list(f.keys()))
        offsets = f['offsets'][:]
        if {'x', 'y', 'k'}.issubset(f.keys()):
            hits = np.column_stack([f['x'][:], f['y'][:], f['k'][:]])
            mode = 'xyz'
        elif 'showers' in f.keys():
            hits = f['showers'][:]
            mode = 'showers'
        else:
            raise KeyError('No encuentro x,y,z ni showers')
        other = {}
        for k in f.keys():
            if k in {'x','y','k','showers','offsets'}:
                continue
            try:
                other[k] = f[k][:]
            except Exception:
                pass
    events = [hits[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
    return events, other, mode


def save_events(path, events, other=None, mode='xyz'):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    lengths = [len(ev) for ev in events]
    offsets = np.zeros(len(events)+1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    all_hits = np.concatenate(events, axis=0) if len(events) else np.empty((0,3), dtype=np.float32)
    with h5py.File(path, 'w') as f:
        if mode == 'xyz':
            f.create_dataset('x', data=all_hits[:,0])
            f.create_dataset('y', data=all_hits[:,1])
            f.create_dataset('z', data=all_hits[:,2])
        else:
            f.create_dataset('showers', data=all_hits)
        f.create_dataset('offsets', data=offsets)
        if other is not None:
            for k,v in other.items():
                try:
                    if len(v) == len(events):
                        f.create_dataset(k, data=v)
                except Exception:
                    pass


def quantize_hits(hits, pad_x, pad_y, pad_z, z_as_layer):
    x_idx = np.rint(hits[:,0] / pad_x).astype(np.int32)
    y_idx = np.rint(hits[:,1] / pad_y).astype(np.int32)
    z_idx = np.rint(hits[:,2]).astype(np.int32) if z_as_layer else np.rint(hits[:,2] / pad_z).astype(np.int32)
    return x_idx, y_idx, z_idx


def unique_occupied_pairs(a_idx, z_idx):
    pairs = np.column_stack([a_idx, z_idx])
    uniq = np.unique(pairs, axis=0)
    return uniq[:,0], uniq[:,1]


def hough_line_from_points(a, z, theta_bins=360, rho_bins=800):
    if len(a) < 2:
        return None
    a = a.astype(np.float64)
    z = z.astype(np.float64)
    thetas = np.linspace(-np.pi/2, np.pi/2, theta_bins, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    rho_max = np.sqrt(np.max(a*a + z*z)) + 1e-6
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
        'rho': float(rhos[best_rho_idx]),
        'theta': float(thetas[best_theta_idx]),
        'votes': int(acc[best_rho_idx, best_theta_idx]),
        'accumulator': acc
    }


def point_line_distance_2d(a, z, rho, theta):
    return np.abs(a * np.cos(theta) + z * np.sin(theta) - rho)


def clean_event_with_hough(hits, args):
    n_hits = len(hits)
    if n_hits < args.min_hits:
        if args.keep_fallback == 'empty':
            return np.empty((0,3), dtype=hits.dtype), {'reason':'few_hits','votes_xz':0,'votes_yz':0,'kept_frac':0.0}
        return hits, {'reason':'few_hits','votes_xz':0,'votes_yz':0,'kept_frac':1.0}

    x_idx, y_idx, z_idx = quantize_hits(hits, args.pad_x, args.pad_y, args.pad_z, args.z_as_layer)
    x_occ, z_occ_x = unique_occupied_pairs(x_idx, z_idx)
    y_occ, z_occ_y = unique_occupied_pairs(y_idx, z_idx)

    model_xz = hough_line_from_points(x_occ, z_occ_x, args.theta_bins, args.rho_bins)
    model_yz = hough_line_from_points(y_occ, z_occ_y, args.theta_bins, args.rho_bins)

    if model_xz is None and model_yz is None:
        if args.keep_fallback == 'empty':
            return np.empty((0,3), dtype=hits.dtype), {'reason':'no_line','votes_xz':0,'votes_yz':0,'kept_frac':0.0}
        return hits, {'reason':'no_line','votes_xz':0,'votes_yz':0,'kept_frac':1.0}

    if model_xz is None:
        if args.keep_fallback == 'empty':
            return np.empty((0,3), dtype=hits.dtype), {'reason':'no_xz','votes_xz':0,'votes_yz':model_yz['votes'],'kept_frac':0.0}
        return hits, {'reason':'no_xz','votes_xz':0,'votes_yz':model_yz['votes'],'kept_frac':1.0}

    if model_yz is None:
        if args.keep_fallback == 'empty':
            return np.empty((0,3), dtype=hits.dtype), {'reason':'no_yz','votes_xz':model_xz['votes'],'votes_yz':0,'kept_frac':0.0}
        return hits, {'reason':'no_yz','votes_xz':model_xz['votes'],'votes_yz':0,'kept_frac':1.0}

    dist_xz = point_line_distance_2d(
        x_idx.astype(np.float64), z_idx.astype(np.float64),
        model_xz['rho'], model_xz['theta']
    )
    dist_yz = point_line_distance_2d(
        y_idx.astype(np.float64), z_idx.astype(np.float64),
        model_yz['rho'], model_yz['theta']
    )

    mask = (dist_xz <= args.band_pads) & (dist_yz <= args.band_pads)
    if mask.sum() == 0:
        if args.keep_fallback == 'empty':
            return np.empty((0,3), dtype=hits.dtype), {'reason':'empty_band','votes_xz':model_xz['votes'],'votes_yz':model_yz['votes'],'kept_frac':0.0}
        return hits, {'reason':'empty_band','votes_xz':model_xz['votes'],'votes_yz':model_yz['votes'],'kept_frac':1.0}

    filtered = hits[mask]
    return filtered, {'reason':'ok','votes_xz':model_xz['votes'],'votes_yz':model_yz['votes'],'kept_frac':len(filtered)/len(hits)}


def main():
    args = parse_args()
    print(f'Cargando: {args.input}')
    events, other, mode = load_events(args.input)
    filtered_events = []
    kept_fracs = []
    reason_counts = {}

    for i, ev in enumerate(events):
        ev_filtered, info = clean_event_with_hough(ev, args)
        filtered_events.append(ev_filtered)
        kept_fracs.append(info['kept_frac'])
        reason_counts[info['reason']] = reason_counts.get(info['reason'], 0) + 1

        if args.debug_every > 0 and i % args.debug_every == 0:
            print(f"evt={i} hits={len(ev)} kept={len(ev_filtered)} reason={info['reason']} votes_xz={info['votes_xz']} votes_yz={info['votes_yz']} kept_frac={info['kept_frac']:.3f}")

    print(f'Guardando: {args.output}')
    save_events(args.output, filtered_events, other=other, mode=mode)

    kept_fracs = np.array(kept_fracs, dtype=float)
    print('\\nResumen')
    print(f'  Eventos procesados: {len(events)}')
    print(f'  Fracción media conservada: {kept_fracs.mean():.3f}')
    print(f'  Fracción mediana conservada: {np.median(kept_fracs):.3f}')
    print(f'  reasons: {reason_counts}')


if __name__ == '__main__':
    main()
