import os
import argparse
import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Split HDF5 dataset by events into train/test preserving schema")
    p.add_argument("--input", required=True, help="Input .h5 file")
    p.add_argument("--train-output", required=True, help="Output train .h5")
    p.add_argument("--test-output", required=True, help="Output test .h5")
    p.add_argument("--train-frac", type=float, default=0.8, help="Training fraction, default 0.8")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument("--shuffle", action="store_true", help="Shuffle events before split")
    return p.parse_args()


def copy_attrs(src_obj, dst_obj):
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v


def classify_root_datasets(f, offsets):
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


def build_event_masks(offsets, selected_events):
    n_events = len(offsets) - 1
    n_hits_total = int(offsets[-1])

    event_mask = np.zeros(n_events, dtype=bool)
    event_mask[selected_events] = True

    hit_mask = np.zeros(n_hits_total, dtype=bool)
    for ev in selected_events:
        a = int(offsets[ev])
        b = int(offsets[ev + 1])
        hit_mask[a:b] = True

    return event_mask, hit_mask


def recompute_offsets(offsets, selected_events):
    lengths = []
    for ev in selected_events:
        lengths.append(int(offsets[ev + 1] - offsets[ev]))
    new_offsets = np.zeros(len(selected_events) + 1, dtype=np.int64)
    if lengths:
        new_offsets[1:] = np.cumsum(lengths)
    return new_offsets


def write_split_file(input_path, output_path, selected_events):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        copy_attrs(fin, fout)

        offsets = fin["offsets"][:]
        hit_keys, event_keys, other_keys = classify_root_datasets(fin, offsets)

        event_mask, hit_mask = build_event_masks(offsets, selected_events)
        new_offsets = recompute_offsets(offsets, selected_events)

        for key in fin.keys():
            obj = fin[key]

            if not isinstance(obj, h5py.Dataset):
                fin.copy(obj, fout, name=key)
                continue

            if key == "offsets":
                ds = fout.create_dataset("offsets", data=new_offsets)
                copy_attrs(obj, ds)

            elif key in hit_keys:
                data = obj[hit_mask]
                ds = fout.create_dataset(key, data=data, dtype=obj.dtype)
                copy_attrs(obj, ds)

            elif key in event_keys:
                data = obj[event_mask]
                ds = fout.create_dataset(key, data=data, dtype=obj.dtype)
                copy_attrs(obj, ds)

            else:
                data = obj[()]
                ds = fout.create_dataset(key, data=data, dtype=obj.dtype)
                copy_attrs(obj, ds)


def main():
    args = parse_args()

    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("--train-frac must be between 0 and 1")

    with h5py.File(args.input, "r") as f:
        if "offsets" not in f:
            raise KeyError("El archivo no tiene la clave 'offsets'")

        offsets = f["offsets"][:]
        n_events = len(offsets) - 1
        keys = list(f.keys())

    print("Keys:", keys)
    print(f"Eventos totales: {n_events}")

    indices = np.arange(n_events)
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(indices)

    n_train = int(np.floor(args.train_frac * n_events))
    train_events = np.sort(indices[:n_train])
    test_events = np.sort(indices[n_train:])

    print(f"Train events: {len(train_events)}")
    print(f"Test events: {len(test_events)}")

    write_split_file(args.input, args.train_output, train_events)
    write_split_file(args.input, args.test_output, test_events)

    with h5py.File(args.train_output, "r") as ftr:
        train_hits = int(ftr["offsets"][-1])
        train_events_out = len(ftr["offsets"]) - 1

    with h5py.File(args.test_output, "r") as fte:
        test_hits = int(fte["offsets"][-1])
        test_events_out = len(fte["offsets"]) - 1

    print("\nResumen")
    print(f"  Train: {train_events_out} eventos, {train_hits} hits")
    print(f"  Test:  {test_events_out} eventos, {test_hits} hits")
    print(f"  Train file: {args.train_output}")
    print(f"  Test file:  {args.test_output}")


if __name__ == "__main__":
    main()
