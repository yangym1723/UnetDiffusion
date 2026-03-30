#!/usr/bin/env python3
"""
Count mode statistics in a robomimic-style HDF5 dataset.

Example:
python tools/count_robomimic_mode_stats.py --src data/dataset_H480W640_128.hdf5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


TARGET_VALUES = (0.0, 1.0, 2.0)


def _demo_sort_key(name: str):
    try:
        return int(name.split("_")[-1])
    except Exception:
        return name


def _value_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _constant_mode(arr: np.ndarray, atol: float) -> str | None:
    if arr.size == 0:
        return None
    for value in TARGET_VALUES:
        if np.all(np.isclose(arr, value, atol=atol)):
            return _value_label(value)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to the HDF5 dataset.")
    parser.add_argument("--group", default="data", help="Root group containing demos.")
    parser.add_argument("--obs-key", default="mode", help="Observation key to inspect.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Tolerance for float comparisons.")
    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        raise FileNotFoundError(f"Dataset not found: {src_path}")

    frame_counts = {_value_label(v): 0 for v in TARGET_VALUES}
    strict_constant_demo_counts = {_value_label(v): 0 for v in TARGET_VALUES}
    tail_constant_demo_counts = {_value_label(v): 0 for v in TARGET_VALUES}
    strict_mixed_demos: list[str] = []
    tail_mixed_demos: list[str] = []
    missing_mode_demos: list[str] = []
    unknown_value_frame_count = 0

    num_demos = 0
    num_demos_with_mode = 0
    total_frames = 0

    with h5py.File(src_path, "r") as f:
        if args.group not in f:
            raise KeyError(f"Group '{args.group}' not found in {src_path}")

        demos = sorted(f[args.group].keys(), key=_demo_sort_key)
        num_demos = len(demos)

        for demo_name in demos:
            demo = f[args.group][demo_name]
            obs_group = demo.get("obs", None)
            if obs_group is None or args.obs_key not in obs_group:
                missing_mode_demos.append(demo_name)
                continue

            num_demos_with_mode += 1
            values = np.asarray(obs_group[args.obs_key][...], dtype=np.float32).reshape(-1)
            total_frames += int(values.size)

            matched_mask = np.zeros(values.shape, dtype=bool)
            for value in TARGET_VALUES:
                value_mask = np.isclose(values, value, atol=args.atol)
                frame_counts[_value_label(value)] += int(value_mask.sum())
                matched_mask |= value_mask
            unknown_value_frame_count += int((~matched_mask).sum())

            strict_mode = _constant_mode(values, args.atol)
            if strict_mode is None:
                strict_mixed_demos.append(demo_name)
            else:
                strict_constant_demo_counts[strict_mode] += 1

            tail_values = values[1:] if values.size > 1 else values
            tail_mode = _constant_mode(tail_values, args.atol)
            if tail_mode is None:
                tail_mixed_demos.append(demo_name)
            else:
                tail_constant_demo_counts[tail_mode] += 1

    result = {
        "dataset": str(src_path),
        "group": args.group,
        "obs_key": args.obs_key,
        "num_demos": num_demos,
        "num_demos_with_mode": num_demos_with_mode,
        "num_demos_missing_mode": len(missing_mode_demos),
        "total_frames": total_frames,
        "frame_value_counts": frame_counts,
        "unknown_value_frame_count": unknown_value_frame_count,
        "strict_constant_demo_counts": strict_constant_demo_counts,
        "strict_mixed_demo_count": len(strict_mixed_demos),
        "tail_constant_demo_counts_ignore_first_frame": tail_constant_demo_counts,
        "tail_mixed_demo_count_ignore_first_frame": len(tail_mixed_demos),
    }

    print(json.dumps(result, indent=2, sort_keys=True))

    if missing_mode_demos:
        print("\nMissing mode demos:")
        for name in missing_mode_demos[:20]:
            print(name)
        if len(missing_mode_demos) > 20:
            print(f"... and {len(missing_mode_demos) - 20} more")

    if strict_mixed_demos:
        print("\nStrict mixed demos (sample):")
        for name in strict_mixed_demos[:20]:
            print(name)
        if len(strict_mixed_demos) > 20:
            print(f"... and {len(strict_mixed_demos) - 20} more")

    if tail_mixed_demos:
        print("\nTail mixed demos after dropping first frame (sample):")
        for name in tail_mixed_demos[:20]:
            print(name)
        if len(tail_mixed_demos) > 20:
            print(f"... and {len(tail_mixed_demos) - 20} more")


if __name__ == "__main__":
    main()
