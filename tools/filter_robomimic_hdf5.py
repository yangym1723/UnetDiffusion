import argparse
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter a robomimic-style HDF5 file and write a smaller subset. "
            "Supports demo selection, per-demo timestep slicing, and key filtering."
        )
    )
    parser.add_argument("--src", type=Path, required=True, help="Source HDF5 path.")
    parser.add_argument("--dst", type=Path, required=True, help="Output HDF5 path.")

    parser.add_argument(
        "--demo-names",
        nargs="*",
        default=None,
        help="Explicit demo names to keep, e.g. demo_0 demo_3 demo_10.",
    )
    parser.add_argument(
        "--demo-start",
        type=int,
        default=0,
        help="Start demo index in sorted order, inclusive.",
    )
    parser.add_argument(
        "--demo-end",
        type=int,
        default=None,
        help="End demo index in sorted order, exclusive. Defaults to all demos.",
    )
    parser.add_argument(
        "--demo-stride",
        type=int,
        default=1,
        help="Keep every N-th demo in sorted order.",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Maximum number of demos to keep after demo filtering.",
    )

    parser.add_argument(
        "--step-start",
        type=int,
        default=0,
        help="Start timestep within each kept demo, inclusive.",
    )
    parser.add_argument(
        "--step-end",
        type=int,
        default=None,
        help="End timestep within each kept demo, exclusive. Defaults to demo length.",
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=1,
        help="Keep every N-th timestep within each kept demo.",
    )
    parser.add_argument(
        "--max-steps-per-demo",
        type=int,
        default=None,
        help="Maximum number of timesteps to keep per demo after timestep slicing.",
    )

    parser.add_argument(
        "--keep-obs-keys",
        nargs="*",
        default=None,
        help="Observation keys under demo/obs to keep. Defaults to all obs keys.",
    )
    parser.add_argument(
        "--drop-obs-keys",
        nargs="*",
        default=None,
        help="Observation keys under demo/obs to remove after keep filtering.",
    )
    parser.add_argument(
        "--keep-demo-keys",
        nargs="*",
        default=None,
        help=(
            "Direct children under each demo group to keep, e.g. "
            "actions processed_actions obs states initial_state."
        ),
    )
    parser.add_argument(
        "--drop-demo-keys",
        nargs="*",
        default=None,
        help="Direct children under each demo group to remove after keep filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the filtering summary without writing the output file.",
    )
    return parser.parse_args()


def sorted_demo_names(data_group: h5py.Group):
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def copy_attrs(src_obj, dst_obj):
    for key, value in src_obj.attrs.items():
        dst_obj.attrs[key] = value


def normalize_optional_set(values: Optional[Iterable[str]]) -> Optional[set[str]]:
    if values is None:
        return None
    return set(values)


def should_keep_name(
    name: str,
    keep_set: Optional[set[str]],
    drop_set: Optional[set[str]],
) -> bool:
    if keep_set is not None and name not in keep_set:
        return False
    if drop_set is not None and name in drop_set:
        return False
    return True


def resolve_demo_names(data_group: h5py.Group, args) -> list[str]:
    demo_names = sorted_demo_names(data_group)
    if args.demo_names:
        keep = []
        existing = set(demo_names)
        for name in args.demo_names:
            if name not in existing:
                raise KeyError(f"Demo {name} not found in /data.")
            keep.append(name)
        return keep

    if args.demo_stride < 1:
        raise ValueError("--demo-stride must be at least 1.")

    demo_start = max(args.demo_start, 0)
    demo_end = len(demo_names) if args.demo_end is None else args.demo_end
    selected = demo_names[demo_start:demo_end:args.demo_stride]
    if args.max_demos is not None:
        if args.max_demos < 0:
            raise ValueError("--max-demos must be non-negative.")
        selected = selected[: args.max_demos]
    return selected


def build_step_indices(num_samples: int, args) -> np.ndarray:
    if args.step_stride < 1:
        raise ValueError("--step-stride must be at least 1.")

    step_start = max(args.step_start, 0)
    step_end = num_samples if args.step_end is None else min(args.step_end, num_samples)
    if step_end < step_start:
        step_end = step_start

    step_indices = np.arange(step_start, step_end, args.step_stride, dtype=np.int64)
    if args.max_steps_per_demo is not None:
        if args.max_steps_per_demo < 0:
            raise ValueError("--max-steps-per-demo must be non-negative.")
        step_indices = step_indices[: args.max_steps_per_demo]
    return step_indices


def create_dataset_from_source(
    dst_group: h5py.Group,
    name: str,
    src_dataset: h5py.Dataset,
    data: np.ndarray,
):
    kwargs = {}
    if src_dataset.compression is not None:
        kwargs["compression"] = src_dataset.compression
        if src_dataset.compression_opts is not None:
            kwargs["compression_opts"] = src_dataset.compression_opts
    if src_dataset.shuffle:
        kwargs["shuffle"] = True
    if src_dataset.fletcher32:
        kwargs["fletcher32"] = True

    dst_dataset = dst_group.create_dataset(name, data=data, dtype=src_dataset.dtype, **kwargs)
    copy_attrs(src_dataset, dst_dataset)
    return dst_dataset


def copy_demo_item(
    src_obj,
    dst_parent: h5py.Group,
    name: str,
    num_samples: int,
    step_indices: np.ndarray,
    keep_obs_keys: Optional[set[str]],
    drop_obs_keys: Optional[set[str]],
):
    if isinstance(src_obj, h5py.Dataset):
        data = src_obj[:]
        if src_obj.ndim > 0 and src_obj.shape[0] == num_samples:
            data = data[step_indices]
        create_dataset_from_source(dst_parent, name, src_obj, data)
        return

    dst_group = dst_parent.create_group(name)
    copy_attrs(src_obj, dst_group)

    for child_name, child_obj in src_obj.items():
        if name == "obs" and not should_keep_name(child_name, keep_obs_keys, drop_obs_keys):
            continue
        copy_demo_item(
            src_obj=child_obj,
            dst_parent=dst_group,
            name=child_name,
            num_samples=num_samples,
            step_indices=step_indices,
            keep_obs_keys=keep_obs_keys,
            drop_obs_keys=drop_obs_keys,
        )


def maybe_copy_mask_group(
    src_file: h5py.File,
    dst_file: h5py.File,
    kept_demo_names: set[str],
):
    if "mask" not in src_file:
        return

    src_mask = src_file["mask"]
    dst_mask = dst_file.create_group("mask")
    copy_attrs(src_mask, dst_mask)
    for name, obj in src_mask.items():
        if isinstance(obj, h5py.Group):
            nested_group = dst_mask.create_group(name)
            copy_attrs(obj, nested_group)
            for child_name, child_obj in obj.items():
                copy_mask_dataset(child_obj, nested_group, child_name, kept_demo_names)
        else:
            copy_mask_dataset(obj, dst_mask, name, kept_demo_names)


def copy_mask_dataset(
    src_dataset: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    kept_demo_names: set[str],
):
    data = src_dataset[:]
    if data.ndim == 1:
        kept_mask = []
        for value in data:
            if isinstance(value, bytes):
                demo_name = value.decode("utf-8")
            else:
                demo_name = str(value)
            kept_mask.append(demo_name in kept_demo_names)
        data = data[np.asarray(kept_mask, dtype=bool)]
    create_dataset_from_source(dst_group, name, src_dataset, data)


def summarize_demo_lengths(src_file: h5py.File, demo_names: list[str], args):
    kept_lengths = []
    skipped_empty = []
    for demo_name in demo_names:
        src_demo = src_file[f"data/{demo_name}"]
        num_samples = int(src_demo.attrs["num_samples"])
        step_indices = build_step_indices(num_samples, args)
        if len(step_indices) == 0:
            skipped_empty.append(demo_name)
            continue
        kept_lengths.append((demo_name, num_samples, len(step_indices)))
    return kept_lengths, skipped_empty


def print_summary(kept_lengths, skipped_empty):
    original_total = sum(item[1] for item in kept_lengths)
    kept_total = sum(item[2] for item in kept_lengths)
    print(f"Kept demos: {len(kept_lengths)}")
    print(f"Original total steps in kept demos: {original_total}")
    print(f"Filtered total steps: {kept_total}")
    if original_total > 0:
        print(f"Kept ratio: {kept_total / original_total:.4f}")
    if skipped_empty:
        print(
            "Skipped demos with zero kept steps after timestep filtering:",
            ", ".join(skipped_empty[:10]),
            "..." if len(skipped_empty) > 10 else "",
        )


def main():
    args = parse_args()

    if not args.src.exists():
        raise FileNotFoundError(args.src)
    if args.dst.exists() and args.dst.resolve() == args.src.resolve():
        raise ValueError("--dst must be different from --src.")

    keep_obs_keys = normalize_optional_set(args.keep_obs_keys)
    drop_obs_keys = normalize_optional_set(args.drop_obs_keys)
    keep_demo_keys = normalize_optional_set(args.keep_demo_keys)
    drop_demo_keys = normalize_optional_set(args.drop_demo_keys)

    with h5py.File(args.src, "r") as src_file:
        if "data" not in src_file:
            raise KeyError("Expected robomimic-style /data group in source file.")

        selected_demo_names = resolve_demo_names(src_file["data"], args)
        kept_lengths, skipped_empty = summarize_demo_lengths(
            src_file=src_file,
            demo_names=selected_demo_names,
            args=args,
        )
        print_summary(kept_lengths, skipped_empty)

        if args.dry_run:
            return

        args.dst.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.dst, "w") as dst_file:
            copy_attrs(src_file, dst_file)

            for root_name, root_obj in src_file.items():
                if root_name in {"data", "mask"}:
                    continue
                src_file.copy(root_name, dst_file)

            dst_data = dst_file.create_group("data")
            copy_attrs(src_file["data"], dst_data)

            kept_demo_names = set()
            filtered_total = 0
            for demo_name, original_num_samples, kept_num_samples in kept_lengths:
                del original_num_samples
                src_demo = src_file[f"data/{demo_name}"]
                step_indices = build_step_indices(int(src_demo.attrs["num_samples"]), args)
                if len(step_indices) == 0:
                    continue

                dst_demo = dst_data.create_group(demo_name)
                copy_attrs(src_demo, dst_demo)
                dst_demo.attrs["num_samples"] = np.int64(kept_num_samples)

                for child_name, child_obj in src_demo.items():
                    if not should_keep_name(child_name, keep_demo_keys, drop_demo_keys):
                        continue
                    copy_demo_item(
                        src_obj=child_obj,
                        dst_parent=dst_demo,
                        name=child_name,
                        num_samples=int(src_demo.attrs["num_samples"]),
                        step_indices=step_indices,
                        keep_obs_keys=keep_obs_keys,
                        drop_obs_keys=drop_obs_keys,
                    )

                kept_demo_names.add(demo_name)
                filtered_total += kept_num_samples

            dst_data.attrs["total"] = np.int64(filtered_total)
            maybe_copy_mask_group(src_file, dst_file, kept_demo_names)

    print(f"Wrote filtered HDF5 to: {args.dst}")


if __name__ == "__main__":
    main()
