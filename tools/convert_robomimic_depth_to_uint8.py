import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a robomimic-style depth observation into a uint8 image "
            "that diffusion_policy can consume as an rgb modality."
        )
    )
    parser.add_argument("--src", type=Path, required=True, help="Source HDF5 path.")
    parser.add_argument("--dst", type=Path, required=True, help="Output HDF5 path.")
    parser.add_argument(
        "--depth-key",
        type=str,
        required=True,
        help="Observation key under demo/obs that contains the float32 depth frames.",
    )
    parser.add_argument(
        "--output-key",
        type=str,
        required=True,
        help=(
            "Observation key under demo/obs to write the converted uint8 image. "
            "It can be the same as an existing image key if you want to overwrite it."
        ),
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=None,
        help="Fixed minimum depth used for clipping and normalization.",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=None,
        help="Fixed maximum depth used for clipping and normalization.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the grayscale mapping after normalization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output observation key in the copied file.",
    )
    parser.add_argument(
        "--repeat-channels",
        type=int,
        default=3,
        help=(
            "Number of channels in the output image. Use 3 to treat depth as a "
            "pseudo-rgb camera in the current hybrid/image pipelines."
        ),
    )
    return parser.parse_args()


def sorted_demo_names(data_group: h5py.Group):
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def squeeze_depth_frames(depth_frames: np.ndarray) -> np.ndarray:
    if depth_frames.ndim == 3:
        return depth_frames
    if depth_frames.ndim == 4 and depth_frames.shape[-1] == 1:
        return depth_frames[..., 0]
    if depth_frames.ndim == 4 and depth_frames.shape[1] == 1:
        return depth_frames[:, 0]
    raise ValueError(
        f"Unsupported depth array shape {depth_frames.shape}. "
        "Expected (T,H,W), (T,H,W,1), or (T,1,H,W)."
    )


def compute_global_minmax(src_file: h5py.File, depth_key: str):
    global_min = None
    global_max = None
    for demo_name in sorted_demo_names(src_file["data"]):
        depth_frames = src_file[f"data/{demo_name}/obs/{depth_key}"][:]
        depth_frames = squeeze_depth_frames(depth_frames)
        this_min = float(np.nanmin(depth_frames))
        this_max = float(np.nanmax(depth_frames))
        global_min = this_min if global_min is None else min(global_min, this_min)
        global_max = this_max if global_max is None else max(global_max, this_max)
    return global_min, global_max


def convert_depth_to_uint8(
    depth_frames: np.ndarray,
    depth_min: float,
    depth_max: float,
    repeat_channels: int,
    invert: bool,
) -> np.ndarray:
    if depth_max <= depth_min:
        raise ValueError(
            f"depth_max ({depth_max}) must be larger than depth_min ({depth_min})."
        )

    depth_frames = squeeze_depth_frames(depth_frames).astype(np.float32)
    depth_frames = np.clip(depth_frames, depth_min, depth_max)
    depth_frames = (depth_frames - depth_min) / (depth_max - depth_min)
    if invert:
        depth_frames = 1.0 - depth_frames

    depth_uint8 = np.round(depth_frames * 255.0).astype(np.uint8)
    if repeat_channels > 1:
        depth_uint8 = np.repeat(depth_uint8[..., None], repeat_channels, axis=-1)
    else:
        depth_uint8 = depth_uint8[..., None]
    return depth_uint8


def main():
    args = parse_args()

    if args.repeat_channels < 1:
        raise ValueError("--repeat-channels must be at least 1.")
    if not args.src.exists():
        raise FileNotFoundError(args.src)
    if args.dst.exists() and args.dst.resolve() == args.src.resolve():
        raise ValueError("--dst must be different from --src.")

    args.dst.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.src, "r") as src_file:
        depth_min = args.depth_min
        depth_max = args.depth_max
        if depth_min is None or depth_max is None:
            auto_min, auto_max = compute_global_minmax(src_file, args.depth_key)
            if depth_min is None:
                depth_min = auto_min
            if depth_max is None:
                depth_max = auto_max

        with h5py.File(args.dst, "w") as dst_file:
            for key in src_file.keys():
                src_file.copy(key, dst_file)
            for key, value in src_file.attrs.items():
                dst_file.attrs[key] = value

            for demo_name in sorted_demo_names(dst_file["data"]):
                obs_group = dst_file[f"data/{demo_name}/obs"]
                if args.output_key in obs_group:
                    if not args.overwrite:
                        raise ValueError(
                            f"{args.output_key} already exists in {demo_name}. "
                            "Pass --overwrite to replace it."
                        )
                    del obs_group[args.output_key]

                src_depth = src_file[f"data/{demo_name}/obs/{args.depth_key}"][:]
                depth_uint8 = convert_depth_to_uint8(
                    depth_frames=src_depth,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    repeat_channels=args.repeat_channels,
                    invert=args.invert,
                )
                output_dataset = obs_group.create_dataset(
                    args.output_key,
                    data=depth_uint8,
                    dtype=np.uint8,
                    compression="gzip",
                )
                output_dataset.attrs["source_depth_key"] = args.depth_key
                output_dataset.attrs["depth_min"] = float(depth_min)
                output_dataset.attrs["depth_max"] = float(depth_max)
                output_dataset.attrs["invert"] = bool(args.invert)

    print(
        "Converted depth observation",
        f"{args.depth_key} -> {args.output_key}",
        f"with range [{depth_min}, {depth_max}]",
        f"and repeat_channels={args.repeat_channels}.",
    )


if __name__ == "__main__":
    main()
