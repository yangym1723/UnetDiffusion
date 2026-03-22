import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a robomimic-style rgb/depth frame pair from the same demo "
            "and optionally browse demos and frames with the keyboard."
        )
    )
    parser.add_argument("--hdf5", type=Path, required=True, help="Input HDF5 path.")
    demo_group = parser.add_mutually_exclusive_group()
    demo_group.add_argument(
        "--demo-index",
        type=int,
        default=0,
        help="Demo index under /data, e.g. 0 for demo_0.",
    )
    demo_group.add_argument(
        "--demo-name",
        type=str,
        default=None,
        help="Explicit demo group name under /data, e.g. demo_12.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index within the selected demo.",
    )
    parser.add_argument(
        "--rgb-key",
        type=str,
        default="camera_rgb",
        help="Observation key for the rgb image.",
    )
    parser.add_argument(
        "--depth-key",
        type=str,
        default="camera_depth",
        help="Observation key for the processed depth image.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help=(
            "Optional output path. In static mode this is the saved image path. "
            "In interactive mode, a directory or filename prefix for snapshots."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it in static mode.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open an interactive viewer with keyboard controls.",
    )
    return parser.parse_args()


def sorted_demo_names(data_group: h5py.Group):
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def resolve_demo_name(data_group: h5py.Group, demo_name: str, demo_index: int) -> str:
    if demo_name is not None:
        if demo_name not in data_group:
            raise KeyError(f"Demo {demo_name} not found in /data.")
        return demo_name
    demo_names = sorted_demo_names(data_group)
    if demo_index < 0 or demo_index >= len(demo_names):
        raise IndexError(
            f"demo_index={demo_index} is out of range. There are {len(demo_names)} demos."
        )
    return demo_names[demo_index]


def normalize_hwc(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        return frame[..., None]
    if frame.ndim != 3:
        raise ValueError(f"Unsupported frame shape {frame.shape}.")

    if frame.shape[-1] in (1, 3, 4):
        return frame
    if frame.shape[0] in (1, 3, 4):
        return np.moveaxis(frame, 0, -1)
    raise ValueError(f"Cannot infer channel dimension for shape {frame.shape}.")


def prepare_rgb_for_display(frame: np.ndarray) -> np.ndarray:
    frame = normalize_hwc(frame)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]

    if frame.dtype == np.uint8:
        return frame

    frame = frame.astype(np.float32)
    if frame.max() <= 1.0:
        frame = np.clip(frame, 0.0, 1.0)
    else:
        frame = np.clip(frame / 255.0, 0.0, 1.0)
    return frame


def prepare_depth_for_display(frame: np.ndarray) -> np.ndarray:
    frame = normalize_hwc(frame)
    if frame.shape[-1] == 1:
        return frame[..., 0]
    return frame[..., :3].mean(axis=-1)


def array_stats(array: np.ndarray):
    array = np.asarray(array)
    return {
        "shape": array.shape,
        "dtype": str(array.dtype),
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
    }


def print_stats(name: str, stats: dict):
    print(
        f"{name}: shape={stats['shape']}, dtype={stats['dtype']}, "
        f"min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}"
    )


def build_default_save_path(base_path: Path, demo_name: str, frame_index: int) -> Path:
    stem = base_path.stem
    filename = f"{stem}_{demo_name}_frame_{frame_index:06d}_rgb_depth_check.png"
    return base_path.parent / filename


class RgbDepthInspector:
    def __init__(self, args, plt):
        self.args = args
        self.plt = plt
        self.file = h5py.File(args.hdf5, "r")
        self.data_group = self.file["data"]
        self.demo_names = sorted_demo_names(self.data_group)
        selected_demo_name = resolve_demo_name(
            self.data_group, args.demo_name, args.demo_index
        )
        self.demo_pos = self.demo_names.index(selected_demo_name)
        self.frame_index = args.frame_index
        self.colorbar = None

        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.fig.text(
            0.5,
            0.01,
            "Left/Right: frame  Up/Down: demo  PageUp/PageDown: +/-10 frames  Home/End: first/last frame  s: save  h: help  q: quit",
            ha="center",
            fontsize=9,
        )

    @property
    def demo_name(self):
        return self.demo_names[self.demo_pos]

    def get_obs_group(self):
        return self.data_group[self.demo_name]["obs"]

    def get_frame_count(self):
        obs_group = self.get_obs_group()
        return int(obs_group[self.args.rgb_key].shape[0])

    def clamp_frame_index(self):
        frame_count = self.get_frame_count()
        self.frame_index = max(0, min(self.frame_index, frame_count - 1))

    def get_current_frames(self):
        obs_group = self.get_obs_group()
        if self.args.rgb_key not in obs_group:
            raise KeyError(f"RGB key {self.args.rgb_key} not found in {self.demo_name}/obs.")
        if self.args.depth_key not in obs_group:
            raise KeyError(
                f"Depth key {self.args.depth_key} not found in {self.demo_name}/obs."
            )

        rgb_dataset = obs_group[self.args.rgb_key]
        depth_dataset = obs_group[self.args.depth_key]
        if int(depth_dataset.shape[0]) != int(rgb_dataset.shape[0]):
            raise ValueError(
                f"Frame count mismatch in {self.demo_name}: "
                f"{self.args.rgb_key} has {rgb_dataset.shape[0]}, "
                f"{self.args.depth_key} has {depth_dataset.shape[0]}."
            )

        self.clamp_frame_index()
        return rgb_dataset[self.frame_index], depth_dataset[self.frame_index]

    def save_current_view(self):
        if self.args.save_path is None:
            out_path = build_default_save_path(self.args.hdf5, self.demo_name, self.frame_index)
        else:
            save_path = self.args.save_path
            if save_path.suffix:
                out_path = save_path.with_name(
                    f"{save_path.stem}_{self.demo_name}_frame_{self.frame_index:06d}{save_path.suffix}"
                )
            else:
                out_path = save_path / (
                    f"{self.demo_name}_frame_{self.frame_index:06d}_rgb_depth_check.png"
                )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out_path, dpi=150)
        print(f"Saved comparison image to: {out_path}")

    def print_help(self):
        print("Keyboard controls:")
        print("  Left / Right : previous / next frame")
        print("  PageUp / PageDown : jump -10 / +10 frames")
        print("  Home / End : first / last frame in current demo")
        print("  Up / Down : previous / next demo")
        print("  s : save the current comparison image")
        print("  h : print this help")
        print("  q or Esc : quit")

    def refresh(self, print_current_stats=True):
        rgb_frame_raw, depth_frame_raw = self.get_current_frames()
        rgb_stats = array_stats(rgb_frame_raw)
        depth_stats = array_stats(depth_frame_raw)

        if print_current_stats:
            print(f"Current view: {self.demo_name}, frame {self.frame_index}")
            print_stats("rgb_frame_raw", rgb_stats)
            print_stats("depth_frame_raw", depth_stats)
            if depth_frame_raw.ndim == 3:
                depth_hwc = normalize_hwc(depth_frame_raw)
                if depth_hwc.shape[-1] >= 3:
                    identical_channels = (
                        np.array_equal(depth_hwc[..., 0], depth_hwc[..., 1])
                        and np.array_equal(depth_hwc[..., 1], depth_hwc[..., 2])
                    )
                    print(f"depth_rgb_channels_identical={identical_channels}")

        rgb_display = prepare_rgb_for_display(rgb_frame_raw)
        depth_display = prepare_depth_for_display(depth_frame_raw)

        for ax in self.axes:
            ax.clear()

        self.axes[0].imshow(rgb_display)
        self.axes[0].set_title(
            f"{self.demo_name} frame {self.frame_index}\n"
            f"{self.args.rgb_key} | dtype={rgb_stats['dtype']} | shape={rgb_stats['shape']}"
        )
        self.axes[0].axis("off")

        depth_vmin = depth_stats["min"]
        depth_vmax = depth_stats["max"]
        im = self.axes[1].imshow(
            depth_display, cmap="gray", vmin=depth_vmin, vmax=depth_vmax
        )
        self.axes[1].set_title(
            f"{self.demo_name} frame {self.frame_index}\n"
            f"{self.args.depth_key} | dtype={depth_stats['dtype']} | shape={depth_stats['shape']}"
        )
        self.axes[1].axis("off")

        if self.colorbar is not None:
            self.colorbar.remove()
        self.colorbar = self.fig.colorbar(im, ax=self.axes[1], fraction=0.046, pad=0.04)

        self.fig.suptitle(
            f"RGB vs Processed Depth\n"
            f"rgb[{rgb_stats['min']:.2f}, {rgb_stats['max']:.2f}]  "
            f"depth[{depth_stats['min']:.2f}, {depth_stats['max']:.2f}]",
            fontsize=12,
        )
        self.fig.canvas.draw_idle()

    def move_demo(self, delta: int):
        self.demo_pos = max(0, min(self.demo_pos + delta, len(self.demo_names) - 1))
        self.clamp_frame_index()
        self.refresh(print_current_stats=True)

    def move_frame(self, delta: int):
        self.frame_index += delta
        self.clamp_frame_index()
        self.refresh(print_current_stats=True)

    def on_key_press(self, event):
        key = (event.key or "").lower()
        if key in ("left", "a"):
            self.move_frame(-1)
        elif key in ("right", "d"):
            self.move_frame(1)
        elif key == "pageup":
            self.move_frame(-10)
        elif key == "pagedown":
            self.move_frame(10)
        elif key == "home":
            self.frame_index = 0
            self.refresh(print_current_stats=True)
        elif key == "end":
            self.frame_index = self.get_frame_count() - 1
            self.refresh(print_current_stats=True)
        elif key in ("up", "w"):
            self.move_demo(-1)
        elif key in ("down", "n"):
            self.move_demo(1)
        elif key == "s":
            self.save_current_view()
        elif key in ("h", "?"):
            self.print_help()
        elif key in ("q", "escape"):
            self.plt.close(self.fig)

    def on_close(self, _event):
        if self.file is not None:
            self.file.close()
            self.file = None


def run_static(args, plt):
    inspector = RgbDepthInspector(args, plt)
    inspector.refresh(print_current_stats=True)

    if args.save_path is None:
        out_path = build_default_save_path(args.hdf5, inspector.demo_name, inspector.frame_index)
    else:
        out_path = args.save_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inspector.fig.savefig(out_path, dpi=150)
    print(f"Saved comparison image to: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(inspector.fig)
        inspector.on_close(None)


def run_interactive(args, plt):
    inspector = RgbDepthInspector(args, plt)
    inspector.print_help()
    inspector.refresh(print_current_stats=True)
    plt.show()


def main():
    args = parse_args()
    if not args.hdf5.exists():
        raise FileNotFoundError(args.hdf5)

    if not args.interactive:
        import matplotlib

        if not args.show:
            matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if args.interactive:
        run_interactive(args, plt)
    else:
        run_static(args, plt)


if __name__ == "__main__":
    main()
