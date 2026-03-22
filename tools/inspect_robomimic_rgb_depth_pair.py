import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


ARROW_LEFT_CODES = {81, 2424832, 65361}
ARROW_UP_CODES = {82, 2490368, 65362}
ARROW_RIGHT_CODES = {83, 2555904, 65363}
ARROW_DOWN_CODES = {84, 2621440, 65364}
PAGE_UP_CODES = {2162688, 65365}
PAGE_DOWN_CODES = {2228224, 65366}
HOME_CODES = {2359296, 65360}
END_CODES = {2293760, 65367}
ESC_CODES = {27}


FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (240, 240, 240)
TEXT_BG = (20, 20, 20)
SEPARATOR_COLOR = (70, 70, 70)
PANEL_BG = (18, 18, 18)
WINDOW_NAME = "rgb_depth_inspector"


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
        help="Display the composed image window in static mode.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open an interactive OpenCV viewer with keyboard controls.",
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
        rgb = frame
    else:
        frame = frame.astype(np.float32)
        if frame.max() <= 1.0:
            frame = np.clip(frame, 0.0, 1.0)
            rgb = np.round(frame * 255.0).astype(np.uint8)
        else:
            rgb = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def prepare_depth_for_display(frame: np.ndarray) -> np.ndarray:
    frame = normalize_hwc(frame)
    if frame.shape[-1] == 1:
        depth_gray = frame[..., 0]
    else:
        depth_gray = frame[..., 0]

    if depth_gray.dtype != np.uint8:
        depth_gray = np.clip(depth_gray, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)


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


def put_text_lines(img: np.ndarray, lines, origin_x: int, origin_y: int, line_gap: int = 22):
    y = origin_y
    for line in lines:
        cv2.putText(
            img,
            line,
            (origin_x, y),
            FONT,
            0.55,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        y += line_gap


class RgbDepthInspector:
    def __init__(self, args):
        self.args = args
        self.file = h5py.File(args.hdf5, "r")
        self.data_group = self.file["data"]
        self.demo_names = sorted_demo_names(self.data_group)
        selected_demo_name = resolve_demo_name(
            self.data_group, args.demo_name, args.demo_index
        )
        self.demo_pos = self.demo_names.index(selected_demo_name)
        self.frame_index = args.frame_index

    @property
    def demo_name(self):
        return self.demo_names[self.demo_pos]

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

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

    def save_current_view(self, canvas: np.ndarray):
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
        cv2.imwrite(str(out_path), canvas)
        print(f"Saved comparison image to: {out_path}")

    def print_help(self):
        print("Keyboard controls:")
        print("  Left / a : previous frame")
        print("  Right / d : next frame")
        print("  PageUp : -10 frames")
        print("  PageDown : +10 frames")
        print("  Home / End : first / last frame in current demo")
        print("  Up / w : previous demo")
        print("  Down / n : next demo")
        print("  s : save the current comparison image")
        print("  h : print this help")
        print("  q or Esc : quit")

    def render_current_view(self, print_current_stats=True):
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

        rgb_panel = prepare_rgb_for_display(rgb_frame_raw)
        depth_panel = prepare_depth_for_display(depth_frame_raw)

        height = max(rgb_panel.shape[0], depth_panel.shape[0])
        width = rgb_panel.shape[1] + depth_panel.shape[1] + 16
        header_height = 92
        footer_height = 90
        canvas = np.full((height + header_height + footer_height, width, 3), PANEL_BG, dtype=np.uint8)

        rgb_y = header_height
        depth_y = header_height
        canvas[rgb_y:rgb_y + rgb_panel.shape[0], 0:rgb_panel.shape[1]] = rgb_panel
        depth_x = rgb_panel.shape[1] + 16
        canvas[depth_y:depth_y + depth_panel.shape[0], depth_x:depth_x + depth_panel.shape[1]] = depth_panel
        canvas[:, rgb_panel.shape[1] + 8:rgb_panel.shape[1] + 9] = SEPARATOR_COLOR

        header_lines = [
            f"{self.demo_name} | frame {self.frame_index}/{self.get_frame_count() - 1}",
            f"rgb key={self.args.rgb_key} | shape={rgb_stats['shape']} | dtype={rgb_stats['dtype']} | min={rgb_stats['min']:.1f} max={rgb_stats['max']:.1f} mean={rgb_stats['mean']:.1f}",
            f"depth key={self.args.depth_key} | shape={depth_stats['shape']} | dtype={depth_stats['dtype']} | min={depth_stats['min']:.1f} max={depth_stats['max']:.1f} mean={depth_stats['mean']:.1f}",
        ]
        footer_lines = [
            "Keys: Left/Right frame | Up/Down demo | PageUp/PageDown +/-10 | Home/End first/last | s save | h help | q quit",
        ]
        put_text_lines(canvas, header_lines, origin_x=12, origin_y=28)
        put_text_lines(canvas, footer_lines, origin_x=12, origin_y=height + header_height + 30)

        cv2.putText(canvas, "RGB", (12, header_height - 12), FONT, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Depth", (depth_x + 12, header_height - 12), FONT, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)

        return canvas

    def move_demo(self, delta: int):
        self.demo_pos = max(0, min(self.demo_pos + delta, len(self.demo_names) - 1))
        self.clamp_frame_index()

    def move_frame(self, delta: int):
        self.frame_index += delta
        self.clamp_frame_index()


class KeyAction:
    def __init__(self, quit_requested=False, save_requested=False, help_requested=False, demo_delta=0, frame_delta=0, set_home=False, set_end=False):
        self.quit_requested = quit_requested
        self.save_requested = save_requested
        self.help_requested = help_requested
        self.demo_delta = demo_delta
        self.frame_delta = frame_delta
        self.set_home = set_home
        self.set_end = set_end


def decode_key(key_code: int) -> KeyAction:
    if key_code < 0:
        return KeyAction()

    if key_code in ESC_CODES or key_code in (ord("q"), ord("Q")):
        return KeyAction(quit_requested=True)
    if key_code in (ord("s"), ord("S")):
        return KeyAction(save_requested=True)
    if key_code in (ord("h"), ord("H"), ord("?")):
        return KeyAction(help_requested=True)
    if key_code in ARROW_LEFT_CODES or key_code in (ord("a"), ord("A")):
        return KeyAction(frame_delta=-1)
    if key_code in ARROW_RIGHT_CODES or key_code in (ord("d"), ord("D")):
        return KeyAction(frame_delta=1)
    if key_code in PAGE_UP_CODES:
        return KeyAction(frame_delta=-10)
    if key_code in PAGE_DOWN_CODES:
        return KeyAction(frame_delta=10)
    if key_code in HOME_CODES:
        return KeyAction(set_home=True)
    if key_code in END_CODES:
        return KeyAction(set_end=True)
    if key_code in ARROW_UP_CODES or key_code in (ord("w"), ord("W")):
        return KeyAction(demo_delta=-1)
    if key_code in ARROW_DOWN_CODES or key_code in (ord("n"), ord("N")):
        return KeyAction(demo_delta=1)
    return KeyAction()


def run_static(args):
    inspector = RgbDepthInspector(args)
    try:
        canvas = inspector.render_current_view(print_current_stats=True)
        if args.save_path is None:
            out_path = build_default_save_path(args.hdf5, inspector.demo_name, inspector.frame_index)
        else:
            out_path = args.save_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas)
        print(f"Saved comparison image to: {out_path}")

        if args.show:
            cv2.imshow(WINDOW_NAME, canvas)
            while True:
                key = cv2.waitKeyEx(0)
                action = decode_key(key)
                if action.quit_requested or key >= 0:
                    break
            cv2.destroyAllWindows()
    finally:
        inspector.close()


def run_interactive(args):
    inspector = RgbDepthInspector(args)
    cv2.setNumThreads(1)
    inspector.print_help()
    try:
        while True:
            canvas = inspector.render_current_view(print_current_stats=True)
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKeyEx(0)
            action = decode_key(key)

            if action.help_requested:
                inspector.print_help()
                continue
            if action.save_requested:
                inspector.save_current_view(canvas)
                continue
            if action.quit_requested:
                break
            if action.set_home:
                inspector.frame_index = 0
                continue
            if action.set_end:
                inspector.frame_index = inspector.get_frame_count() - 1
                continue
            if action.demo_delta != 0:
                inspector.move_demo(action.demo_delta)
                continue
            if action.frame_delta != 0:
                inspector.move_frame(action.frame_delta)
                continue
    finally:
        inspector.close()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    if not args.hdf5.exists():
        raise FileNotFoundError(args.hdf5)

    if args.interactive:
        run_interactive(args)
    else:
        run_static(args)


if __name__ == "__main__":
    main()
