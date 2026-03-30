"""
IsaacLab image runner for diffusion_policy evaluation.

This runner is designed for IsaacLab tasks that expose image observations and
custom low-dimensional observations through scene sensors. It keeps IsaacLab
imports lazy so the training process can instantiate the runner without
requiring IsaacLab at module import time.
"""

from __future__ import annotations

import collections
import os
import pathlib
import sys
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import tqdm

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

try:
    import wandb
    import wandb.sdk.data_types.video as wv
except ImportError:
    wandb = None
    wv = None


class IsaacLabImageRunner(BaseImageRunner):
    """
    Evaluate an image policy inside an IsaacLab environment.

    Notes
    -----
    - IsaacLab must be launched via AppLauncher before importing Isaac Sim /
      IsaacLab modules. The runner therefore initializes the simulation lazily
      on the first call to :meth:`run`.
    - The current policy is queried once every `n_action_steps` environment
      steps and the returned action chunk is executed sequentially. This aligns
      with the stateful action-history RNN used in the current hybrid policy.
    - Depth observations can be converted online into the same pseudo-RGB
      representation used during dataset preprocessing, ensuring train / eval
      consistency.
    """

    def __init__(
        self,
        output_dir: str,
        shape_meta: dict,
        dataset_path: Optional[str] = None,
        task_name: str = "Template-Threefingers-v0",
        threefingers_project_root: Optional[str] = None,
        n_train: int = 0,
        n_train_vis: int = 0,
        train_start_idx: int = 0,
        n_test: int = 10,
        n_test_vis: int = 2,
        test_start_seed: int = 10000,
        max_steps: int = 200,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        render_obs_key: str = "camera_rgb",
        fps: int = 20,
        crf: int = 22,
        past_action: bool = False,
        abs_action: bool = True,
        tqdm_interval_sec: float = 5.0,
        n_envs: int = 1,
        enable_cameras: bool = True,
        headless: bool = True,
        device: str = "cuda:0",
        camera_obs_group: str = "camera",
        camera_sensor_name: str = "camera",
        depth_min: Optional[float] = None,
        depth_max: Optional[float] = None,
        depth_invert: bool = False,
        depth_data_type: str = "distance_to_camera",
        use_initial_image_obs_only: bool = False,
        observe_before_act_steps: int = 0,
        binary_action_indices=None,
        binary_action_low: float = -1.0,
        binary_action_high: float = 1.0,
        binary_action_threshold: float = 0.0,
        quaternion_action_indices=None,
    ):
        super().__init__(output_dir)

        self.shape_meta = shape_meta
        self.dataset_path = dataset_path
        self.task_name = task_name
        self.threefingers_project_root = threefingers_project_root

        self.n_train = n_train
        self.n_train_vis = n_train_vis
        self.train_start_idx = train_start_idx
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.test_start_seed = test_start_seed

        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.render_obs_key = render_obs_key
        self.fps = fps
        self.crf = crf
        self.past_action = past_action
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.n_envs = n_envs
        self.enable_cameras = enable_cameras
        self.headless = headless
        self.sim_device = device
        self.camera_obs_group = camera_obs_group
        self.camera_sensor_name = camera_sensor_name
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_invert = depth_invert
        self.depth_data_type = depth_data_type
        self.use_initial_image_obs_only = use_initial_image_obs_only
        self.observe_before_act_steps = int(observe_before_act_steps)

        self._env = None
        self._simulation_app = None
        self._isaaclab_initialized = False

        self.rgb_keys = [
            key for key, attr in shape_meta["obs"].items()
            if attr.get("type", "low_dim") == "rgb"
        ]
        self.depth_keys = [
            key for key, attr in shape_meta["obs"].items()
            if attr.get("type", "low_dim") == "depth"
        ]
        self.lowdim_keys = [
            key for key, attr in shape_meta["obs"].items()
            if attr.get("type", "low_dim") == "low_dim"
        ]

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        self.action_dim = int(action_shape[0])
        self.binary_action_indices = self._resolve_action_indices(
            binary_action_indices, self.action_dim)
        self.binary_action_low = float(binary_action_low)
        self.binary_action_high = float(binary_action_high)
        self.binary_action_threshold = float(binary_action_threshold)
        self.quaternion_action_indices = self._resolve_action_indices(
            quaternion_action_indices, self.action_dim)

        camera_depth_meta = shape_meta["obs"].get("camera_depth")
        if camera_depth_meta is not None:
            self.depth_target_shape = tuple(camera_depth_meta["shape"])
        else:
            self.depth_target_shape = None

        camera_rgb_meta = shape_meta["obs"].get("camera_rgb")
        if camera_rgb_meta is not None:
            self.rgb_target_shape = tuple(camera_rgb_meta["shape"])
        else:
            self.rgb_target_shape = None

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _as_bool(value) -> bool:
        arr = np.asarray(IsaacLabImageRunner._to_numpy(value))
        return bool(np.any(arr))

    @staticmethod
    def _resolve_action_indices(indices, action_dim):
        if indices is None:
            return []
        resolved = []
        for idx in indices:
            idx = int(idx)
            if idx < 0:
                idx += action_dim
            if idx < 0 or idx >= action_dim:
                raise IndexError(
                    f'action index {idx} is out of bounds for action_dim={action_dim}')
            resolved.append(idx)
        if len(set(resolved)) != len(resolved):
            raise ValueError(f'duplicate action indices are not allowed: {resolved}')
        return resolved

    @staticmethod
    def _resolve_repo_root() -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parents[2]

    def _resolve_threefingers_project_root(self) -> pathlib.Path:
        if self.threefingers_project_root is not None:
            path = pathlib.Path(os.path.expanduser(self.threefingers_project_root)).resolve()
            if not path.exists():
                raise FileNotFoundError(
                    f"threefingers_project_root does not exist: {path}"
                )
            return path

        candidates = [
            os.environ.get("THREEFINGERS_PROJECT_ROOT"),
            self._resolve_repo_root().parent / "ThreeFingers",
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            path = pathlib.Path(candidate).resolve()
            if path.exists():
                return path

        raise FileNotFoundError(
            "Could not locate the ThreeFingers project. Set "
            "`task.env_runner.threefingers_project_root` or the "
            "`THREEFINGERS_PROJECT_ROOT` environment variable."
        )

    def _extend_python_path(self, project_root: pathlib.Path):
        path_candidates = [
            project_root,
            project_root / "config",
            project_root / "source" / "ThreeFingers",
        ]
        for path in path_candidates:
            path_str = str(path.resolve())
            if path.exists() and path_str not in sys.path:
                sys.path.insert(0, path_str)

    def _ensure_isaaclab_initialized(self):
        if self._isaaclab_initialized:
            return

        from isaaclab.app import AppLauncher

        launcher_args = {
            "headless": self.headless,
            "enable_cameras": self.enable_cameras,
        }
        app_launcher = AppLauncher(launcher_args=launcher_args)
        self._simulation_app = app_launcher.app

        project_root = self._resolve_threefingers_project_root()
        self._extend_python_path(project_root)

        import gymnasium as gym
        import ThreeFingers  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        env_cfg = parse_env_cfg(
            self.task_name,
            device=self.sim_device,
            num_envs=1,
        )

        control_hz = 60.0
        env_cfg.episode_length_s = max(
            float(env_cfg.episode_length_s),
            self.max_steps / control_hz + 1.0,
        )

        self._env = gym.make(self.task_name, cfg=env_cfg)
        self._isaaclab_initialized = True

    @staticmethod
    def _resize_image_chw(
            image_chw: np.ndarray,
            target_shape: tuple[int, int, int],
            interpolation=None) -> np.ndarray:
        c, target_h, target_w = target_shape
        if image_chw.shape == target_shape:
            return image_chw

        image_hwc = np.transpose(image_chw, (1, 2, 0))
        if interpolation is None:
            interpolation = cv2.INTER_AREA
            if target_h > image_hwc.shape[0] or target_w > image_hwc.shape[1]:
                interpolation = cv2.INTER_LINEAR
        resized_hwc = cv2.resize(image_hwc, (target_w, target_h), interpolation=interpolation)
        if resized_hwc.ndim == 2:
            resized_hwc = resized_hwc[..., None]
        resized_chw = np.transpose(resized_hwc, (2, 0, 1))
        if resized_chw.shape[0] != c:
            if resized_chw.shape[0] == 1 and c > 1:
                resized_chw = np.repeat(resized_chw, c, axis=0)
            else:
                raise RuntimeError(
                    f"Unexpected channel count after resize: got {resized_chw.shape[0]}, expected {c}"
                )
        return resized_chw.astype(np.float32)

    @staticmethod
    def _ensure_batched_hwc(array: np.ndarray, expected_last_dim: Optional[int] = None) -> np.ndarray:
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4:
            raise RuntimeError(f"Expected image batch with 4 dims, got shape {array.shape}")
        if expected_last_dim is not None and array.shape[-1] != expected_last_dim:
            raise RuntimeError(
                f"Expected last image dimension {expected_last_dim}, got shape {array.shape}"
            )
        return array

    @staticmethod
    def _ensure_batched_depth_hwc(array: np.ndarray) -> np.ndarray:
        if array.ndim == 2:
            array = array[None, ..., None]
        elif array.ndim == 3:
            if array.shape[-1] == 1:
                array = array[None, ...]
            else:
                array = array[..., None]
        if array.ndim != 4:
            raise RuntimeError(f"Expected depth image batch with 4 dims, got shape {array.shape}")
        if array.shape[-1] != 1:
            raise RuntimeError(f"Expected depth image with one channel, got shape {array.shape}")
        return array

    def _prepare_depth_chw(self, depth_raw: np.ndarray) -> np.ndarray:
        if self.depth_target_shape is None:
            raise RuntimeError("camera_depth is not present in shape_meta.")

        depth_raw = self._to_numpy(depth_raw).astype(np.float32)
        if depth_raw.ndim == 2:
            depth = depth_raw
        elif depth_raw.ndim == 3 and depth_raw.shape[-1] == 1:
            depth = depth_raw[..., 0]
        else:
            raise RuntimeError(f"Expected depth image with one channel, got shape {depth_raw.shape}")
        if self.depth_min is not None and self.depth_max is not None:
            if self.depth_max <= self.depth_min:
                raise RuntimeError(
                    f"depth_max ({self.depth_max}) must be greater than depth_min ({self.depth_min})."
                )
            depth = np.clip(depth, self.depth_min, self.depth_max)
            depth = (depth - self.depth_min) / (self.depth_max - self.depth_min)
            if self.depth_invert:
                depth = 1.0 - depth
        depth = np.clip(depth, 0.0, 1.0)[..., None]

        depth_chw = np.transpose(depth, (2, 0, 1)).astype(np.float32)
        depth_chw = self._resize_image_chw(
            depth_chw,
            self.depth_target_shape,
            interpolation=cv2.INTER_NEAREST)
        return depth_chw

    def _extract_camera_output(self, isaac_obs: dict, env, key: str):
        camera_obs = isaac_obs.get(self.camera_obs_group)
        if isinstance(camera_obs, dict) and key in camera_obs:
            return camera_obs[key]

        camera_sensor = env.unwrapped.scene.sensors.get(self.camera_sensor_name)
        if camera_sensor is None:
            return None
        return camera_sensor.data.output.get(key)

    def _extract_obs(self, isaac_obs: dict, env) -> dict:
        from isaaclab.utils.math import subtract_frame_transforms

        unwrapped_env = env.unwrapped
        result = {}

        if "ee_pose" in self.lowdim_keys or "ee_pos" in self.lowdim_keys:
            ee_frame = unwrapped_env.scene["ee_frame"]
            robot = unwrapped_env.scene["robot"]
            ee_pos, _ = subtract_frame_transforms(
                robot.data.root_pos_w,
                robot.data.root_quat_w,
                ee_frame.data.target_pos_w[:, 0, :],
            )
            ee_pos_np = ee_pos.detach().cpu().numpy().astype(np.float32)
            if "ee_pose" in self.lowdim_keys:
                result["ee_pose"] = ee_pos_np
            if "ee_pos" in self.lowdim_keys:
                result["ee_pos"] = ee_pos_np

        if "ee_quat" in self.lowdim_keys:
            ee_frame = unwrapped_env.scene["ee_frame"]
            ee_quat = ee_frame.data.target_quat_w[:, 0, :]
            result["ee_quat"] = ee_quat.detach().cpu().numpy().astype(np.float32)

        if "contact_force_z" in self.lowdim_keys:
            force_z_list = []
            for sensor_name in ["contact_sensor_link1", "contact_sensor_link2", "contact_sensor_link3"]:
                sensor = unwrapped_env.scene.sensors.get(sensor_name)
                if sensor is not None:
                    force_z = sensor.data.net_forces_w[:, :, 2].sum(dim=-1)
                    force_z_list.append(force_z)
                else:
                    force_z_list.append(
                        torch.zeros(unwrapped_env.num_envs, device=unwrapped_env.device, dtype=torch.float32)
                    )
            contact_force = torch.stack(force_z_list, dim=-1).to(dtype=torch.float32)
            result["contact_force_z"] = contact_force.detach().cpu().numpy()

        if "camera_rgb" in self.rgb_keys:
            rgb_raw = self._extract_camera_output(isaac_obs, env, "rgb")
            if rgb_raw is None:
                raise RuntimeError("Failed to extract RGB camera observations from IsaacLab.")
            rgb = self._to_numpy(rgb_raw)
            rgb = self._ensure_batched_hwc(rgb)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            rgb = rgb.astype(np.float32)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            rgb = np.transpose(rgb, (0, 3, 1, 2))
            rgb_target = self.rgb_target_shape
            rgb_out = np.stack(
                [self._resize_image_chw(rgb[idx], rgb_target) for idx in range(rgb.shape[0])],
                axis=0,
            ).astype(np.float32)
            result["camera_rgb"] = rgb_out

        if "camera_depth" in self.depth_keys:
            depth_raw = self._extract_camera_output(isaac_obs, env, self.depth_data_type)
            if depth_raw is None:
                raise RuntimeError("Failed to extract depth camera observations from IsaacLab.")
            depth_np = self._ensure_batched_depth_hwc(self._to_numpy(depth_raw))
            depth_out = np.stack(
                [self._prepare_depth_chw(depth_np[idx]) for idx in range(depth_np.shape[0])],
                axis=0,
            ).astype(np.float32)
            result["camera_depth"] = depth_out

        missing_keys = [key for key in self.shape_meta["obs"].keys() if key not in result]
        if missing_keys:
            raise RuntimeError(f"Failed to extract required observations: {missing_keys}")
        return result

    def _obs_to_stacked(self, obs_history: list[dict], n_steps: int) -> dict:
        result = {}
        keys = obs_history[-1].keys()
        for key in keys:
            all_obs = [obs[key] for obs in obs_history]
            n_available = len(all_obs)
            latest_shape = all_obs[-1].shape
            batch_size = latest_shape[0]
            obs_shape = latest_shape[1:]

            stacked = np.zeros((batch_size, n_steps) + obs_shape, dtype=all_obs[-1].dtype)
            start_idx = max(0, n_steps - n_available)
            src_start = max(0, n_available - n_steps)
            for t_idx, src_idx in enumerate(range(src_start, n_available)):
                stacked[:, start_idx + t_idx] = all_obs[src_idx]
            if start_idx > 0:
                for t_idx in range(start_idx):
                    stacked[:, t_idx] = stacked[:, start_idx]
            result[key] = stacked
        return result

    def _freeze_image_obs(self, obs: dict, frozen_image_obs: Optional[dict]) -> dict:
        if not self.use_initial_image_obs_only or frozen_image_obs is None:
            return obs
        result = dict(obs)
        for key, value in frozen_image_obs.items():
            result[key] = value.copy()
        return result

    def _apply_binary_action_quantization(self, action: np.ndarray) -> np.ndarray:
        if len(self.binary_action_indices) == 0:
            return action
        action = action.copy()
        selected = action[..., self.binary_action_indices]
        action[..., self.binary_action_indices] = np.where(
            selected >= self.binary_action_threshold,
            self.binary_action_high,
            self.binary_action_low).astype(action.dtype)
        return action

    def _normalize_quaternion_action(self, action: np.ndarray) -> np.ndarray:
        if len(self.quaternion_action_indices) == 0:
            return action
        action = action.copy()
        quat = action[..., self.quaternion_action_indices]
        quat_norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        quat_norm = np.maximum(quat_norm, 1e-8)
        action[..., self.quaternion_action_indices] = quat / quat_norm
        return action

    @staticmethod
    def _frame_from_obs(obs: dict, render_obs_key: str) -> Optional[np.ndarray]:
        if render_obs_key not in obs:
            return None
        frame = obs[render_obs_key][0]
        if frame.ndim != 3:
            return None
        frame = np.transpose(frame, (1, 2, 0))
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        frame = np.clip(frame, 0.0, 1.0)
        return np.round(frame * 255.0).astype(np.uint8)

    def _save_video(self, frames: list[np.ndarray], file_path: str):
        if len(frames) == 0:
            return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(file_path, fourcc, self.fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    def _build_episode_specs(self) -> list[dict]:
        specs = []
        for i in range(self.n_train):
            specs.append({
                "prefix": "train/",
                "seed": self.train_start_idx + i,
                "enable_render": i < self.n_train_vis,
            })
        for i in range(self.n_test):
            specs.append({
                "prefix": "test/",
                "seed": self.test_start_seed + i,
                "enable_render": i < self.n_test_vis,
            })
        return specs

    def run(self, policy: BaseImagePolicy) -> Dict:
        self._ensure_isaaclab_initialized()

        env = self._env
        device = policy.device
        dtype = policy.dtype

        episode_specs = self._build_episode_specs()
        if len(episode_specs) == 0:
            return {}

        max_env_steps = self.max_steps
        total_progress_steps = self.observe_before_act_steps + max_env_steps
        log_data = {}
        grouped_returns = collections.defaultdict(list)
        grouped_max_rewards = collections.defaultdict(list)

        for ep_idx, spec in enumerate(episode_specs):
            seed = spec["seed"]
            prefix = spec["prefix"]
            enable_render = spec["enable_render"]

            torch.manual_seed(seed)
            np.random.seed(seed)

            reset_result = env.reset(seed=seed)
            if isinstance(reset_result, tuple):
                isaac_obs, _info = reset_result
            else:
                isaac_obs = reset_result

            policy.reset()
            obs = self._extract_obs(isaac_obs, env)
            frozen_image_obs = None
            if self.use_initial_image_obs_only:
                frozen_image_obs = {
                    key: obs[key].copy()
                    for key in (self.rgb_keys + self.depth_keys)
                    if key in obs
                }
                obs = self._freeze_image_obs(obs, frozen_image_obs)
            obs_history = [obs]

            video_frames = []
            first_frame = self._frame_from_obs(obs, self.render_obs_key)
            if enable_render and first_frame is not None:
                video_frames.append(first_frame)

            episode_rewards = []
            done = False
            step_count = 0
            action_chunk = None
            action_chunk_step = 0
            execution_step_count = 0

            pbar = tqdm.tqdm(
                total=total_progress_steps,
                desc=f"Eval {prefix.rstrip('/')} {ep_idx + 1}/{len(episode_specs)}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            for _ in range(self.observe_before_act_steps):
                if done:
                    break
                stacked_obs = self._obs_to_stacked(obs_history, self.n_obs_steps)
                obs_dict = {
                    key: torch.from_numpy(val).to(device=device, dtype=dtype)
                    for key, val in stacked_obs.items()
                }
                with torch.no_grad():
                    policy.observe_only(obs_dict)
                obs_history.append(obs)
                if len(obs_history) > self.n_obs_steps + 1:
                    obs_history.pop(0)
                if enable_render:
                    frame = self._frame_from_obs(obs, self.render_obs_key)
                    if frame is not None:
                        video_frames.append(frame)
                step_count += 1
                pbar.update(1)

            while not done and execution_step_count < max_env_steps:
                if action_chunk is None or action_chunk_step >= action_chunk.shape[1]:
                    stacked_obs = self._obs_to_stacked(obs_history, self.n_obs_steps)
                    obs_dict = {
                        key: torch.from_numpy(val).to(device=device, dtype=dtype)
                        for key, val in stacked_obs.items()
                    }
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)
                    action_chunk = action_dict["action"].detach().cpu().numpy()
                    action_chunk_step = 0

                current_action = action_chunk[:, action_chunk_step, :]
                current_action = self._normalize_quaternion_action(current_action)
                current_action = self._apply_binary_action_quantization(current_action)
                action_tensor = torch.from_numpy(current_action).to(
                    device=env.unwrapped.device,
                    dtype=torch.float32,
                )

                step_result = env.step(action_tensor)
                if len(step_result) == 5:
                    isaac_obs, reward, terminated, truncated, info = step_result
                else:
                    raise RuntimeError(
                        f"Unexpected IsaacLab step return length: {len(step_result)}"
                    )

                obs = self._extract_obs(isaac_obs, env)
                obs = self._freeze_image_obs(obs, frozen_image_obs)
                obs_history.append(obs)
                if len(obs_history) > self.n_obs_steps + 1:
                    obs_history.pop(0)

                reward_np = self._to_numpy(reward).astype(np.float32)
                episode_rewards.append(float(np.mean(reward_np)))

                if enable_render:
                    frame = self._frame_from_obs(obs, self.render_obs_key)
                    if frame is not None:
                        video_frames.append(frame)

                done = self._as_bool(terminated) or self._as_bool(truncated)
                action_chunk_step += 1
                step_count += 1
                execution_step_count += 1
                pbar.update(1)

            pbar.close()

            if len(episode_rewards) > 0:
                episode_return = float(np.sum(episode_rewards))
                max_reward = float(np.max(episode_rewards))
            else:
                episode_return = 0.0
                max_reward = 0.0

            grouped_returns[prefix].append(episode_return)
            grouped_max_rewards[prefix].append(max_reward)
            log_data[f"{prefix}episode_return_{seed}"] = episode_return
            log_data[f"{prefix}max_reward_{seed}"] = max_reward

            if enable_render and len(video_frames) > 0 and wandb is not None:
                video_dir = pathlib.Path(self.output_dir) / "media"
                video_dir.mkdir(parents=True, exist_ok=True)
                video_path = str(video_dir / f"{wv.util.generate_id()}.mp4")
                self._save_video(video_frames, video_path)
                log_data[f"{prefix}video_{seed}"] = wandb.Video(video_path)

        for prefix, values in grouped_returns.items():
            log_data[prefix + "mean_episode_return"] = float(np.mean(values))
        for prefix, values in grouped_max_rewards.items():
            mean_value = float(np.mean(values))
            log_data[prefix + "mean_max_reward"] = mean_value
            log_data[prefix + "mean_score"] = mean_value

        return log_data

    def __del__(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        if self._simulation_app is not None:
            try:
                self._simulation_app.close()
            except Exception:
                pass
