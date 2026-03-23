from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
import cv2
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_mixed_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()


def _build_rotation_transformer_if_needed(abs_action, rotation_rep, action_preprocess):
    if action_preprocess == 'pos_quat' or not abs_action:
        return None

    from diffusion_policy.model.common.rotation_transformer import RotationTransformer

    return RotationTransformer(
        from_rep='axis_angle', to_rep=rotation_rep)


def _to_builtin_config(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {k: _to_builtin_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin_config(v) for v in value]
    return value


def _resize_hwc_image(image: np.ndarray, target_shape, interpolation=None) -> np.ndarray:
    target_c, target_h, target_w = tuple(target_shape)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise RuntimeError(f'Expected image with shape (H, W, C), got {image.shape}')

    source_h, source_w, source_c = image.shape
    if source_c != target_c:
        if source_c == 1 and target_c > 1:
            image = np.repeat(image, target_c, axis=-1)
            source_c = target_c
        elif target_c == 1 and source_c > 1:
            image = image[..., :1]
            source_c = 1
        else:
            raise RuntimeError(
                f'Cannot resize image with {source_c} channels to target '
                f'channel count {target_c}.')

    if (source_h, source_w, source_c) == (target_h, target_w, target_c):
        return image

    if interpolation is None:
        interpolation = cv2.INTER_AREA
        if target_h > source_h or target_w > source_w:
            interpolation = cv2.INTER_LINEAR
    resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    if resized.ndim == 2:
        resized = resized[..., None]
    if resized.shape[-1] != target_c:
        if resized.shape[-1] == 1 and target_c > 1:
            resized = np.repeat(resized, target_c, axis=-1)
        else:
            raise RuntimeError(
                f'Unexpected channel count after resize: got {resized.shape[-1]}, '
                f'expected {target_c}.')
    return resized.astype(image.dtype, copy=False)


def _prepare_depth_frame(
        frame: np.ndarray,
        target_shape,
        depth_min=None,
        depth_max=None,
        depth_invert=False) -> np.ndarray:
    frame = _resize_hwc_image(
        np.asarray(frame),
        target_shape,
        interpolation=cv2.INTER_NEAREST)

    depth = frame.astype(np.float32, copy=False)
    if np.issubdtype(frame.dtype, np.integer):
        depth = depth / 255.0
    else:
        if depth_min is not None and depth_max is not None:
            if depth_max <= depth_min:
                raise RuntimeError(
                    f'depth_max ({depth_max}) must be greater than depth_min ({depth_min}).')
            depth = np.clip(depth, depth_min, depth_max)
            depth = (depth - depth_min) / (depth_max - depth_min)
            if depth_invert:
                depth = 1.0 - depth
        else:
            if np.min(depth) < 0.0 or np.max(depth) > 1.0:
                raise RuntimeError(
                    'Depth frames are not in [0,1]. Set dataset.depth_min and '
                    'dataset.depth_max so they can be normalized consistently.')

    depth = np.clip(depth, 0.0, 1.0).astype(np.float32, copy=False)
    return depth

class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            action_preprocess='auto',
            action_pos_indices=None,
            action_quat_indices=None,
            action_range_indices=None,
            normalize_action_quaternion=False,
            use_legacy_normalizer=False,
            return_action_history=False,
            return_persistent_action_value=False,
            persistent_action_indices=None,
            persistent_action_window=0,
            persistent_action_reduce='mean',
            binary_action_indices=None,
            binary_action_low=-1.0,
            binary_action_high=1.0,
            binary_action_threshold=0.0,
            depth_min=None,
            depth_max=None,
            depth_invert=False,
            use_initial_image_obs_only=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0
        ):
        action_preprocess = str(action_preprocess).lower()
        if action_preprocess not in {'auto', 'pos_quat'}:
            raise ValueError(f'Unsupported action_preprocess: {action_preprocess}')

        rotation_transformer = _build_rotation_transformer_if_needed(
            abs_action=abs_action,
            rotation_rep=rotation_rep,
            action_preprocess=action_preprocess)

        replay_buffer = None
        cache_config = _to_builtin_config({
            'shape_meta': OmegaConf.to_container(shape_meta, resolve=True)
                if OmegaConf.is_config(shape_meta) else copy.deepcopy(shape_meta),
            'cache_storage_version': 4,
            'abs_action': abs_action,
            'rotation_rep': rotation_rep,
            'action_preprocess': action_preprocess,
            'action_pos_indices': action_pos_indices,
            'action_quat_indices': action_quat_indices,
            'action_range_indices': action_range_indices,
            'normalize_action_quaternion': normalize_action_quaternion,
            'return_persistent_action_value': return_persistent_action_value,
            'persistent_action_indices': persistent_action_indices,
            'persistent_action_window': persistent_action_window,
            'persistent_action_reduce': persistent_action_reduce,
            'binary_action_indices': binary_action_indices,
            'binary_action_low': binary_action_low,
            'binary_action_high': binary_action_high,
            'binary_action_threshold': binary_action_threshold,
            'depth_min': depth_min,
            'depth_max': depth_max,
            'depth_invert': depth_invert,
            'use_initial_image_obs_only': use_initial_image_obs_only,
        })
        cache_hash = hashlib.sha1(
            json.dumps(cache_config, sort_keys=True).encode('utf-8')
        ).hexdigest()[:10]
        if use_cache:
            cache_zarr_path = dataset_path + f'.{cache_hash}.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer,
                            action_preprocess=action_preprocess,
                            action_pos_indices=action_pos_indices,
                            action_quat_indices=action_quat_indices,
                            action_range_indices=action_range_indices,
                            normalize_action_quaternion=normalize_action_quaternion,
                            binary_action_indices=binary_action_indices,
                            binary_action_low=binary_action_low,
                            binary_action_high=binary_action_high,
                            binary_action_threshold=binary_action_threshold,
                            depth_min=depth_min,
                            depth_max=depth_max,
                            depth_invert=depth_invert,
                            use_initial_image_obs_only=use_initial_image_obs_only)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer,
                action_preprocess=action_preprocess,
                action_pos_indices=action_pos_indices,
                action_quat_indices=action_quat_indices,
                action_range_indices=action_range_indices,
                normalize_action_quaternion=normalize_action_quaternion,
                binary_action_indices=binary_action_indices,
                binary_action_low=binary_action_low,
                binary_action_high=binary_action_high,
                binary_action_threshold=binary_action_threshold,
                depth_min=depth_min,
                depth_max=depth_max,
                depth_invert=depth_invert,
                use_initial_image_obs_only=use_initial_image_obs_only)

        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'depth':
                depth_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + depth_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.obs_shape_meta = obs_shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        self.return_action_history = return_action_history
        self.return_persistent_action_value = return_persistent_action_value
        self.use_initial_image_obs_only = use_initial_image_obs_only
        self.action_dim = shape_meta['action']['shape'][0]
        self.action_preprocess = action_preprocess
        self.action_pos_indices = _resolve_action_indices(action_pos_indices, self.action_dim)
        self.action_quat_indices = _resolve_action_indices(action_quat_indices, self.action_dim)
        self.action_range_indices = _resolve_action_indices(action_range_indices, self.action_dim)
        self.normalize_action_quaternion = bool(normalize_action_quaternion)
        self.persistent_action_indices = _resolve_action_indices(
            persistent_action_indices, self.action_dim)
        self.persistent_action_window = int(persistent_action_window)
        self.persistent_action_reduce = persistent_action_reduce
        self.binary_action_indices = _resolve_action_indices(
            binary_action_indices, self.action_dim)
        self.binary_action_low = float(binary_action_low)
        self.binary_action_high = float(binary_action_high)
        self.binary_action_threshold = float(binary_action_threshold)
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_invert = bool(depth_invert)
        self.episode_ends = replay_buffer.episode_ends[:].astype(np.int64)
        self.episode_starts = np.concatenate([
            np.zeros((1,), dtype=np.int64),
            self.episode_ends[:-1]
        ])
        self.max_episode_length = int(np.max(self.episode_ends - self.episode_starts))
        self.initial_image_arrays = {}
        if self.use_initial_image_obs_only:
            meta_group = self.replay_buffer.meta
            if 'initial_images' not in meta_group:
                raise RuntimeError(
                    "ReplayBuffer cache is missing `meta/initial_images`. "
                    "Delete the old cache and rebuild it with the current code.")
            initial_image_group = meta_group['initial_images']
            for key in self.rgb_keys + self.depth_keys:
                if key not in initial_image_group:
                    raise RuntimeError(
                        f"ReplayBuffer cache is missing initial image data for `{key}`.")
                self.initial_image_arrays[key] = initial_image_group[key]

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.action_preprocess == 'pos_quat':
            range_indices = list(self.action_pos_indices) + list(self.action_range_indices)
            this_normalizer = get_mixed_normalizer_from_stat(
                stat,
                range_indices=range_indices)
        elif self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            attr = self.obs_shape_meta[key]
            normalizer_type = attr.get('normalizer', None)

            if normalizer_type == 'identity':
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif normalizer_type == 'range':
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                # Custom IsaacLab keys (for example ee_pose/contact_force_z)
                # should remain usable without renaming them to robomimic
                # suffix conventions.
                this_normalizer = get_range_normalizer_from_stat(stat)
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        if self.use_initial_image_obs_only:
            buffer_start_idx, _, _, _ = self.sampler.indices[idx]
            episode_idx = np.searchsorted(self.episode_ends, buffer_start_idx, side='right')
            image_steps = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
            for key in self.rgb_keys:
                initial_frame = np.asarray(self.initial_image_arrays[key][episode_idx])
                tiled = np.repeat(initial_frame[None, ...], image_steps, axis=0)
                obs_dict[key] = np.moveaxis(tiled, -1, 1).astype(np.float32) / 255.
            for key in self.depth_keys:
                initial_frame = np.asarray(self.initial_image_arrays[key][episode_idx])
                tiled = np.repeat(initial_frame[None, ...], image_steps, axis=0)
                obs_dict[key] = np.moveaxis(tiled, -1, 1).astype(np.float32)
        else:
            for key in self.rgb_keys:
                # move channel last to channel first
                # T,H,W,C
                # convert uint8 image to float32
                obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                    ).astype(np.float32) / 255.
                # T,C,H,W
                del data[key]
            for key in self.depth_keys:
                obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32)
                del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        if self.return_action_history:
            action_history, action_history_mask, action_history_length = self._get_action_history(idx)
            torch_data['action_history'] = torch.from_numpy(action_history)
            torch_data['action_history_mask'] = torch.from_numpy(action_history_mask)
            torch_data['action_history_length'] = torch.tensor(action_history_length, dtype=torch.long)
        if self.return_persistent_action_value:
            persistent_action_value = self._get_persistent_action_value(idx)
            torch_data['persistent_action_value'] = torch.from_numpy(persistent_action_value)
        return torch_data

    def _get_action_history(self, idx: int):
        buffer_start_idx, _, sample_start_idx, _ = self.sampler.indices[idx]
        episode_idx = np.searchsorted(self.episode_ends, buffer_start_idx, side='right')
        episode_start = int(self.episode_starts[episode_idx])
        episode_end = int(self.episode_ends[episode_idx])

        current_obs_offset = 0
        if self.n_obs_steps is not None:
            current_obs_offset = self.n_obs_steps - 1 - sample_start_idx
        current_abs_idx = buffer_start_idx + current_obs_offset
        current_abs_idx = max(current_abs_idx, episode_start)
        current_abs_idx = min(current_abs_idx, episode_end)

        action_history = self.replay_buffer['action'][episode_start:current_abs_idx].astype(np.float32)
        action_history = _apply_binary_action_quantization(
            action_history,
            self.binary_action_indices,
            self.binary_action_low,
            self.binary_action_high,
            self.binary_action_threshold)
        action_history_length = action_history.shape[0]

        padded_history = np.zeros(
            (self.max_episode_length, self.action_dim),
            dtype=np.float32)
        history_mask = np.zeros((self.max_episode_length,), dtype=bool)
        if action_history_length > 0:
            padded_history[:action_history_length] = action_history
            history_mask[:action_history_length] = True
        return padded_history, history_mask, action_history_length

    def _get_persistent_action_value(self, idx: int):
        if len(self.persistent_action_indices) == 0:
            raise RuntimeError(
                'persistent_action_indices is empty while return_persistent_action_value=True.')

        buffer_start_idx, _, _, _ = self.sampler.indices[idx]
        episode_idx = np.searchsorted(self.episode_ends, buffer_start_idx, side='right')
        episode_start = int(self.episode_starts[episode_idx])
        episode_end = int(self.episode_ends[episode_idx])

        window_end = episode_end
        if self.persistent_action_window > 0:
            window_end = min(episode_start + self.persistent_action_window, episode_end)
        action_window = self.replay_buffer['action'][episode_start:window_end].astype(np.float32)
        if action_window.shape[0] == 0:
            raise RuntimeError('persistent action window is empty.')

        selected = action_window[:, self.persistent_action_indices]
        if self.persistent_action_reduce == 'mean':
            persistent_action_value = selected.mean(axis=0)
        elif self.persistent_action_reduce == 'last':
            persistent_action_value = selected[-1]
        else:
            raise ValueError(
                f'Unsupported persistent_action_reduce: {self.persistent_action_reduce}')
        return persistent_action_value.astype(np.float32)


def _normalize_quaternion_components(actions, quat_indices):
    if len(quat_indices) == 0:
        return actions
    actions = actions.copy()
    quat = actions[..., quat_indices]
    quat_norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat_norm = np.maximum(quat_norm, 1e-8)
    actions[..., quat_indices] = quat / quat_norm
    return actions


def _convert_actions(
        raw_actions,
        abs_action,
        rotation_transformer,
        action_preprocess='auto',
        action_quat_indices=None,
        normalize_action_quaternion=False):
    actions = raw_actions.astype(np.float32)
    if action_preprocess == 'pos_quat':
        if normalize_action_quaternion:
            actions = _normalize_quaternion_components(
                actions, action_quat_indices or [])
        return actions

    if abs_action:
        if rotation_transformer is None:
            raise RuntimeError(
                "rotation_transformer is required when abs_action=True and "
                "action_preprocess is not `pos_quat`.")
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions


def _apply_binary_action_quantization(
        actions,
        binary_action_indices,
        low_value,
        high_value,
        threshold):
    if len(binary_action_indices) == 0:
        return actions
    actions = actions.copy()
    selected = actions[..., binary_action_indices]
    quantized = np.where(selected >= threshold, high_value, low_value).astype(actions.dtype)
    actions[..., binary_action_indices] = quantized
    return actions


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


def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer,
        action_preprocess='auto',
        action_pos_indices=None,
        action_quat_indices=None,
        action_range_indices=None,
        normalize_action_quaternion=False,
        binary_action_indices=None,
        binary_action_low=-1.0,
        binary_action_high=1.0,
        binary_action_threshold=0.0,
        depth_min=None,
        depth_max=None,
        depth_invert=False,
        use_initial_image_obs_only=False, n_workers=None, max_inflight_tasks=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    action_dim = int(shape_meta['action']['shape'][0])
    action_pos_indices = _resolve_action_indices(action_pos_indices, action_dim)
    action_quat_indices = _resolve_action_indices(action_quat_indices, action_dim)
    action_range_indices = _resolve_action_indices(action_range_indices, action_dim)
    binary_action_indices = _resolve_action_indices(binary_action_indices, action_dim)
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'depth':
            depth_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                    action_preprocess=action_preprocess,
                    action_quat_indices=action_quat_indices,
                    normalize_action_quaternion=normalize_action_quaternion,
                )
                this_data = _apply_binary_action_quantization(
                    this_data,
                    binary_action_indices,
                    binary_action_low,
                    binary_action_high,
                    binary_action_threshold)
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        def img_copy(zarr_arr, zarr_idx, image):
            try:
                zarr_arr[zarr_idx] = image
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False

        image_progress_total = (
            len(demos) * (len(rgb_keys) + len(depth_keys)) if use_initial_image_obs_only
            else n_steps * (len(rgb_keys) + len(depth_keys))
        )
        with tqdm(total=image_progress_total, desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                if use_initial_image_obs_only:
                    initial_image_group = meta_group.require_group('initial_images', overwrite=True)
                    for key in rgb_keys:
                        shape = tuple(shape_meta['obs'][key]['shape'])
                        c, h, w = shape
                        this_compressor = Jpeg2k(level=50)
                        img_arr = initial_image_group.require_dataset(
                            name=key,
                            shape=(len(demos), h, w, c),
                            chunks=(1, h, w, c),
                            compressor=this_compressor,
                            dtype=np.uint8
                        )
                        for episode_idx in range(len(demos)):
                            demo = demos[f'demo_{episode_idx}']
                            hdf5_arr = demo['obs'][key]
                            if hdf5_arr.shape[0] == 0:
                                raise RuntimeError(
                                    f'Image key `{key}` in demo_{episode_idx} has zero frames.')
                            initial_image = _resize_hwc_image(hdf5_arr[0], shape)

                            if len(futures) >= max_inflight_tasks:
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            futures.add(executor.submit(
                                img_copy,
                                img_arr,
                                episode_idx,
                                initial_image))
                    for key in depth_keys:
                        shape = tuple(shape_meta['obs'][key]['shape'])
                        c, h, w = shape
                        depth_arr = initial_image_group.require_dataset(
                            name=key,
                            shape=(len(demos), h, w, c),
                            chunks=(1, h, w, c),
                            compressor=None,
                            dtype=np.float32
                        )
                        for episode_idx in range(len(demos)):
                            demo = demos[f'demo_{episode_idx}']
                            hdf5_arr = demo['obs'][key]
                            if hdf5_arr.shape[0] == 0:
                                raise RuntimeError(
                                    f'Image key `{key}` in demo_{episode_idx} has zero frames.')
                            initial_depth = _prepare_depth_frame(
                                hdf5_arr[0],
                                shape,
                                depth_min=depth_min,
                                depth_max=depth_max,
                                depth_invert=depth_invert)

                            if len(futures) >= max_inflight_tasks:
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            futures.add(executor.submit(
                                img_copy,
                                depth_arr,
                                episode_idx,
                                initial_depth))
                else:
                    for key in rgb_keys:
                        data_key = 'obs/' + key
                        shape = tuple(shape_meta['obs'][key]['shape'])
                        c,h,w = shape
                        this_compressor = Jpeg2k(level=50)
                        img_arr = data_group.require_dataset(
                            name=key,
                            shape=(n_steps,h,w,c),
                            chunks=(1,h,w,c),
                            compressor=this_compressor,
                            dtype=np.uint8
                        )
                        for episode_idx in range(len(demos)):
                            demo = demos[f'demo_{episode_idx}']
                            hdf5_arr = demo['obs'][key]
                            episode_length = int(demo['actions'].shape[0])
                            if hdf5_arr.shape[0] == 0:
                                raise RuntimeError(
                                    f'Image key `{key}` in demo_{episode_idx} has zero frames.')

                            if hdf5_arr.shape[0] != episode_length:
                                raise RuntimeError(
                                    f'Image key `{key}` in demo_{episode_idx} has '
                                    f'{hdf5_arr.shape[0]} frames, but actions has '
                                    f'{episode_length} steps. Set '
                                    f'`task.dataset.use_initial_image_obs_only=True` '
                                    f'to broadcast the first frame across the episode.')
                            image_sequence = [hdf5_arr[hdf5_idx] for hdf5_idx in range(hdf5_arr.shape[0])]

                            for hdf5_idx, image in enumerate(image_sequence):
                                image = _resize_hwc_image(image, shape)
                                if len(futures) >= max_inflight_tasks:
                                    # limit number of inflight tasks
                                    completed, futures = concurrent.futures.wait(futures, 
                                        return_when=concurrent.futures.FIRST_COMPLETED)
                                    for f in completed:
                                        if not f.result():
                                            raise RuntimeError('Failed to encode image!')
                                    pbar.update(len(completed))

                                zarr_idx = episode_starts[episode_idx] + hdf5_idx
                                futures.add(
                                    executor.submit(img_copy, 
                                        img_arr, zarr_idx, image))
                    for key in depth_keys:
                        shape = tuple(shape_meta['obs'][key]['shape'])
                        c, h, w = shape
                        depth_arr = data_group.require_dataset(
                            name=key,
                            shape=(n_steps, h, w, c),
                            chunks=(1, h, w, c),
                            compressor=None,
                            dtype=np.float32
                        )
                        for episode_idx in range(len(demos)):
                            demo = demos[f'demo_{episode_idx}']
                            hdf5_arr = demo['obs'][key]
                            episode_length = int(demo['actions'].shape[0])
                            if hdf5_arr.shape[0] == 0:
                                raise RuntimeError(
                                    f'Image key `{key}` in demo_{episode_idx} has zero frames.')

                            if hdf5_arr.shape[0] != episode_length:
                                raise RuntimeError(
                                    f'Image key `{key}` in demo_{episode_idx} has '
                                    f'{hdf5_arr.shape[0]} frames, but actions has '
                                    f'{episode_length} steps. Set '
                                    f'`task.dataset.use_initial_image_obs_only=True` '
                                    f'to broadcast the first frame across the episode.')
                            image_sequence = [
                                _prepare_depth_frame(
                                    hdf5_arr[hdf5_idx],
                                    shape,
                                    depth_min=depth_min,
                                    depth_max=depth_max,
                                    depth_invert=depth_invert)
                                for hdf5_idx in range(hdf5_arr.shape[0])
                            ]

                            for hdf5_idx, image in enumerate(image_sequence):
                                if len(futures) >= max_inflight_tasks:
                                    completed, futures = concurrent.futures.wait(
                                        futures,
                                        return_when=concurrent.futures.FIRST_COMPLETED)
                                    for f in completed:
                                        if not f.result():
                                            raise RuntimeError('Failed to encode image!')
                                    pbar.update(len(completed))

                                zarr_idx = episode_starts[episode_idx] + hdf5_idx
                                futures.add(
                                    executor.submit(img_copy,
                                        depth_arr, zarr_idx, image))

                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
