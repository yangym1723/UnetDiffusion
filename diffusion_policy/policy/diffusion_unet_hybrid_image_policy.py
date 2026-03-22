from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.action_history_rnn import ActionHistoryRNN
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            action_history_rnn=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            this_type = attr.get('type', 'low_dim')
            if this_type == 'rgb':
                obs_config['rgb'].append(key)
            elif this_type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {this_type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)

        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features)
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc)
            )

        obs_feature_dim = obs_encoder.output_shape()[0]

        action_history_rnn = dict(action_history_rnn or {})
        self.action_history_enabled = bool(action_history_rnn.pop('enabled', False))
        self.action_history_encoder = None
        self.action_history_feature_dim = 0
        if self.action_history_enabled:
            self.action_history_encoder = ActionHistoryRNN(
                input_dim=action_dim,
                **action_history_rnn
            )
            self.action_history_feature_dim = self.action_history_encoder.output_dim

        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
        if self.action_history_enabled:
            if global_cond_dim is None:
                global_cond_dim = 0
            global_cond_dim += self.action_history_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        self._history_state = None
        self._history_feature = None
        self._history_cumsum = None

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        if self.action_history_enabled:
            print("History RNN params: %e" % sum(p.numel() for p in self.action_history_encoder.parameters()))

    def reset(self, mask: Optional[torch.Tensor] = None):
        if not self.action_history_enabled:
            return
        if mask is None or self._history_feature is None:
            self._history_state = None
            self._history_feature = None
            self._history_cumsum = None
            return

        mask = mask.to(device=self._history_feature.device, dtype=torch.bool).flatten()
        if mask.shape[0] != self._history_feature.shape[0]:
            raise ValueError("History reset mask has incompatible batch dimension.")

        self._history_feature = self._history_feature.clone()
        self._history_feature[mask] = 0
        self._history_cumsum = self._history_cumsum.clone()
        self._history_cumsum[mask] = 0

        if self.action_history_encoder.is_lstm:
            hidden, cell = self._history_state
            hidden = hidden.clone()
            cell = cell.clone()
            hidden[:, mask] = 0
            cell[:, mask] = 0
            self._history_state = (hidden, cell)
        else:
            history_state = self._history_state.clone()
            history_state[:, mask] = 0
            self._history_state = history_state

    def _merge_global_condition(self,
            global_cond: Optional[torch.Tensor],
            history_feature: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if history_feature is None:
            return global_cond
        if global_cond is None:
            return history_feature
        return torch.cat([global_cond, history_feature], dim=-1)

    def _ensure_history_buffers(self, batch_size: int, device, dtype) -> Optional[torch.Tensor]:
        if not self.action_history_enabled:
            return None
        needs_reset = (
            self._history_feature is None
            or self._history_feature.shape[0] != batch_size
            or self._history_feature.device != device
            or self._history_feature.dtype != dtype
        )
        if needs_reset:
            self._history_state = self.action_history_encoder.get_zero_state(
                batch_size=batch_size,
                device=device,
                dtype=dtype)
            self._history_feature = torch.zeros(
                (batch_size, self.action_history_feature_dim),
                device=device,
                dtype=dtype)
            self._history_cumsum = torch.zeros(
                (batch_size, self.action_dim),
                device=device,
                dtype=dtype)
        return self._history_feature

    def _encode_action_history_from_batch(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.action_history_enabled:
            return None
        required_keys = ['action_history', 'action_history_mask', 'action_history_length']
        for key in required_keys:
            if key not in batch:
                raise RuntimeError(
                    f"Missing `{key}` in batch. Enable task.dataset.return_action_history in the config.")

        action_history = self.normalizer['action'].normalize(batch['action_history'])
        action_history_mask = batch['action_history_mask'].to(
            device=action_history.device,
            dtype=action_history.dtype).unsqueeze(-1)
        action_history = action_history * action_history_mask
        action_history_length = batch['action_history_length'].to(
            device=action_history.device,
            dtype=torch.long)
        return self.action_history_encoder(action_history, action_history_length)

    def _update_history_from_action_chunk(self, naction: torch.Tensor):
        if not self.action_history_enabled:
            return
        naction = naction.detach()
        self._ensure_history_buffers(
            batch_size=naction.shape[0],
            device=naction.device,
            dtype=naction.dtype)
        for step_idx in range(naction.shape[1]):
            self._history_cumsum = self._history_cumsum + naction[:, step_idx]
            self._history_feature, self._history_state = self.action_history_encoder.step(
                self._history_cumsum,
                self._history_state)

    def conditional_sample(self,
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(
                trajectory,
                t,
                local_cond=local_cond,
                global_cond=global_cond)
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **kwargs).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _predict_action_internal(self,
            obs_dict: Dict[str, torch.Tensor],
            history_feature: Optional[torch.Tensor],
            update_history: bool) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size, _, = value.shape[:2]
        horizon = self.horizon
        action_dim = self.action_dim
        obs_feature_dim = self.obs_feature_dim
        n_obs_steps = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
            global_cond = self._merge_global_condition(global_cond, history_feature)
            cond_data = torch.zeros(size=(batch_size, horizon, action_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, n_obs_steps, -1)
            global_cond = self._merge_global_condition(None, history_feature)
            cond_data = torch.zeros(
                size=(batch_size, horizon, action_dim + obs_feature_dim),
                device=device,
                dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :n_obs_steps, action_dim:] = nobs_features
            cond_mask[:, :n_obs_steps, action_dim:] = True

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)

        naction_pred = nsample[..., :action_dim]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = n_obs_steps - 1
        end = start + self.n_action_steps
        naction = naction_pred[:, start:end]
        action = action_pred[:, start:end]

        if update_history and self.action_history_enabled:
            self._update_history_from_action_chunk(naction)

        return {
            'action': action,
            'action_pred': action_pred
        }

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        history_feature = None
        if self.action_history_enabled:
            value = next(iter(obs_dict.values()))
            history_feature = self._ensure_history_buffers(
                batch_size=value.shape[0],
                device=self.device,
                dtype=self.dtype)
        return self._predict_action_internal(
            obs_dict=obs_dict,
            history_feature=history_feature,
            update_history=True)

    def predict_action_from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        history_feature = self._encode_action_history_from_batch(batch)
        return self._predict_action_internal(
            obs_dict=batch['obs'],
            history_feature=history_feature,
            update_history=False)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        history_feature = self._encode_action_history_from_batch(batch)

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
            global_cond = self._merge_global_condition(global_cond, history_feature)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
            global_cond = self._merge_global_condition(None, history_feature)

        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory,
            noise,
            timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
