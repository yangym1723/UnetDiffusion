"""
Usage:
python eval.py --checkpoint data/outputs/2026.03.26/16.33.36_train_diffusion_unet_hybrid_ThreeFingers/checkpoints/latest.ckpt -o data/ThreeFingers_eval_output
python eval.py --checkpoint data/outputs/2026.03.26/16.33.36_train_diffusion_unet_hybrid_ThreeFingers/checkpoints/latest.ckpt -o data/ThreeFingers_eval_output --gui --n-test 1 --n-test-vis 1
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--headless/--gui', default=None, help='Override runner headless mode.')
@click.option('--n-test', type=int, default=None, help='Override number of test episodes.')
@click.option('--n-test-vis', type=int, default=None, help='Override number of rendered test episodes.')
@click.option('--test-start-seed', type=int, default=None, help='Override the first evaluation seed.')
@click.option('--render-obs-key', type=str, default=None, help='Override rendered observation key, e.g. camera_rgb or camera_depth.')
def main(checkpoint, output_dir, device, headless, n_test, n_test_vis, test_start_seed, render_obs_key):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    # eval.py loads runner config from the checkpoint payload, not the current yaml.
    # Allow a few common eval-time overrides without retraining.
    if headless is not None:
        cfg.task.env_runner.headless = headless
    if n_test is not None:
        cfg.task.env_runner.n_test = int(n_test)
    if n_test_vis is not None:
        cfg.task.env_runner.n_test_vis = int(n_test_vis)
    if test_start_seed is not None:
        cfg.task.env_runner.test_start_seed = int(test_start_seed)
    if render_obs_key is not None:
        cfg.task.env_runner.render_obs_key = str(render_obs_key)
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
