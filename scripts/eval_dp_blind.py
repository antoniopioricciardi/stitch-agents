"""Blind check for a trained ManiSkill Diffusion Policy agent (third_party/maniskill_diffusion_policy, unmodified).

Usage: uv run python scripts/eval_dp_blind.py <run_dir> <env_id> <demo_h5> [<out-tag>]
  run_dir: results/<run>/ (with runs/<name>/checkpoints/<final>.pt saved by run_dp.sh's --save_freq)

Loads the final EMA agent and evaluates it with the baseline's own evaluate() and env construction, three ways:
  full     = the agent as trained;
  zeroed   = the visual feature (PlainConv output, 256-d per frame) replaced by zeros;
  shuffled = the visual feature replaced, per frame and per call, by the feature of a random training frame
             (bank of 2048 frames from the first 100 demos, encoded by the same encoder).
The state part of the conditioning is untouched, so zeroed/shuffled say how much the agent solves without vision.
Run it from the repo root with PYTHONPATH=<repo>:<repo>/third_party/maniskill_diffusion_policy (import of train_rgbd).
"""
import glob
import json
import sys
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import torch

import train_rgbd  # the baseline script; its training code sits under `if __name__ == "__main__"`
from stitch.models import swap_dino
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

N_EPISODES = 250
N_BANK = 2048
SEED = 0


# the guard is needed: the eval envs are forkserver workers, which re-import this script
if __name__ == "__main__":
    RUN_DIR, ENV_ID, DEMOS = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    TAG = f"_{sys.argv[4]}" if len(sys.argv) > 4 else ""  # e.g. "in_look2" when an agent is run in another domain (Step 3)
    OUT = Path("results") / f"{date.today():%Y%m%d}_blind_{RUN_DIR.name.split('_', 1)[1]}{TAG}"
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(SEED), np.random.seed(SEED)

    ckpt = max(glob.glob(str(RUN_DIR / "runs/*/checkpoints/[0-9]*.pt")), key=lambda p: int(Path(p).stem))  # final weights
    args = train_rgbd.Args()  # defaults = the settings used by run_dp.sh (obs horizon 2, act 8, pred 16, U-Net dims)
    env_kwargs = dict(control_mode="pd_ee_delta_pos", reward_mode="sparse", obs_mode="rgb", render_mode="rgb_array",
                      human_render_camera_configs=dict(shader_pack="default"), max_episode_steps=100)  # as in train_rgbd.py
    envs = make_eval_envs(ENV_ID, args.num_eval_envs, "physx_cpu", env_kwargs, dict(obs_horizon=args.obs_horizon),
                          video_dir=None, wrappers=[FlattenRGBDObservationWrapper])
    sd = torch.load(ckpt)["ema_agent"]
    agent = swap_dino(train_rgbd.Agent(envs, args).to(dev), sd)  # Step 3 agents have a DINO encoder
    agent.load_state_dict(sd)

    # bank of visual features of random training frames: (N_BANK, 256)
    rng = np.random.default_rng(SEED)
    with h5py.File(DEMOS, "r") as f:
        frames = []
        for i in rng.integers(0, 100, N_BANK):
            rgb = f[f"traj_{i}"]["obs"]["sensor_data"]["base_camera"]["rgb"]
            frames.append(rgb[rng.integers(len(rgb))])
    frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().to(dev) / 255.0  # (N, 3, H, W), as encode_obs
    encoder_forward = agent.visual_encoder.forward
    with torch.no_grad():
        bank = torch.cat([encoder_forward(x) for x in frames.split(256)])

    modes = {
        "full": encoder_forward,
        "zeroed": lambda x: torch.zeros(x.shape[0], bank.shape[1], device=x.device),
        "shuffled": lambda x: bank[torch.randint(len(bank), (x.shape[0],), device=x.device)],
    }
    m = dict(checkpoint=ckpt)
    for name, fwd in modes.items():
        agent.visual_encoder.forward = fwd
        metrics = evaluate(N_EPISODES, agent, envs, dev, "physx_cpu", progress_bar=False)
        m[name] = dict(success_once=float(metrics["success_once"].mean()), success_at_end=float(metrics["success_at_end"].mean()),
                       episodes=int(len(metrics["success_once"])))
        print(name, m[name], flush=True)
    envs.close()
    json.dump(dict(run_dir=str(RUN_DIR), env_id=ENV_ID, demos=DEMOS, n_episodes=N_EPISODES, n_bank=N_BANK, seed=SEED),
              open(OUT / "config.json", "w"), indent=1)
    json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
