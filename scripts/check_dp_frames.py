"""Step 2c check 1 (no training): do the DP baseline's eval frames and state match the training data, on both envs?

Envs are built as the baseline's make_eval_envs builds them (gym.make with reconfiguration_freq=1 and its env_kwargs,
FlattenRGBDObservationWrapper). For the first 5 seeds of each demo set: first eval frame vs first training frame of the
demo with that seed, on the matching env; and our env vs PickCube-v1 for the same seed. Also camera intrinsics /
extrinsics and the flattened state vector (fields, dims, values).
"""
import json
import sys
from datetime import date
from pathlib import Path

import gymnasium as gym
import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stitch.envs  # noqa: F401  registers StitchPickCube-v1
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

DEMOS = {
    "ours": Path("results/20260930_dp_ours_demos_default_panda_cam0/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"),
    "theirs": Path.home() / ".maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5",
}
ENVS = {"ours": "StitchPickCube-v1", "theirs": "PickCube-v1"}
ENV_KWARGS = dict(control_mode="pd_ee_delta_pos", reward_mode="sparse", obs_mode="rgb", render_mode="rgb_array",
                  human_render_camera_configs=dict(shader_pack="default"), max_episode_steps=100)
OUT = Path("results") / f"{date.today():%Y%m%d}_step2c_check_dp_frames"
OUT.mkdir(parents=True, exist_ok=True)


def make(env_name):
    return FlattenRGBDObservationWrapper(gym.make(ENVS[env_name], reconfiguration_freq=1, **ENV_KWARGS))


def first_obs(env, seed):
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    cam = u.get_sensor_params()["base_camera"]
    return obs["rgb"][0].cpu().numpy(), obs["state"][0].cpu().numpy(), cam["intrinsic_cv"][0].cpu().numpy(), cam["extrinsic_cv"][0].cpu().numpy()


envs = {k: make(k) for k in ENVS}
m, panels = {}, []
for demo_name, path in DEMOS.items():
    meta = json.load(open(str(path)[:-2] + "json"))
    with h5py.File(path, "r") as f:
        for ep in meta["episodes"][:5]:
            seed = ep["episode_seed"]
            train = f[f"traj_{ep['episode_id']}"]["obs"]["sensor_data"]["base_camera"]["rgb"][0]
            ev = {k: first_obs(e, seed) for k, e in envs.items()}
            d_train = np.abs(ev[demo_name][0].astype(int) - train.astype(int))
            d_envs = np.abs(ev["ours"][0].astype(int) - ev["theirs"][0].astype(int))
            row = dict(
                seed=seed,
                eval_vs_train_pixels_diff=int((d_train.max(-1) > 10).sum()), eval_vs_train_max=int(d_train.max()),
                ours_vs_theirs_pixels_diff=int((d_envs.max(-1) > 10).sum()),
                state_dims={k: len(v[1]) for k, v in ev.items()},
                state_max_abs_diff_ours_vs_theirs=float(np.abs(ev["ours"][1] - ev["theirs"][1]).max()),
                intrinsics_equal=bool(np.allclose(ev["ours"][2], ev["theirs"][2])),
                extrinsics_equal=bool(np.allclose(ev["ours"][3], ev["theirs"][3], atol=1e-5)),
            )
            m[f"{demo_name}_seed{seed}"] = row
            print(demo_name, row, flush=True)
            panels.append((f"{demo_name} demo seed {seed}: train", train))
            panels.append((f"eval ours", ev["ours"][0]))
            panels.append((f"eval theirs", ev["theirs"][0]))
            panels.append((f"|ours - theirs|", (d_envs.max(-1) > 10).astype(float)))

# state field layout: agent (qpos, qvel) + extra keys, in order
raw = envs["ours"].unwrapped.get_obs()
m["state_fields_ours"] = {k: list(v.shape) for k, v in list(raw["agent"].items()) + list(raw["extra"].items())}
raw = envs["theirs"].unwrapped.get_obs()
m["state_fields_theirs"] = {k: list(v.shape) for k, v in list(raw["agent"].items()) + list(raw["extra"].items())}
print("state fields ours", m["state_fields_ours"], "theirs", m["state_fields_theirs"])

fig, axes = plt.subplots(len(panels) // 4, 4, figsize=(8, 2.1 * len(panels) // 4))
for ax, (title, img) in zip(axes.flat, panels):
    ax.imshow(img, cmap="gray"), ax.set_title(title, fontsize=6), ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "frames.png", dpi=130)
json.dump(dict(env_kwargs=ENV_KWARGS, reconfiguration_freq=1, demos={k: str(v) for k, v in DEMOS.items()}), open(OUT / "config.json", "w"), indent=1)
json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
