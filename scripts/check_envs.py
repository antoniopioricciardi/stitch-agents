"""Step 1: save one RGB frame per env variant and print the rendering FPS.

Frames: all 18 visual combos (Panda, default task), the 3 task variants and the xArm6 (default visual).
Also checks that a state saved in one visual variant re-renders identically (same poses) in another.
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stitch.envs import CAMERAS, GOAL_Y, LOOKS, N_LIGHTS, TASKS, make_env

SEED = 0
FPS_STEPS = 200
OUT = Path("results") / f"{date.today():%Y%m%d}_step1_check_envs"
OUT.mkdir(parents=True, exist_ok=True)


def frame(obs):
    return obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()  # (128, 128, 3) uint8


variants = [((c, l, li), "default", "panda") for c in range(len(CAMERAS)) for l in range(len(LOOKS)) for li in range(N_LIGHTS)]
variants += [((0, 0, 0), t, "panda") for t in TASKS if t != "default"]
variants += [((0, 0, 0), "default", "xarm6"), ((1, 1, 0), "default", "xarm6")]

frames, names, init_states = [], [], {}
for visual, task, robot in variants:
    env = make_env(visual, task, robot)
    obs, _ = env.reset(seed=SEED)
    name = f"cam{visual[0]}_look{visual[1]}_light{visual[2]}_{task}_{robot}"
    plt.imsave(OUT / f"{name}.png", frame(obs))
    frames.append(frame(obs)), names.append(name)
    init_states[name] = env.unwrapped.get_state()[0].cpu().numpy()
    env.close()

# same seed -> same initial state across visual variants (Panda, default task)
ref = init_states["cam0_look0_light0_default_panda"]
max_diff = max(np.abs(s - ref).max() for n, s in init_states.items() if n.endswith("default_panda"))
print(f"max |state - state_ref| over visual variants, seed {SEED}: {max_diff:.2e}")

# grid of all frames
cols = 6
rows = int(np.ceil(len(frames) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
for ax in axes.flat:
    ax.axis("off")
for ax, f, n in zip(axes.flat, frames, names):
    ax.imshow(f)
    ax.set_title(n.replace("_default", "").replace("_panda", ""), fontsize=6)
fig.tight_layout()
fig.savefig(OUT / "grid.png", dpi=150)

# rendering FPS: env.step (CPU sim) + 128x128 RGB sensor render, 1 env, random joint-position actions
fps = {}
for visual in [(0, 0, 0), (2, 2, 1)]:
    env = make_env(visual)
    env.reset(seed=SEED)
    t = time.time()
    for _ in range(FPS_STEPS):
        env.step(env.action_space.sample())
    fps[f"step+render {visual}"] = FPS_STEPS / (time.time() - t)
    t = time.time()
    for _ in range(FPS_STEPS):
        env.unwrapped.get_obs()
    fps[f"render only {visual}"] = FPS_STEPS / (time.time() - t)
    env.close()
for k, v in fps.items():
    print(f"FPS {k}: {v:.0f}")

# paired re-render: roll out 30 random steps in visual A, restore each state in visual B.
# The cube must end up at the same pose in B as in A.
env_a, env_b = make_env((0, 0, 0)), make_env((1, 2, 1))
env_a.reset(seed=SEED), env_b.reset(seed=SEED)
for _ in range(30):
    obs_a, *_ = env_a.step(env_a.action_space.sample())
env_b.unwrapped.set_state(env_a.unwrapped.get_state())
obs_b = env_b.unwrapped.get_obs()
pose_diff = (env_a.unwrapped.cube.pose.raw_pose - env_b.unwrapped.cube.pose.raw_pose).abs().max().item()
print(f"paired re-render: max cube pose diff A vs B = {pose_diff:.2e}")
fig, axes = plt.subplots(1, 2, figsize=(4, 2.2))
for ax, f, n in zip(axes, [frame(obs_a), frame(obs_b)], ["A cam0_look0_light0", "B cam1_look2_light1 (restored)"]):
    ax.imshow(f), ax.axis("off"), ax.set_title(n, fontsize=6)
fig.tight_layout()
fig.savefig(OUT / "paired_rerender.png", dpi=150)

# goal variant: is the goal sphere inside the frame for every camera over the whole goal region?
# Project the 8 corners of the region (x, y, z ranges of the goal variant) and require the sphere
# (centre +- its radius in pixels) to lie inside the 128x128 image. Occlusion by the robot is not checked.
goal_view = {}
for c in range(len(CAMERAS)):
    env = make_env((c, 0, 0), "goal")
    obs, _ = env.reset(seed=SEED)
    u = env.unwrapped
    K = obs["sensor_param"]["base_camera"]["intrinsic_cv"][0].cpu().numpy()  # (3, 3)
    E = obs["sensor_param"]["base_camera"]["extrinsic_cv"][0].cpu().numpy()  # (3, 4) world -> camera
    xs = u.cube_spawn_center[0] - u.cube_spawn_half_size, u.cube_spawn_center[0] + u.cube_spawn_half_size
    zs = u.cube_half_size, u.cube_half_size + u.max_goal_height
    corners = np.array([[x, y, z, 1.0] for x in xs for y in GOAL_Y for z in zs])  # (8, 4)
    pc = corners @ E.T                                    # (8, 3) camera frame, z = depth
    uv = pc[:, :2] / pc[:, 2:] * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
    r_px = u.goal_thresh / pc[:, 2] * K[0, 0]             # sphere radius in pixels
    inside = (uv - r_px[:, None] >= 0).all(1) & (uv + r_px[:, None] < 128).all(1)
    goal_view[f"cam{c}"] = dict(all_inside=bool(inside.all()), uv_min=uv.min(0).round(1).tolist(), uv_max=uv.max(0).round(1).tolist())
    print(f"goal region in cam{c}: all corners inside = {inside.all()}, u,v range {uv.min(0).round(1)} .. {uv.max(0).round(1)}")
    env.close()

json.dump(dict(seed=SEED, variants=names, fps_steps=FPS_STEPS), open(OUT / "config.json", "w"), indent=1)
json.dump(dict(fps=fps, init_state_max_diff=float(max_diff), paired_pose_diff=pose_diff, goal_view=goal_view), open(OUT / "metrics.json", "w"), indent=1)
print("frames in", OUT)
