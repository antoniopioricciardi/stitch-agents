"""Oracle check (Step 1b): does the agent find the cube from the image, or follow goal_pos from the state?

Usage: uv run python scripts/fake_goal_check.py <run_dir> <domain>   (domain: see ENV;
       PYTHONPATH=<repo>:<repo>/third_party/maniskill_diffusion_policy)
Final EMA agent, one CPU env (the domain's core setup, 100 steps, as the baseline's eval), the agent's own action loop
(obs horizon 2, 8 of 16 predicted actions executed, as diffusion_policy/evaluate.py). Two conditions, N episodes each,
eval seeds 10000+i (disjoint from the demos):
  true: unchanged;
  fake: goal_pos in the state (state[25:28]) replaced by the goal of reset(seed=20000+i), i.e. a goal from the env's
        own distribution; the real marker stays where it is in the image.
Per episode: success once (w.r.t. the real goal), grasped once, closest approach of the TCP to the cube, the real goal
and the fake goal, and placed_fake = the cube came within the goal threshold (0.025 m) of the fake goal. A working
agent also ends near the fake goal (carrying the cube there is the task); the shortcut shows as a lost grasp rate.
Also saves the 128x128 policy frames (every 10th step) of the first 3 failed true-goal episodes as one grid.
"""
import glob
import json
import sys
from datetime import date
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from PIL import Image

import train_rgbd
import stitch.envs  # noqa: F401 (registers the env ids)
from stitch.models import swap_dino

ENV = {"cam0": "StitchPickCubeLollipopNoGrasp-v1", "cam1": "StitchPickCubeLollipopNoGraspCam1-v1",
       "cam2": "StitchPickCubeLollipopNoGraspCam2-v1", "look1": "StitchPickCubeLollipopNoGraspLook1-v1",
       "cam2goal": "StitchPickCubeLollipopNoGraspCam2Goal-v1",
       "look2": "StitchPickCubeLollipopNoGraspLook2-v1",
       "light1": "StitchPickCubeLollipopNoGraspLight1-v1",
       "look2light1": "StitchPickCubeLollipopNoGraspLook2Light1-v1",
       "cam3": "StitchPickCubeLollipopNoGraspCam3-v1", "xarm_cam0": "StitchPickCubeLollipopNoGraspXarm-v1",
       "xarm_look2": "StitchPickCubeLollipopNoGraspLook2Xarm-v1", "goal_cam0": "StitchPickCubeLollipopNoGraspGoal-v1",
       "goal_look2": "StitchPickCubeLollipopNoGraspLook2Goal-v1"}
N = 50
GOAL = slice(-3, None)  # state = qpos, qvel, tcp_pose 7, goal_pos 3 (Panda 28-d, xArm6 + Robotiq 34-d)
DEV = "cuda"


def run(agent, env, goal_env, seed, fake, frames=None):
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    assert torch.allclose(obs["state"][0, GOAL], u.goal_site.pose.p[0], atol=1e-5)
    true_goal = u.goal_site.pose.p[0].clone()
    if fake:
        goal_env.reset(seed=seed + 10000)
        fake_goal = goal_env.unwrapped.goal_site.pose.p[0].clone()
    else:
        fake_goal = true_goal
    hist = [obs, obs]  # the baseline's FrameStack repeats the first observation
    d = dict(cube=np.inf, goal=np.inf, fake_goal=np.inf, cube_to_fake_goal=np.inf, success=False, grasped=False)
    step = 0
    while step < 100:
        state = torch.stack([h["state"][0] for h in hist])[None].clone().float()  # (1, 2, 28)
        state[..., GOAL] = fake_goal
        rgb = torch.stack([h["rgb"][0] for h in hist])[None]  # (1, 2, H, W, 3)
        acts = agent.get_action(dict(state=state.to(DEV), rgb=rgb.to(DEV)))[0].cpu().numpy()
        for a in acts:
            obs, _, _, _, info = env.step(a)
            step += 1
            hist = [hist[1], obs]
            tcp = u.agent.tcp.pose.p[0]
            d["cube"] = min(d["cube"], (tcp - u.cube.pose.p[0]).norm().item())
            d["goal"] = min(d["goal"], (tcp - true_goal).norm().item())
            d["fake_goal"] = min(d["fake_goal"], (tcp - fake_goal).norm().item())
            d["cube_to_fake_goal"] = min(d["cube_to_fake_goal"], (u.cube.pose.p[0] - fake_goal).norm().item())
            d["success"] |= bool(info["success"][0])
            d["grasped"] |= bool(u.agent.is_grasping(u.cube)[0])
            if frames is not None and step % 10 == 0:
                frames.append(obs["rgb"][0].cpu().numpy())
            if step >= 100:
                break
    d["fake_goal_to_goal"] = (fake_goal - true_goal).norm().item()
    d["placed_fake"] = d["cube_to_fake_goal"] < 0.025
    return d


if __name__ == "__main__":
    RUN_DIR, DOMAIN = sys.argv[1], sys.argv[2]
    ENV_ID = ENV[DOMAIN]
    OUT = Path("results") / f"{date.today():%Y%m%d}_step1b_fake_goal_{DOMAIN}_{Path(RUN_DIR).name.split('_', 1)[1]}"
    OUT.mkdir(parents=True, exist_ok=True)
    kw = dict(control_mode="pd_ee_delta_pos", reward_mode="sparse", obs_mode="rgb", sim_backend="cpu", max_episode_steps=100)
    env = FlattenRGBDObservationWrapper(gym.make(ENV_ID, **kw))
    goal_env = gym.make(ENV_ID, **{**kw, "obs_mode": "state"})

    # Agent only needs the spaces: state (2, S) with S from the env, rgb (2, 128, 128, 3), action in [-1, 1]^4
    S = env.observation_space["state"].shape[-1]
    fake_vec = type("E", (), dict(
        single_observation_space=spaces.Dict(state=spaces.Box(-np.inf, np.inf, (2, S)), rgb=spaces.Box(0, 255, (2, 128, 128, 3), np.uint8)),
        single_action_space=spaces.Box(-1, 1, (4,))))()
    ckpt = max(glob.glob(f"{RUN_DIR}/runs/*/checkpoints/[0-9]*.pt"), key=lambda p: int(Path(p).stem))
    sd = torch.load(ckpt)["ema_agent"]
    agent = swap_dino(train_rgbd.Agent(fake_vec, train_rgbd.Args()).to(DEV), sd)  # Step 3 agents have a DINO encoder
    agent.load_state_dict(sd)
    agent.eval()

    m, grid = dict(checkpoint=ckpt), []
    for cond in ("true", "fake"):
        eps = []
        for i in range(N):
            frames = [] if cond == "true" and len(grid) < 3 else None
            e = run(agent, env, goal_env, 10000 + i, cond == "fake", frames)
            eps.append(e)
            if frames is not None and not e["success"]:
                grid.append(np.concatenate(frames, 1))
        summ = {k: float(np.mean([e[k] for e in eps])) for k in ("success", "grasped", "placed_fake", "cube", "goal", "fake_goal", "cube_to_fake_goal", "fake_goal_to_goal")}
        summ["closer_to_fake_goal_than_cube"] = float(np.mean([e["fake_goal"] < e["cube"] for e in eps]))
        summ["closer_to_fake_goal_than_goal"] = float(np.mean([e["fake_goal"] < e["goal"] for e in eps]))
        print(cond, json.dumps(summ), flush=True)
        m[cond] = dict(summary=summ, episodes=eps)
    json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
    if grid:  # a good oracle may have no failed episode among the first ones
        Image.fromarray(np.concatenate(grid, 0)).save(OUT / "failed_true_goal_frames.png")
    json.dump(dict(run_dir=RUN_DIR, domain=DOMAIN, env_id=ENV_ID, n=N, eval_seeds="10000+i", fake_goal_seeds="20000+i",
                   grid="first 3 failed true-goal episodes, every 10th step"), open(OUT / "config.json", "w"), indent=1)
