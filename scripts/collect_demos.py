"""Step 1: motion-planning demos for one (task, robot), converted to EE delta pose; pixels only for the Step 2 slice.

Usage: uv run python scripts/collect_demos.py <task> <robot> <N>

Per demo (seed = SEED0 + i):
  1. plan with ManiSkill's mplib solution (pd_joint_pos), recording the joint targets and the phase of each step;
  2. replay the joint targets and convert them to pd_ee_delta_pose (ManiSkill's replay conversion), recording
     the converted actions and the env state before every converted step;
  3. for (default, panda) only: restore each converted state in every visual variant in VISUALS and render it
     (paired frames by construction). Other combinations are rendered on demand later from the stored states.
Only demos where both planning and conversion succeed are saved.

Output, results/<date>_step1_demos_<task>_<robot>/:
  demos.npz: action (T,7) EE delta pose, proprio (T,P), task, episode, seed, phase, steps_to_grasp, t, state (T,S)
    — one row per converted step, aligned with action (state and proprio are taken before the action);
  <domain>.npz per rendered visual variant: the same keys plus obs (T,128,128,3) uint8 and domain;
    phase = planner segment (see PHASES); steps_to_grasp = steps until the gripper has closed (0 at the first
    carry step, negative afterwards);
  raw_joint.npz: the planner's joint targets (action_joint, episode, seed, phase, steps_to_grasp) for all saved demos;
  metrics.json: per-demo success / conversion success / times / lengths.
proprio: see stitch.envs.proprio.
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stitch.envs import make_env, proprio
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.xarm6.motionplanner import XArm6RobotiqMotionPlanningSolver
from mani_skill.trajectory.utils.actions.conversion import from_pd_joint_pos_to_ee

TASK, ROBOT, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
SEED0 = 0
# cameras 0, 1, 2 with default look and light, for Step 2 (default task, Panda) only
VISUALS = [(0, 0, 0), (1, 0, 0), (2, 0, 0)] if (TASK, ROBOT) == ("default", "panda") else []
# the planner's own segments: approach = move above the cube, descend = move down to the grasp pose,
# grasp = close the gripper, carry = lift + move to goal (a single straight screw motion, so no separate lift)
PHASES = ["approach", "descend", "grasp", "carry"]
OUT = Path("results") / f"{date.today():%Y%m%d}_step1_demos_{TASK}_{ROBOT}"
OUT.mkdir(parents=True, exist_ok=True)


class Recorder(gym.Wrapper):
    # records the full env state and proprio before every step, and the action taken
    def __init__(self, env):
        super().__init__(env)
        self.states, self.actions, self.proprio = [], [], []

    def step(self, action):
        self.states.append(self.env.unwrapped.get_state()[0].cpu().numpy())
        self.proprio.append(proprio(self.env))
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        return self.env.step(action)


def plan(env, seed):
    # ManiSkill's PickCube solutions (panda / xarm6), copied to mark where each phase starts.
    # Returns (success, phase start indices into the recorded joint actions).
    env.reset(seed=seed)
    u = env.unwrapped
    solver = PandaArmMotionPlanningSolver if ROBOT == "panda" else XArm6RobotiqMotionPlanningSolver
    planner = solver(env, debug=False, vis=False, base_pose=u.agent.robot.pose, visualize_target_grasp_pose=False, print_env_info=False)
    obb = get_actor_obb(u.cube)
    target_closing = u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(obb, approaching=np.array([0, 0, -1]), target_closing=target_closing, depth=0.025)
    grasp_pose = u.agent.build_grasp_pose(np.array([0, 0, -1]), grasp_info["closing"], u.cube.pose.sp.p)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    starts = [0]
    res = planner.move_to_pose_with_screw(reach_pose) if ROBOT == "panda" else planner.move_to_pose_with_RRTStar(reach_pose)
    starts.append(len(env.actions))
    if res != -1:
        res = planner.move_to_pose_with_screw(grasp_pose)
    starts.append(len(env.actions))
    if res != -1:
        planner.close_gripper()
        starts.append(len(env.actions))
        res = planner.move_to_pose_with_screw(sapien.Pose(u.goal_site.pose.sp.p, grasp_pose.q))
    planner.close()
    return res != -1 and bool(res[-1]["success"].item()), starts


def phase_of(t, starts):
    return int(np.searchsorted(starts, t, side="right") - 1)


env_jp = Recorder(make_env(task=TASK, robot=ROBOT, obs_mode="state", control_mode="pd_joint_pos"))
env_ee_raw = make_env(task=TASK, robot=ROBOT, obs_mode="state", control_mode="pd_ee_delta_pose")
render_envs = {v: make_env(v, TASK, ROBOT, obs_mode="rgb", control_mode="pd_ee_delta_pose") for v in VISUALS}
domain_name = {v: f"cam{v[0]}_look{v[1]}_light{v[2]}" for v in VISUALS}

demo_keys = ["action", "proprio", "episode", "seed", "phase", "steps_to_grasp", "t", "state"]
data = {k: [] for k in demo_keys}
frames = {v: [] for v in VISUALS}
raw = {k: [] for k in ["action_joint", "episode", "seed", "phase", "steps_to_grasp"]}
demos = []
for i in range(N):
    seed = SEED0 + i
    d = dict(seed=seed)

    # 1. plan (joint targets)
    env_jp.states, env_jp.actions = [], []
    t0 = time.time()
    d["plan_success"], starts = plan(env_jp, seed)
    d["plan_time"] = time.time() - t0
    joint_actions = np.array(env_jp.actions)
    d["len_joint"] = len(joint_actions)

    # 2. convert to EE delta pose: replay joint targets in env_jp, step env_ee to track them.
    # The conversion may take up to 4 EE steps per joint step when an EE action would be clipped.
    t0 = time.time()
    d["conv_success"] = False
    if d["plan_success"]:
        env_jp.reset(seed=seed)
        env_ee = Recorder(env_ee_raw)
        env_ee.reset(seed=seed)
        env_ee.unwrapped.set_state(env_jp.unwrapped.get_state())
        n_jp = []  # joint step index behind each EE step

        class Counter(gym.Wrapper):
            def step(self, action):
                n_jp.append(len(env_jp.actions))
                return self.env.step(action)

        env_jp.states, env_jp.actions = [], []
        info = from_pd_joint_pos_to_ee("pd_ee_delta_pose", joint_actions, env_jp, Counter(env_ee))
        d["conv_success"] = bool(info["success"].item())
    d["conv_time"] = time.time() - t0

    # 3. store the converted rollout; render its states in every visual variant in VISUALS
    t0 = time.time()
    if d["conv_success"]:
        states, actions = np.array(env_ee.states), np.array(env_ee.actions)
        phase = np.array([phase_of(j - 1, starts) for j in n_jp])  # n_jp counts joint steps already taken
        d["len_ee"] = len(actions)
        to_grasp = np.argmax(phase == 3) - np.arange(len(actions))  # first carry step = gripper closed
        data["action"].append(actions), data["state"].append(states), data["proprio"].append(np.array(env_ee.proprio))
        data["phase"].append(phase), data["steps_to_grasp"].append(to_grasp), data["t"].append(np.arange(len(actions)))
        data["episode"].append(np.full(len(actions), i)), data["seed"].append(np.full(len(actions), seed))
        for v, renv in render_envs.items():
            renv.reset(seed=seed)
            for s in states:
                renv.unwrapped.set_state(torch.from_numpy(s[None]))
                frames[v].append(renv.unwrapped.get_obs()["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy())
        raw["action_joint"].append(joint_actions), raw["episode"].append(np.full(len(joint_actions), i))
        raw["seed"].append(np.full(len(joint_actions), seed))
        raw["phase"].append(np.array([phase_of(t, starts) for t in range(len(joint_actions))]))
        raw["steps_to_grasp"].append(starts[3] - np.arange(len(joint_actions)))
    d["render_time"] = time.time() - t0
    demos.append(d)
    print(f"demo {i} seed {seed}: plan {d['plan_success']} conv {d['conv_success']} "
          f"len joint/ee {d['len_joint']}/{d.get('len_ee', '-')} time plan/conv/render "
          f"{d['plan_time']:.1f}/{d['conv_time']:.1f}/{d['render_time']:.1f} s", flush=True)

D = {k: np.concatenate(x) for k, x in data.items()}
D["task"] = np.full(len(D["action"]), TASK)
np.savez_compressed(OUT / "demos.npz", **D)
for v in VISUALS:
    obs = np.stack(frames.pop(v))  # (T, 128, 128, 3) uint8; pop to free the list before the next camera
    np.savez_compressed(OUT / f"{domain_name[v]}.npz", obs=obs, domain=np.full(len(obs), domain_name[v]), **D)
    del obs
np.savez_compressed(OUT / "raw_joint.npz", **{k: np.concatenate(x) for k, x in raw.items()})

ok = [d for d in demos if d["conv_success"]]
summary = dict(
    n=N, plan_success=np.mean([d["plan_success"] for d in demos]),
    conv_success_given_plan=len(ok) / max(1, sum(d["plan_success"] for d in demos)),
    saved=len(ok), mean_len_ee=np.mean([d["len_ee"] for d in ok]), mean_len_joint=np.mean([d["len_joint"] for d in ok]),
    mean_time_per_demo=np.mean([d["plan_time"] + d["conv_time"] + d["render_time"] for d in demos]),
)
print(json.dumps(summary, indent=1))
json.dump(dict(task=TASK, robot=ROBOT, n=N, seed0=SEED0, visuals=[domain_name[v] for v in VISUALS], phases=PHASES,
               control_mode="pd_ee_delta_pose", proprio="qpos, qvel, tcp_pose"), open(OUT / "config.json", "w"), indent=1)
json.dump(dict(summary=summary, demos=demos), open(OUT / "metrics.json", "w"), indent=1)
