"""Step B: zero-shot Perception Stitching, no map. Encoder agent u (a PesEncoder with its own domain's anchor frames)
-> controller agent v, closed loop.

Usage: uv run python scripts/stepB_pes_stitch.py matrix <shift> <pair> <cam0_run_dir> <shift_run_dir> [--episodes=250] [--tag=x]
       uv run python scripts/stepB_pes_stitch.py goal <pair> <encoder_run_dir> <controller_run_dir> [--episodes=250] [--tag=x]
Run from the repo root with PYTHONPATH=<repo>:<repo>/third_party/maniskill_diffusion_policy.

matrix: the directions of scripts/step5b_motion_matrix.py: fwd = cam0 encoder -> shift controller, played in cam0;
bwd = shift encoder -> cam0 controller, played in the shift domain.
goal: look 2 (default task) encoder -> cam0 goal-variant controller, played on the goal variant in look 2
(as scripts/step5b_motion_stepR.py goal).
Stitched agent = the controller agent with its visual_encoder replaced by the encoder agent's (PeS's relative
representation is meant to be directly compatible: no map). Final checkpoints; the baseline's evaluate().
"""
import copy
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1b_stitch import ENV, ENV_KWARGS, load_agent
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

GOAL_ENV = "stitch.envs:StitchPickCubeLollipopNoGraspLook2Goal-v1"
N_EPISODES = 250
DEV = "cuda"

if __name__ == "__main__":
    MODE = sys.argv[1]
    opt = lambda k, d: next((a.split("=")[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    N_EPISODES = int(opt("episodes", N_EPISODES))
    if MODE == "matrix":
        SHIFT, PAIR, RUN0, RUN1 = sys.argv[2:6]
        agents = {"cam0": load_agent(RUN0), SHIFT: load_agent(RUN1)}
        stitches = [("fwd", "cam0", SHIFT, ENV["cam0"]), ("bwd", SHIFT, "cam0", ENV[SHIFT])]  # (name, enc, ctrl, env)
        OUT = Path("results") / f"{date.today():%Y%m%d}_stepB_pes_matrix_{SHIFT}_{PAIR}{opt('tag', '')}"
        runs = {"cam0": RUN0, SHIFT: RUN1}
    else:
        PAIR, RUN0, RUN1 = sys.argv[2:5]
        agents = {"look2": load_agent(RUN0), "goal_cam0": load_agent(RUN1)}
        stitches = [("goal", "look2", "goal_cam0", GOAL_ENV)]
        OUT = Path("results") / f"{date.today():%Y%m%d}_stepB_pes_goal_{PAIR}{opt('tag', '')}"
        runs = {"look2": RUN0, "goal_cam0": RUN1}
    OUT.mkdir(parents=True, exist_ok=True)
    m = dict(checkpoints={d: a[1] for d, a in agents.items()}, stitch={})
    for name, u, v, env_id in stitches:
        stitched = copy.deepcopy(agents[v][0])
        stitched.visual_encoder = copy.deepcopy(agents[u][0].visual_encoder)
        envs = make_eval_envs(env_id, 10, "physx_cpu", ENV_KWARGS, dict(obs_horizon=2), video_dir=None,
                              wrappers=[FlattenRGBDObservationWrapper])
        ev = evaluate(N_EPISODES, stitched, envs, DEV, "physx_cpu", progress_bar=False)
        envs.close()
        m["stitch"][name] = dict(encoder=u, controller=v, env=env_id, success_once=float(ev["success_once"].mean()),
                                 success_at_end=float(ev["success_at_end"].mean()), episodes=len(ev["success_once"]))
        print(name, m["stitch"][name], flush=True)
        json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
    json.dump(dict(mode=MODE, pair=PAIR, runs=runs, n_episodes=N_EPISODES, map="none (PeS relative representation)"),
              open(OUT / "config.json", "w"), indent=1)
