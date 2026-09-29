"""Step 2 sanity: BC from the true state (no pixels) with the same Controller, to separate recipe from pipeline.

Usage: uv run python scripts/train_state_bc.py <pose|pos> <single|chunk>
  pose = pd_ee_delta_pose demos (7-d), pos = pd_ee_delta_pos demos (4-d)
  single = one action per step; chunk = 2-step history, 16-step action chunk, 8 executed (ManiSkill's DP setting)
Both standardise proprio and actions per dimension. z = [cube pose (7), goal position (3)].
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stitch.data import chunk_targets, prev_index
from stitch.envs import make_env
from stitch.evaluate import evaluate
from stitch.models import Controller

CONTROL, RECIPE = sys.argv[1], sys.argv[2]
CONTROL_MODE = {"pose": "pd_ee_delta_pose", "pos": "pd_ee_delta_pos"}[CONTROL]
DEMOS = {"pose": "results/20260929_step1_demos_default_panda", "pos": "results/20260930_step1_demos_default_panda_pos"}[CONTROL]
HIST, CHUNK, EXEC = (2, 16, 8) if RECIPE == "chunk" else (1, 1, 1)
SEED = 0
STEPS = 20_000
BATCH = 256
LR = 3e-4
N_EVAL = 100
OUT = Path("results") / f"{date.today():%Y%m%d}_step2_state_bc_{CONTROL}_{RECIPE}"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"


def privileged(env, obs=None):
    # z = [cube pose (7), goal position (3)] (1, 10)
    u = env.unwrapped
    return torch.cat([u.cube.pose.raw_pose, u.goal_site.pose.p], 1).float().to(dev)


D = np.load(Path(DEMOS) / "demos.npz")
senv = make_env(obs_mode="state", control_mode=CONTROL_MODE)
senv.reset(seed=0)
Z = []
for s in D["state"]:
    senv.unwrapped.set_state(torch.from_numpy(s[None]))
    Z.append(privileged(senv))
Z = torch.cat(Z)                                                     # (T, 10)
P = torch.from_numpy(D["proprio"]).float().to(dev)                   # (T, 25)
A = torch.from_numpy(D["action"]).float().to(dev)                    # (T, a)
Y = torch.from_numpy(chunk_targets(D["action"], D["episode"], CHUNK)).to(dev)  # (T, CHUNK, a)
prev = torch.from_numpy(prev_index(D["episode"])).to(dev)


def inputs(idx):
    # history: [x_{t-1}, x_t] for z and proprio (HIST = 2), or x_t alone
    if HIST == 1:
        return Z[idx], P[idx]
    return torch.cat([Z[prev[idx]], Z[idx]], 1), torch.cat([P[prev[idx]], P[idx]], 1)


torch.manual_seed(SEED), np.random.seed(SEED)
ctrl = Controller(Z.shape[1] * HIST, P.shape[1] * HIST, A.shape[1], chunk=CHUNK).to(dev)
ctrl.p_mean.copy_(P.mean(0).repeat(HIST)), ctrl.p_std.copy_((P.std(0) + 1e-6).repeat(HIST))
ctrl.a_mean.copy_(A.mean(0)), ctrl.a_std.copy_(A.std(0) + 1e-6)
opt = torch.optim.Adam(ctrl.parameters(), lr=LR)
t0 = time.time()
for step in range(STEPS):
    idx = torch.randint(len(A), (BATCH,), device=dev)
    loss = F.mse_loss((ctrl(*inputs(idx)) - Y[idx]) / ctrl.a_std, torch.zeros_like(Y[idx]))  # standardised MSE
    opt.zero_grad(), loss.backward(), opt.step()
    if step % 5000 == 0 or step == STEPS - 1:
        print(f"step {step} loss {loss.item():.4f}", flush=True)
m = dict(train_time_s=time.time() - t0, final_loss=loss.item())
ctrl.eval()
for name, seed0, n in [("eval_seeds", 10_000, N_EVAL), ("train_seeds_0_9", 0, 10)]:
    once, at_end, first = evaluate(privileged, ctrl, (0, 0, 0), n, seed0=seed0, control_mode=CONTROL_MODE,
                                   hist=HIST, n_exec=EXEC, obs_mode="state")
    m[name] = dict(success_once=float(once.mean()), success_at_end=float(at_end.mean()),
                   mean_first_success=float(first[once].mean()) if once.any() else None)
    print(f"{CONTROL} {RECIPE} {name}: success any step {once.mean():.2f}, at end {at_end.mean():.2f}", flush=True)
torch.save(ctrl.state_dict(), OUT / f"seed{SEED}.pt")
json.dump(dict(control_mode=CONTROL_MODE, recipe=RECIPE, hist=HIST, chunk=CHUNK, n_exec=EXEC, seed=SEED, steps=STEPS,
               batch=BATCH, lr=LR, demos=DEMOS, z="cube raw pose (7) + goal pos (3)", max_episode_steps=160),
          open(OUT / "config.json", "w"), indent=1)
json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
