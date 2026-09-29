"""Step 2 sanity check: is 0/100 pixel BC a pipeline bug or a weak recipe?

1. Pipeline equality: in the evaluation env, replay the stored actions of demos 0-4 and compare every frame and
   proprio vector with the training data (cam0). Must match, and the replay must succeed.
2. State BC: the same MLP Controller with z = true state [cube pose (7), goal position (3)] instead of E(pixels);
   same demos, loss, steps and evaluation (100 episodes, seeds 10000+), plus the first 10 training seeds.
3. Memorisation: the Step 2 pixel agents (seed 0, cam0 and cam1) on the initial states of training demos 0-9.
Ran on 2026-09-30 with a 120-step episode limit (now 160) and single-step, unstandardised controllers.
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
from stitch.envs import make_env, proprio
from stitch.evaluate import evaluate, pixel_encoder
from stitch.models import Controller, Encoder

SEED = 0
STEPS = 20_000
BATCH = 256
LR = 3e-4
DEMOS = Path("results/20260929_step1_demos_default_panda")
PIXEL_RUNS = {c: Path(f"results/20260929_step2_bc_cam{c}") for c in (0, 1)}
OUT = Path("results") / f"{date.today():%Y%m%d}_step2_sanity_state_bc"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
metrics = {}

# 1. pipeline equality
data = np.load(DEMOS / "cam0_look0_light0.npz")
frames, episodes = data["obs"], data["episode"]  # load once: npz arrays are decompressed on every access
env = make_env((0, 0, 0), obs_mode="rgb", control_mode="pd_ee_delta_pose")
eq = []
for ep in range(5):
    m = episodes == ep
    obs, _ = env.reset(seed=int(data["seed"][m][0]))
    px, pr = 0, 0
    for t, a in enumerate(data["action"][m]):
        px = max(px, np.abs(obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy().astype(int) - frames[m][t].astype(int)).max())
        pr = max(pr, np.abs(proprio(env) - data["proprio"][m][t]).max())
        obs, _, _, _, info = env.step(a)
    eq.append(dict(episode=ep, max_pixel_diff=int(px), max_proprio_diff=float(pr), replay_success=bool(info["success"].item())))
    print("pipeline", eq[-1], flush=True)
metrics["pipeline_equality"] = eq


# 2. state BC
def privileged(env, obs=None):
    # z = [cube pose (7), goal position (3)] (1, 10)
    u = env.unwrapped
    return torch.cat([u.cube.pose.raw_pose, u.goal_site.pose.p], 1).float().to(dev)


demos = np.load(DEMOS / "demos.npz")
senv = make_env(obs_mode="state", control_mode="pd_ee_delta_pose")
senv.reset(seed=0)
priv = []
for s in demos["state"]:
    senv.unwrapped.set_state(torch.from_numpy(s[None]))
    priv.append(privileged(senv))
Z = torch.cat(priv)  # (T, 10)
P = torch.from_numpy(demos["proprio"]).float().to(dev)
A = torch.from_numpy(demos["action"]).float().to(dev)
torch.manual_seed(SEED)
ctrl = Controller(Z.shape[1], P.shape[1], A.shape[1]).to(dev)
ctrl.p_mean.copy_(P.mean(0)), ctrl.p_std.copy_(P.std(0) + 1e-6)
opt = torch.optim.Adam(ctrl.parameters(), lr=LR)
t0 = time.time()
for step in range(STEPS):
    idx = torch.randint(len(A), (BATCH,), device=dev)
    loss = F.mse_loss(ctrl(Z[idx], P[idx]), A[idx])
    opt.zero_grad(), loss.backward(), opt.step()
    if step % 5000 == 0 or step == STEPS - 1:
        print(f"state BC step {step} loss {loss.item():.5f}", flush=True)
metrics["state_bc"] = dict(train_time_s=time.time() - t0, final_loss=loss.item())
ctrl.eval()
for name, seed0, n in [("eval_seeds", 10_000, 100), ("train_seeds_0_9", 0, 10)]:
    succ, _, _ = evaluate(privileged, ctrl, (0, 0, 0), n, seed0=seed0)
    metrics["state_bc"][name] = dict(success=float(succ.mean()))
    print(f"state BC {name}: success {succ.mean():.2f}", flush=True)

# 3. pixel agents on training initial states
for c, run in PIXEL_RUNS.items():
    ck = torch.load(run / "seed0.pt")
    enc, pc = Encoder().to(dev), Controller().to(dev)
    enc.load_state_dict(ck["encoder"]), pc.load_state_dict(ck["controller"], strict=False)  # no action buffers: mean 0, std 1
    enc.eval(), pc.eval()
    succ, _, _ = evaluate(pixel_encoder(enc), pc, (c, 0, 0), 10, seed0=0)
    metrics[f"pixel_cam{c}_seed0_train_seeds_0_9"] = dict(success=float(succ.mean()))
    print(f"pixel BC cam{c} seed0 on training seeds 0-9: success {succ.mean():.2f}", flush=True)

json.dump(dict(seed=SEED, steps=STEPS, batch=BATCH, lr=LR, demos=str(DEMOS), z="cube raw pose (7) + goal pos (3)"),
          open(OUT / "config.json", "w"), indent=1)
json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1)
