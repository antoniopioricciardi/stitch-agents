"""Step 2: BC oracles on the minimal slice (default task, Panda, one camera), 3 seeds, then closed-loop evaluation.

Usage: uv run python scripts/train_bc.py <cam>

Trains Encoder + Controller end to end with MSE on the 7-d EE delta action, from all 500 demos of
results/20260929_step1_demos_default_panda/cam<cam>_look0_light0.npz. Evaluates each agent on 100 episodes
(seeds 10000+, disjoint from the demos) with z = E(obs) (full), z = 0 (zeros) and z = latent of a random
training frame (shuffled): the blind checks say how much the controller solves from proprio alone.
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
from stitch.evaluate import EVAL_SEED0, evaluate, pixel_encoder
from stitch.models import Controller, Encoder, random_shift

CAM = int(sys.argv[1])
SEEDS = [0, 1, 2]
D = 256
STEPS = 20_000
BATCH = 256
LR = 3e-4
N_EVAL = 100
DEMOS = Path("results/20260929_step1_demos_default_panda") / f"cam{CAM}_look0_light0.npz"
OUT = Path("results") / f"{date.today():%Y%m%d}_step2_bc_cam{CAM}"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"

data = np.load(DEMOS)
obs = torch.from_numpy(data["obs"]).to(dev)  # (T, 128, 128, 3) uint8, ~1.8 GB for 500 demos
prop = torch.from_numpy(data["proprio"]).float().to(dev)  # (T, 25)
act = torch.from_numpy(data["action"]).float().to(dev)  # (T, 7)
print(f"{len(obs)} frames from {len(np.unique(data['episode']))} demos, cam{CAM}")

metrics = {}
for seed in SEEDS:
    torch.manual_seed(seed), np.random.seed(seed)
    enc, ctrl = Encoder(D).to(dev), Controller(D, prop.shape[1], act.shape[1]).to(dev)
    ctrl.p_mean.copy_(prop.mean(0)), ctrl.p_std.copy_(prop.std(0) + 1e-6)
    opt = torch.optim.Adam(list(enc.parameters()) + list(ctrl.parameters()), lr=LR)

    ckpt = OUT / f"seed{seed}.pt"
    resumed = ckpt.exists()  # trained by an earlier run that crashed during evaluation: reuse it
    if resumed:
        c = torch.load(ckpt)
        enc.load_state_dict(c["encoder"]), ctrl.load_state_dict(c["controller"])
        m = dict(train_time_s=c.get("train_time_s"), final_loss=c.get("final_loss"))
    else:
        t0 = time.time()
        for step in range(STEPS):
            idx = torch.randint(len(obs), (BATCH,), device=dev)
            loss = F.mse_loss(ctrl(enc(random_shift(obs[idx])), prop[idx]), act[idx])
            opt.zero_grad(), loss.backward(), opt.step()
            if step % 2000 == 0 or step == STEPS - 1:
                print(f"seed {seed} step {step} loss {loss.item():.4f}", flush=True)
        m = dict(train_time_s=time.time() - t0, final_loss=loss.item())
        torch.save(dict(encoder=enc.state_dict(), controller=ctrl.state_dict(), seed=seed, cam=CAM, **m), ckpt)

    # evaluation: full, zeros, shuffled (z bank = latents of 2048 random training frames)
    enc.eval(), ctrl.eval()
    with torch.no_grad():
        z_bank = torch.cat([enc(obs[i]) for i in torch.randperm(len(obs), device=dev)[:2048].split(256)])
    for mode in ["full", "zeros", "shuffled"]:
        t0 = time.time()
        succ, lens = evaluate(pixel_encoder(enc), ctrl, (CAM, 0, 0), N_EVAL, z_mode=mode, z_bank=z_bank)
        m[mode] = dict(success=float(succ.mean()), mean_len=float(lens.mean()), eval_time_s=time.time() - t0)
        print(f"seed {seed} {mode}: success {succ.mean():.2f}, mean length {lens.mean():.1f}", flush=True)
    metrics[f"seed{seed}"] = m
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1)

for mode in ["full", "zeros", "shuffled"]:
    s = [metrics[f"seed{k}"][mode]["success"] for k in SEEDS]
    metrics[f"{mode}_mean"], metrics[f"{mode}_std"] = float(np.mean(s)), float(np.std(s))
    print(f"{mode}: {np.mean(s):.2f} ± {np.std(s):.2f}")
json.dump(dict(cam=CAM, seeds=SEEDS, d=D, steps=STEPS, batch=BATCH, lr=LR, aug="random_shift pad 4", demos=str(DEMOS),
               n_eval=N_EVAL, eval_seed0=EVAL_SEED0, task="default", robot="panda", control_mode="pd_ee_delta_pose"),
          open(OUT / "config.json", "w"), indent=1)
json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1)
