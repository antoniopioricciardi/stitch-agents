"""Step B: DART (Kang et al., arXiv 2607.00666), domain arithmetic on DP agents: merge, then closed loop.

Usage: uv run python scripts/stepB_dart.py <name> <base_run_dir> <src_ft_run_dir> <tgt_ft_run_dir> <env_id> [--coef=0.8] [--episodes=250]
Run from the repo root with PYTHONPATH=<repo>:<repo>/third_party/maniskill_diffusion_policy.

Weights: the EMA agents of the base's final checkpoint and of each fine-tune's final checkpoint. A torch port of their
domain_arithmetic/dart.py (merge scope "all"): tau = w_ft - w_base; per tensor, delta = _greedy_subspace_align_merge
(tau_src, tau_tgt), merged = base + delta. Matrices with more than 2 dims (conv kernels) are flattened to (out, -1), as
their reshape of the patch-embedding kernel to (Out, In*H*W). SVDs in float64.
Writes results/<date>_stepB_dart_<name>/{metrics.json (closed loop + per-tensor SAR / kept columns), config.json}.
"""
import glob
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1b_stitch import ENV_KWARGS, load_agent
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

DEV = "cuda"
ENERGY = 0.9975  # their cutoff for the target basis used in SAR ("remove noise")


def final_sd(run_dir):
    ckpt = max(glob.glob(f"{run_dir}/runs/*/checkpoints/[0-9]*.pt"), key=lambda p: int(Path(p).stem))
    return torch.load(ckpt, map_location=DEV)["ema_agent"], ckpt


def dart_delta(tau_src, tau_tgt, coef):
    # their _greedy_subspace_align_merge. tau_*: same shape. -> (delta, info)
    if tau_tgt.ndim <= 1:
        return coef * (tau_tgt - tau_src), None
    A, Bt = tau_src.reshape(len(tau_src), -1).double(), tau_tgt.reshape(len(tau_tgt), -1).double()
    u_t, s_t, _ = torch.linalg.svd(Bt, full_matrices=False)
    u_s, _, _ = torch.linalg.svd(A, full_matrices=False)
    # SAR: share of tau_src inside the (energy-filtered) span of tau_tgt
    e = torch.cumsum(s_t ** 2, 0) / ((s_t ** 2).sum() + 1e-6)
    keep_t = (torch.arange(len(s_t), device=s_t.device) <= torch.searchsorted(e, torch.tensor([ENERGY], device=e.device, dtype=e.dtype))).double()
    basis = u_t * keep_t
    sar = torch.linalg.norm(basis @ (basis.T @ A)) / (torch.linalg.norm(A) + 1e-6)
    # source directions that overlap the target span the most, until SAR of the overlap energy is covered
    energies = torch.linalg.norm(u_t.T @ u_s, dim=0) ** 2
    srt = torch.sort(energies, descending=True).values
    k = torch.clamp(torch.searchsorted(torch.cumsum(srt, 0), (sar * energies.sum()).reshape(1)), 0, len(srt) - 1)
    u_common = u_s * (energies >= srt[k]).double()
    delta = coef * sar * (Bt - u_common @ (u_common.T @ A))
    return delta.reshape(tau_tgt.shape).to(tau_tgt.dtype), dict(sar=float(sar), kept=int((energies >= srt[k]).sum()), cols=len(energies))


if __name__ == "__main__":
    NAME, BASE, SRC, TGT, ENV_ID = sys.argv[1:6]
    opt = lambda k, d: next((a.split("=")[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    COEF, N_EPISODES = float(opt("coef", 0.8)), int(opt("episodes", 250))
    OUT = Path("results") / f"{date.today():%Y%m%d}_stepB_dart_{NAME}"
    OUT.mkdir(parents=True, exist_ok=True)
    (wb, cb), (ws, cs), (wt, ct) = final_sd(BASE), final_sd(SRC), final_sd(TGT)
    merged, info = {}, {}
    for k, w in wb.items():
        if not torch.is_floating_point(w):
            merged[k] = w
            continue
        d, info[k] = dart_delta(ws[k] - w, wt[k] - w, COEF)
        merged[k] = w + d
    agent = load_agent(BASE)[0]
    agent.load_state_dict(merged)
    envs = make_eval_envs(ENV_ID, 10, "physx_cpu", ENV_KWARGS, dict(obs_horizon=2), video_dir=None, wrappers=[FlattenRGBDObservationWrapper])
    ev = evaluate(N_EPISODES, agent, envs, DEV, "physx_cpu", progress_bar=False)
    envs.close()
    sars = [v["sar"] for v in info.values() if v]
    m = dict(checkpoints=dict(base=cb, src=cs, tgt=ct), success_once=float(ev["success_once"].mean()),
             success_at_end=float(ev["success_at_end"].mean()), episodes=len(ev["success_once"]),
             sar_mean=float(np.mean(sars)), sar_min=float(np.min(sars)), per_tensor={k: v for k, v in info.items() if v})
    print(NAME, {k: m[k] for k in ("success_once", "success_at_end", "sar_mean", "sar_min")}, flush=True)
    json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
    json.dump(dict(base=BASE, src_ft=SRC, tgt_ft=TGT, env=ENV_ID, coef=COEF, energy=ENERGY, n_episodes=N_EPISODES, merge_scope="all"),
              open(OUT / "config.json", "w"), indent=1)
