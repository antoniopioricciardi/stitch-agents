"""Step B: label-free Gromov-Wasserstein on the main visual matrix (the 13 Oct motion matrix's 24 cells).

Usage: uv run python scripts/stepB_gw_matrix.py fit                                  (all cells, CPU-light; GPU only to encode)
       uv run python scripts/stepB_gw_matrix.py eval <shift> <pair> --eps=<e> [--episodes=250]
Run from the repo root with PYTHONPATH=<repo>:<repo>/third_party/maniskill_diffusion_policy.

Cells: shift in (look1, look2, look2light1, cam3) x pair p1-p3 x direction (fwd = cam0 encoder -> shift controller,
played in cam0; bwd = the reverse), agents from results/20261013_step5b_motion_matrix_<shift>_<pair>/config.json.
Frames as nn_affine (motion): source = the encoder's own demos 0-49, target = the controller's own demos 50-99.
fit: per cell and eps in EPS, stitch.align.fit_gw's recipe (uniform subsample of N_SUB frames per side, the same
subsample for every eps; entropic GW on [0, 1]-scaled distance matrices; no labels; a NaN plan is recorded as diverged), then two maps from the plan P:
  gw_orth   barycentric Procrustes (fit_gw, Step 4);   gw_affine   affine least squares on the barycentric targets.
Diagnostics (paired truth used only to evaluate): marginal error |P1 - p|_1 + |P^T 1 - q|_1, effective targets per
source frame (exp of the row entropy), GW objective, plan motion error (mm: source motion chunk vs the motion chunk of
its argmax target; nn_affine's pairs are ~2.5-3 mm), held-out z residual ||T(z_s) - z_t||^2 / ||z_t - mean||^2 on
paired frames (demos 400-497), with identity, nn_affine (motion) and the paired affine ceiling (demos 0-99) as
references. Writes results/<date>_stepB_gw_matrix/{maps_<shift>_<pair>_<dir>.npz, metrics.json, config.json}.
eval: closed loop of gw_orth and gw_affine at --eps for both directions of one cell (maps from the fit folder),
results/<date>_stepB_gw_matrix_<shift>_<pair>/metrics.json.
"""
import copy
import glob
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import ot
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step5b_motion_matrix as mm  # sets the domains' demo files in s5.H5
import step5b_stitch as s5
from step1b_stitch import ENV, ENV_KWARGS, affine, encode, load_agent
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from stitch.align import fit_affine_paired, fit_procrustes_paired, gw_objective, structure

SHIFTS, PAIRS = ("look1", "look2", "look2light1", "cam3"), ("p1", "p2", "p3")
EPS = (0.0005, 0.001, 0.005)
N_SUB = 1000
SEED = 0
SRC_DEMOS, TGT_DEMOS, PAIRED_DEMOS, HELDOUT_DEMOS = range(0, 50), range(50, 100), range(0, 100), range(400, 498)
FIT_DIR = Path("results/20261016_stepB_gw_matrix")
N_EPISODES = 250


def cell_runs(shift, pair):
    return json.load(open(f"results/20261013_step5b_motion_matrix_{shift}_{pair}/config.json"))["runs"]


def residual(R, b, Zs, Zt):
    return float(((Zs @ R.T + b - Zt) ** 2).sum() / ((Zt - Zt.mean(0)) ** 2).sum())


if __name__ == "__main__":
    MODE = sys.argv[1]
    opt = lambda k, d: next((a.split("=")[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    if MODE == "fit":
        FIT_DIR.mkdir(parents=True, exist_ok=True)
        m = {}
        for shift in SHIFTS:
            for pair in PAIRS:
                runs = cell_runs(shift, pair)
                agents = {d: load_agent(r)[0] for d, r in runs.items()}
                for direction, u, v in (("fwd", "cam0", shift), ("bwd", shift, "cam0")):
                    src, tgt = s5.load(u, SRC_DEMOS), s5.load(v, TGT_DEMOS)
                    enc, ctrl = agents[u].visual_encoder, agents[v].visual_encoder
                    # float64: in float32, Sinkhorn's exp(-C / eps) underflows to an all-zero plan at eps <= .001
                    Zs, Zt = encode(enc, src["rgb"]).cpu().double().numpy(), encode(ctrl, tgt["rgb"]).cpu().double().numpy()
                    # references: paired affine ceiling (demos 0-99), nn_affine on motion pairs, identity
                    pu, pv = s5.load(u, PAIRED_DEMOS), s5.load(v, PAIRED_DEMOS)
                    hu, hv = s5.load(u, HELDOUT_DEMOS), s5.load(v, HELDOUT_DEMOS)
                    Hs, Ht = encode(enc, hu["rgb"]).cpu().double().numpy(), encode(ctrl, hv["rgb"]).cpu().double().numpy()
                    j = s5.nearest(*s5.zscore(src["motion"], tgt["motion"], "pooled"))
                    maps = {"identity": (np.eye(Zs.shape[1]), np.zeros(Zs.shape[1])), "nn_affine": fit_affine_paired(Zs, Zt[j]),
                            "ceiling": fit_affine_paired(encode(enc, pu["rgb"]).cpu().numpy(), encode(ctrl, pv["rgb"]).cpu().numpy())}
                    rng = np.random.default_rng(SEED)
                    i_s, i_t = rng.permutation(len(Zs))[:N_SUB], rng.permutation(len(Zt))[:N_SUB]
                    C1, C2 = structure(Zs[i_s]), structure(Zt[i_t])
                    cell = dict(encoder=u, controller=v, gw={})
                    for eps in EPS:
                        t0 = time.time()
                        P = ot.gromov.entropic_gromov_wasserstein(C1, C2, epsilon=eps)
                        if np.isnan(P).any():  # Sinkhorn underflow at small eps: no plan (excluded by the eps rule)
                            cell["gw"][str(eps)] = dict(fit_s=round(time.time() - t0, 1), diverged=True)
                            continue
                        row = P / P.sum(1, keepdims=True)
                        bary = row @ Zt[i_t]
                        maps[f"gw_orth_eps{eps}"] = fit_procrustes_paired(Zs[i_s], bary)
                        maps[f"gw_affine_eps{eps}"] = fit_affine_paired(Zs[i_s], bary)
                        k = row.argmax(1)
                        mot = np.linalg.norm((src["motion"][i_s] - tgt["motion"][i_t][k]).reshape(N_SUB, -1), axis=1).mean() * 1e3
                        cell["gw"][str(eps)] = dict(
                            fit_s=round(time.time() - t0, 1), objective=gw_objective(C1, C2, P),
                            marginal_err=float(np.abs(P.sum(1) - 1 / N_SUB).sum() + np.abs(P.sum(0) - 1 / N_SUB).sum()),
                            eff_targets_median=float(np.median(np.exp(-(row * np.log(row + 1e-30)).sum(1)))),
                            plan_motion_err_mm=float(mot))
                    cell["nn_motion_err_mm"] = float(np.linalg.norm((src["motion"] - tgt["motion"][j]).reshape(len(j), -1), axis=1).mean() * 1e3)
                    cell["heldout_residual"] = {name: residual(R, b, Hs, Ht) for name, (R, b) in maps.items()}
                    np.savez(FIT_DIR / f"maps_{shift}_{pair}_{direction}.npz", **{f"{n}_{x}": a for n, ab in maps.items() for x, a in zip("Rb", ab)})
                    m[f"{shift}_{pair}_{direction}"] = cell
                    print(shift, pair, direction, json.dumps(cell), flush=True)
                    json.dump(m, open(FIT_DIR / "metrics.json", "w"), indent=1)
        json.dump(dict(shifts=SHIFTS, pairs=PAIRS, eps=EPS, n_sub=N_SUB, seed=SEED, src_demos=[0, 50], tgt_demos=[50, 100],
                       paired_demos=[0, 100], heldout_demos=[400, 498], labels="none (GW is label-free; motion only in diagnostics)"),
                  open(FIT_DIR / "config.json", "w"), indent=1)
    else:
        SHIFT, PAIR = sys.argv[2:4]
        EPS_CL, N_EPISODES = opt("eps", None), int(opt("episodes", N_EPISODES))
        OUT = Path("results") / f"{date.today():%Y%m%d}_stepB_gw_matrix_{SHIFT}_{PAIR}"
        OUT.mkdir(parents=True, exist_ok=True)
        runs = cell_runs(SHIFT, PAIR)
        agents = {d: load_agent(r) for d, r in runs.items()}
        m = dict(checkpoints={d: a[1] for d, a in agents.items()}, eps=float(EPS_CL), stitch={})
        for direction, u, v in (("fwd", "cam0", SHIFT), ("bwd", SHIFT, "cam0")):
            M = np.load(FIT_DIR / f"maps_{SHIFT}_{PAIR}_{direction}.npz")
            envs = make_eval_envs(ENV[u], 10, "physx_cpu", ENV_KWARGS, dict(obs_horizon=2), video_dir=None,
                                  wrappers=[FlattenRGBDObservationWrapper])
            m["stitch"][direction] = dict(encoder=u, controller=v)
            for name in ("gw_orth", "gw_affine"):
                stitched = copy.deepcopy(agents[v][0])
                stitched.visual_encoder = nn.Sequential(copy.deepcopy(agents[u][0].visual_encoder),
                                                        affine(M[f"{name}_eps{EPS_CL}_R"], M[f"{name}_eps{EPS_CL}_b"]))
                ev = evaluate(N_EPISODES, stitched, envs, "cuda", "physx_cpu", progress_bar=False)
                m["stitch"][direction][name] = dict(success_once=float(ev["success_once"].mean()),
                                                    success_at_end=float(ev["success_at_end"].mean()), episodes=len(ev["success_once"]))
            envs.close()
            print(direction, m["stitch"][direction], flush=True)
            json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
        json.dump(dict(shift=SHIFT, pair=PAIR, runs=runs, maps=str(FIT_DIR), eps=float(EPS_CL), n_episodes=N_EPISODES),
                  open(OUT / "config.json", "w"), indent=1)
