"""Step 1b, offline (CPU), round 2: nearest-neighbour label pairs x map class, shrinkage, proprio, mutual NN, fair ceiling.

Usage: uv run python scripts/step1b_label_maps2.py <cam0_run_dir> <look1_run_dir> <labels.pt>
  (PYTHONPATH=<repo>:<repo>/scripts:<repo>/third_party/maniskill_diffusion_policy; as step1b_label_maps.py, all model
  work on the CPU, the GPU touched only by SAPIEN for the one env made to read the observation space.)

Source frames from demos 0-49 (source domain), target frames from demos 50-99 (target domain), as in round 1.
Pairing features (frames of the two halves matched by nearest neighbour, Euclidean):
  chunk:       z-scored 8-step action chunk (32-d), as round 1's nn
  chunk_prop:  [chunk / sqrt(32), proprio / sqrt(9)], proprio = tcp pose (7) + gripper finger qpos (2), z-scored over the
               fit frames; the 1/sqrt(dim) scaling makes both blocks weigh the same
  each also with mutual-NN filtering (_mnn): keep source i only if i is the nearest source frame of its target match.
Maps on the pairs: orth (Procrustes), affine (least squares), orth_scale / affine_rescale (the same map, then each
output dimension rescaled to the target's per-dimension mean and std; stats of all mapped source frames vs all target
frames of the two halves, no pairing used). All of them are (R, b) maps.
References: identity; saps and affine paired on demos 0-99 (the current ceiling); affine paired on demos 0-49 (the fair
ceiling: the label-based fits' frame budget); k16 orth (round 1's action_pairs, draw 0).
Metrics on the 2000 held-out frames of step1b_stitch.py: z residual, shrinkage (total variance of mapped z / target z,
and the median per-dimension ratio over dimensions holding target variance), chunk distance [all, top 30%].
"""
import json
import sys
from datetime import date
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import step1b_stitch as S
from stitch.align import action_pairs, fit_affine_paired, fit_identity, fit_procrustes_paired
from stitch.labels import assign, standardise

HALF = 50
PROPRIO = [7, 8, 18, 19, 20, 21, 22, 23, 24]  # state = qpos 9 (fingers 7, 8), qvel 9, tcp_pose 7 (18-24), goal_pos 3


def nn_pairs(F, src, tgt, mutual):
    # F (N, k) pairing features -> (ia, ib) frame indices: each source frame with its nearest target frame
    d = torch.cdist(F[src], F[tgt])  # (n_src, n_tgt)
    j = d.argmin(1)
    keep = torch.arange(len(src))
    if mutual:
        keep = keep[d.argmin(0)[j] == keep]  # i is the nearest source of its own nearest target
    return src[keep.numpy()], tgt[j[keep].numpy()]


def rescale(R, b, Zs_all, Zt_all):
    # (R, b) -> (R', b'): mapped z standardised per dimension, then given the target's per-dimension mean and std
    M = Zs_all @ R.T + b
    s = Zt_all.std(0) / (M.std(0) + 1e-8)
    return R * s[:, None], (b - M.mean(0)) * s + Zt_all.mean(0)


if __name__ == "__main__":
    S.DEV = "cpu"
    S.D = {0: "cam0", 1: "look1"}
    torch.set_num_threads(3)  # 6 cores: leave half for the training job and its eval workers
    RUNS = {0: sys.argv[1], 1: sys.argv[2]}
    lab = torch.load(sys.argv[3])
    OUT = Path("results") / f"{date.today():%Y%m%d}_step1b_label_maps2"
    OUT.mkdir(parents=True, exist_ok=True)

    tmp = gym.make(S.ENV["cam0"], **S.ENV_KWARGS)
    obs_space = tmp.observation_space
    tmp.close()
    agents = {c: S.load_agent(RUNS[c])[0] for c in (0, 1)}
    fit = {c: S.load_frames(c, S.FIT_DEMOS, obs_space) for c in (0, 1)}
    off = {c: S.load_frames(c, S.HELDOUT_DEMOS, obs_space) for c in (0, 1)}
    Z = {c: S.encode(agents[c].visual_encoder, fit[c][0][fit[c][1][:, -1]]).numpy().astype(np.float64) for c in (0, 1)}
    demo = fit[0][5]
    src, tgt = np.where(demo < HALF)[0], np.where(demo >= HALF)[0]

    chunk = standardise(fit[0][3], lab["mean"].numpy(), lab["std"].numpy())  # (N, 32), the same in both domains
    prop = fit[0][2][:, -1, PROPRIO].numpy()  # (N, 9) current-frame proprio, the same in both domains
    prop = (prop - prop.mean(0)) / prop.std(0)
    feats = {"chunk": torch.from_numpy(chunk).float(),
             "chunk_prop": torch.from_numpy(np.concatenate([chunk / np.sqrt(32), prop / np.sqrt(9)], 1)).float()}
    pairs = {f"{f}{'_mnn' if mu else ''}": nn_pairs(F, src, tgt, mu) for f, F in feats.items() for mu in (False, True)}
    y16 = assign(feats["chunk"], lab["centroids"].float()).numpy()

    rng = np.random.default_rng(S.SEED)  # held-out frames: the same selection as step1b_stitch.py
    y_off = assign(torch.from_numpy(standardise(off[0][3], lab["mean"].numpy(), lab["std"].numpy())).float(),
                   lab["centroids"].float()).numpy()
    sel = rng.permutation(len(y_off))[:S.N_OFF]
    perm = rng.permutation(S.N_OFF)

    m = dict(n_pairs={k: len(v[0]) for k, v in pairs.items()}, directions={})
    print(m["n_pairs"], flush=True)
    for u, v in ((0, 1), (1, 0)):
        Zs, Zt = Z[u], Z[v]
        ia16, ib16 = action_pairs(y16[src], y16[tgt], np.random.default_rng(0))
        fits = {"identity": fit_identity(Zs, Zt), "saps": fit_procrustes_paired(Zs, Zt),
                "affine_paired_0-99": fit_affine_paired(Zs, Zt),
                "affine_paired_0-49": fit_affine_paired(Zs[src], Zt[src]),  # same demo and step in both domains
                "k16_orth": fit_procrustes_paired(Zs[src[ia16]], Zt[tgt[ib16]])}
        for pname, (ia, ib) in pairs.items():
            for mname, f in (("orth", fit_procrustes_paired), ("affine", fit_affine_paired)):
                R, b = f(Zs[ia], Zt[ib])
                fits[f"{pname}_{mname}"] = (R, b)
                fits[f"{pname}_{mname}_{'scale' if mname == 'orth' else 'rescale'}"] = rescale(R, b, Zs[src], Zt[tgt])

        rgb, state = off[u][0][off[u][1][sel]], off[u][2][sel]
        zu = S.encode(agents[u].visual_encoder, rgb.flatten(0, 1)).reshape(len(sel), 2, -1)
        zv = S.encode(agents[v].visual_encoder, off[v][0][off[v][1][sel][:, -1]])  # (N, 256) paired target z
        native = S.denoise(agents[u], zu, state, seed=1)
        per_frame = lambda a, b: (a - b).flatten(1).norm(dim=1)
        sens = per_frame(native, S.denoise(agents[u], zu[perm], state, seed=1))
        top = sens >= sens.quantile(1 - S.TOP_SENSITIVE)
        vt = zv.var(0)
        live = vt > 1e-3 * vt.max()  # dimensions that hold target variance
        res = dict(chance_z=[sens.mean().item(), sens[top].mean().item()], maps={})
        for name, (R, b) in fits.items():
            R_, b_ = torch.tensor(R, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
            mapped = zu[:, -1] @ R_.T + b_
            d = per_frame(S.denoise(agents[v], zu @ R_.T + b_, state, seed=1), native)
            res["maps"][name] = dict(
                z_residual=(((mapped - zv) ** 2).sum() / ((zv - zv.mean(0)) ** 2).sum()).item(),
                var_ratio_total=(mapped.var(0).sum() / vt.sum()).item(),
                var_ratio_median_dim=(mapped.var(0)[live] / vt[live]).median().item(),
                offline_dist=[d.mean().item(), d[top].mean().item()])
            print(S.D[u], "->", S.D[v], f"{name:28s}", {k: (round(x, 3) if isinstance(x, float) else [round(y, 3) for y in x])
                                                       for k, x in res["maps"][name].items()}, flush=True)
        m["directions"][f"{S.D[u]}_enc_to_{S.D[v]}_ctrl"] = res
        json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
    json.dump(dict(runs={S.D[c]: RUNS[c] for c in (0, 1)}, labels=sys.argv[3], source_demos=[0, HALF],
                   target_demos=[HALF, 100], proprio_state_idx=PROPRIO, heldout=[S.HELDOUT_DEMOS.start, S.HELDOUT_DEMOS.stop],
                   n_off=S.N_OFF, device="cpu"), open(OUT / "config.json", "w"), indent=1)
