"""Step 1b, offline (CPU): label-based correspondences x map class on the row 1 agents (cam0 <-> look 1).

Usage: uv run python scripts/step1b_label_maps.py <cam0_run_dir> <look1_run_dir> <labels.pt>
  (PYTHONPATH=<repo>:<repo>/scripts:<repo>/third_party/maniskill_diffusion_policy; loaders / encoder / denoiser from
  step1b_stitch.py). All model work runs on the CPU; the GPU is touched only by SAPIEN for the one env made to read
  the observation space (with CUDA hidden, SAPIEN segfaults).

Label-based fits take source frames from demos 0-49 (source domain) and target frames from demos 50-99 (target
domain): both agents trained on the same demos, so a pair can never be the same frame of the same demo.
Correspondences:
  k16 / k64 / k256: random pairs of frames in the same chunk cluster (<=100 per cluster, N_DRAWS draws). K = 16 is the
    current labels.pt; K = 64 / 256 are fitted here the same way (k-means on the z-scored chunks of demos 0-99, seed 0).
  nn: each source frame paired with the target frame whose z-scored 8-step chunk is nearest (deterministic).
Maps on the same pairs: orth (orthogonal Procrustes = action_pairs) and affine (least squares).
References on all paired frames of demos 0-99: identity, saps (orthogonal), affine_paired (the paired ceiling).
Metrics on the held-out frames of step1b_stitch.py (same 2000 frames): z residual ||T(z_s) - z_t||^2 / ||z_t - mean||^2,
and the offline chunk distance to the native agent [all, top 30% vision-sensitive], same DDPM noise.
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

import step1b_stitch as S
from stitch.align import action_pairs, fit_affine_paired, fit_identity, fit_procrustes_paired
from stitch.labels import assign, kmeans, standardise

N_DRAWS = 5
KS = (16, 64, 256)
HALF = 50  # source demos 0-49, target demos 50-99

if __name__ == "__main__":
    S.DEV = "cpu"
    S.D = {0: "cam0", 1: "look1"}  # load_frames reads the domain names from this module global
    torch.set_num_threads(3)  # 6 cores: leave half for the training job and its eval workers
    RUNS = {0: sys.argv[1], 1: sys.argv[2]}
    lab = torch.load(sys.argv[3])
    OUT = Path("results") / f"{date.today():%Y%m%d}_step1b_label_maps"
    OUT.mkdir(parents=True, exist_ok=True)

    import gymnasium as gym
    tmp = gym.make(S.ENV["cam0"], **S.ENV_KWARGS)
    obs_space = tmp.observation_space
    tmp.close()
    agents = {c: S.load_agent(RUNS[c])[0] for c in (0, 1)}
    fit = {c: S.load_frames(c, S.FIT_DEMOS, obs_space) for c in (0, 1)}
    off = {c: S.load_frames(c, S.HELDOUT_DEMOS, obs_space) for c in (0, 1)}
    Z = {c: S.encode(agents[c].visual_encoder, fit[c][0][fit[c][1][:, -1]]).numpy() for c in (0, 1)}  # (N, 256)
    demo = fit[0][5]
    X = standardise(fit[0][3], lab["mean"].numpy(), lab["std"].numpy())  # (N, 32) z-scored chunks (same in both domains)
    Xt = torch.from_numpy(X).float()

    # labels: K = 16 from labels.pt, K = 64 / 256 fitted here on the same chunks
    labels = {16: assign(Xt, lab["centroids"].float()).numpy()}
    for K in KS[1:]:
        C, _ = kmeans(Xt, K, seed=0)
        labels[K] = assign(Xt, C).numpy()
    src, tgt = np.where(demo < HALF)[0], np.where(demo >= HALF)[0]  # frame indices of the two halves
    nn = tgt[torch.cdist(Xt[src], Xt[tgt]).argmin(1).numpy()]  # nearest target chunk for each source frame

    # held-out frames: the same selection as step1b_stitch.py
    rng = np.random.default_rng(S.SEED)
    y_off = assign(torch.from_numpy(standardise(off[0][3], lab["mean"].numpy(), lab["std"].numpy())).float(),
                   lab["centroids"].float()).numpy()
    sel = rng.permutation(len(y_off))[:S.N_OFF]
    perm = rng.permutation(S.N_OFF)

    m = dict(n_pairs={}, label_sizes={K: np.bincount(labels[K], minlength=K).tolist() for K in KS}, directions={})
    for u, v in ((0, 1), (1, 0)):
        Zs, Zt = Z[u], Z[v]
        fits = {"identity": [fit_identity(Zs, Zt)], "saps": [fit_procrustes_paired(Zs, Zt)],
                "affine_paired": [fit_affine_paired(Zs, Zt)]}
        for K in KS:
            for name, f in (("orth", fit_procrustes_paired), ("affine", fit_affine_paired)):
                fits[f"k{K}_{name}"] = []
                for s in range(N_DRAWS):
                    ia, ib = action_pairs(labels[K][src], labels[K][tgt], np.random.default_rng(s))
                    fits[f"k{K}_{name}"].append(f(Zs[src[ia]], Zt[tgt[ib]]))
                m["n_pairs"][f"k{K}"] = len(ia)
        fits["nn_orth"] = [fit_procrustes_paired(Zs[src], Zt[nn])]
        fits["nn_affine"] = [fit_affine_paired(Zs[src], Zt[nn])]
        m["n_pairs"]["nn"] = len(src)

        rgb, state = off[u][0][off[u][1][sel]], off[u][2][sel]
        zu = S.encode(agents[u].visual_encoder, rgb.flatten(0, 1)).reshape(len(sel), 2, -1)  # (N, 2, 256)
        zv = S.encode(agents[v].visual_encoder, off[v][0][off[v][1][sel][:, -1]])  # (N, 256) paired target z
        native = S.denoise(agents[u], zu, state, seed=1)
        per_frame = lambda a, b: (a - b).flatten(1).norm(dim=1)
        sens = per_frame(native, S.denoise(agents[u], zu[perm], state, seed=1))
        top = sens >= sens.quantile(1 - S.TOP_SENSITIVE)
        res = dict(chance_z=[sens.mean().item(), sens[top].mean().item()], maps={})
        for name, RBs in fits.items():
            resid, dists = [], []
            for R, b in RBs:
                R_, b_ = torch.tensor(R, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
                resid.append((((zu[:, -1] @ R_.T + b_ - zv) ** 2).sum() / ((zv - zv.mean(0)) ** 2).sum()).item())
                if len(dists) == 0:  # chunk distance on the first draw only (CPU denoising is slow)
                    d = per_frame(S.denoise(agents[v], zu @ R_.T + b_, state, seed=1), native)
                    dists = [d.mean().item(), d[top].mean().item()]
            res["maps"][name] = dict(z_residual=float(np.mean(resid)), z_residual_draws=resid, offline_dist_draw0=dists)
            print(S.D[u], "->", S.D[v], name, round(float(np.mean(resid)), 3), [round(x, 3) for x in dists], flush=True)
        m["directions"][f"{S.D[u]}_enc_to_{S.D[v]}_ctrl"] = res
        json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
    json.dump(dict(runs={S.D[c]: RUNS[c] for c in (0, 1)}, labels=sys.argv[3], ks=KS, n_draws=N_DRAWS,
                   source_demos=[0, HALF], target_demos=[HALF, 100], heldout=[S.HELDOUT_DEMOS.start, S.HELDOUT_DEMOS.stop],
                   n_off=S.N_OFF, kmeans_seed=0, device="cpu"), open(OUT / "config.json", "w"), indent=1)
