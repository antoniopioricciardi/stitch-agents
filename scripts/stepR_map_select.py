"""Map-class selection from held-out label pairs (offline, CPU): nn_affine vs nn_orth, per stitch, by three criteria.

For every stitch already evaluated in closed loop with both maps (Step 1b main matrix: 4 shifts x 3 seed pairs x 2
directions; Step R goal shift and embodiment: 3 pairs each), with the checkpoints recorded in its metrics.json:
  fit (as in step1b_stitch.py): source demos 0-49, each frame paired with the target frame of demos 50-99 whose
    z-scored 8-step chunk is nearest; nn_affine = affine least squares, nn_orth = orthogonal Procrustes.
  held-out label pairs: the same nearest-chunk pairing, source demos 400-448 vs target demos 449-497 (no paired frames).
  criteria, each picking one map (all on the held-out pairs):
    resid: ||T(z_s) - z_t||^2 / ||z_t - mean||^2, lower wins (rewards shrinkage when the pairs are noisy);
    resid_vm: the same after variance matching: T's output standardised per dimension with the stats of the mapped
      fit-source z, then given the paired fit-target z's per-dimension mean and std; lower wins;
    cos: mean cosine between T(z_s) and z_t, each centred by its own held-out mean; higher wins (a uniform shrink
      cannot improve it).
  variance kept = total variance of T(z_s) / total variance of z_t, on the held-out frames of each side.
Per criterion: agreement with the closed-loop winner (higher success_once of the two), and the % of the affine ceiling of
the map it selects, vs always nn_affine and always nn_orth.
Usage: PYTHONPATH=<repo>:<repo>/scripts:<repo>/third_party/maniskill_diffusion_policy uv run python scripts/stepR_map_select.py
Writes results/<date>_stepR_map_select/{metrics.json, table.md}.
"""
import json
from datetime import date
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import step1b_stitch as S
from stitch.align import fit_affine_paired, fit_procrustes_paired
from stitch.labels import standardise

S.DEV = "cpu"
B = Path("/home/ricc/projects/labelstitch-1b/results")
R = Path("results")
FIT_SRC, FIT_TGT = range(0, 50), range(50, 100)
HO_SRC, HO_TGT = range(400, 449), range(449, 498)


def stitches():
    # -> list of dict(group, name, u, v, ckpt_u, ckpt_v, success={affine, nn_orth, nn_affine})
    out = []
    def add(group, name, m_ck, m_cl, key, extra=None):
        u, v = key.split("_enc_to_")[0], key.split("_enc_to_")[1][:-len("_ctrl")]
        succ = {a: x["success_once"] for a, x in m_cl["stitch"][key]["aligners"].items()}
        succ.update(extra or {})
        out.append(dict(group=group, name=f"{name} {u}->{v}", u=u, v=v, ckpt_u=m_ck["checkpoints"][u],
                        ckpt_v=m_ck["checkpoints"][v], success={a: succ[a] for a in ("affine", "nn_orth", "nn_affine")}))
    load = lambda p: json.load(open(p / "metrics.json"))
    for d in ("look1", "look2", "look2light1", "cam3"):
        for p in ("p1", "p2", "p3"):
            if p == "p1" and d == "look1":  # the item-1 pilot, assembled from two runs (as step1b_matrix_table.py)
                m, mc = load(B / "20261001_step1b_stitch_row1labelmaps_look1"), load(B / "20261001_step1b_stitch_row1mapclass_look1")
            else:
                m = mc = load(B / (f"20261002_step1b_stitch_pilot_{d}" if p == "p1" else f"20261003_step1b_stitch_matrix_{p}_{d}"))
            for key in m["stitch"]:
                add("matrix", f"{d} {p}", m, m, key, {"affine": mc["stitch"][key]["aligners"]["affine"]["success_once"]})
    for k, dom, group in (("goal", "goal_look2", "goal"), ("emb", "xarm_look2", "embodiment")):
        for p in (1, 2, 3):
            m = load(next(R.glob(f"2026*_step1b_stitch_stepR_{k}_p{p}_{dom}")))
            for key in m["stitch"]:
                add(group, f"{k} p{p}", m, m, key)
    return out


def domain_data(dom, ckpts):
    # frames of demos 0-99 and 400-497 of domain `dom`, encoded by each checkpoint in ckpts ->
    # {ckpt: z (N, 256)}, chunks (N, 32) z-scored, demo (N,)
    S.D = {0: dom}
    tmp = gym.make(S.ENV[dom], **S.ENV_KWARGS)
    obs_space = tmp.observation_space
    tmp.close()
    Z, X, demo = {c: [] for c in ckpts}, [], []
    for demos in (range(0, 100), range(400, 498)):
        rgb, idx, _, chunks, _, dm = S.load_frames(0, demos, obs_space)
        for c in ckpts:
            agent, _ = S.load_agent(c.rsplit("/runs/", 1)[0])
            Z[c].append(S.encode(agent.visual_encoder, rgb[idx[:, -1]]).numpy())
        X.append(standardise(chunks, LAB["mean"].numpy(), LAB["std"].numpy()))
        demo.append(dm)
        del rgb
    return {c: np.concatenate(z) for c, z in Z.items()}, np.concatenate(X), np.concatenate(demo)


def nn_pairs(Xs, Xt):
    # nearest z-scored chunk: for each source row, the index of the nearest target row
    return torch.cdist(torch.from_numpy(Xs).float(), torch.from_numpy(Xt).float()).argmin(1).numpy()


def resid(R_, b, Zs, Zt):
    M = Zs @ R_.T + b
    return float(((M - Zt) ** 2).sum() / ((Zt - Zt.mean(0)) ** 2).sum())


def var_match(R_, b, Zs_fit, Zt_fit):
    # (R, b) -> (R', b') whose output on the fit pairs has the paired target's per-dimension mean and std
    M = Zs_fit @ R_.T + b
    s = Zt_fit.std(0) / (M.std(0) + 1e-8)
    return R_ * s[:, None], (b - M.mean(0)) * s + Zt_fit.mean(0)


def cos(R_, b, Zs, Zt):
    M = Zs @ R_.T + b
    M, Zt = M - M.mean(0), Zt - Zt.mean(0)
    return float(((M * Zt).sum(1) / (np.linalg.norm(M, axis=1) * np.linalg.norm(Zt, axis=1) + 1e-8)).mean())


CRIT = {"resid": min, "resid_vm": min, "cos": max}  # how each criterion picks


def var_kept(R_, b, Zs, Zt):
    return float((Zs @ R_.T + b).var(0).sum() / Zt.var(0).sum())


if __name__ == "__main__":
    LAB = torch.load("results/20261001_step1b_labels/labels.pt")
    ST = stitches()
    need = {}  # domain -> checkpoints to run on its frames
    for s in ST:
        need.setdefault(s["u"], set()).add(s["ckpt_u"])
        need.setdefault(s["v"], set()).add(s["ckpt_v"])
    data = {}
    for dom, ck in need.items():
        data[dom] = domain_data(dom, sorted(ck))
        print(f"{dom}: {len(data[dom][2])} frames, {len(ck)} encoders", flush=True)

    rows = []
    for s in ST:
        (Zu, Xu, du), (Zv, Xv, dv) = data[s["u"]], data[s["v"]]
        Zs, Zt = Zu[s["ckpt_u"]], Zv[s["ckpt_v"]]
        sel = lambda d, r: np.isin(d, list(r))
        fs, ft, hs, ht = np.where(sel(du, FIT_SRC))[0], np.where(sel(dv, FIT_TGT))[0], np.where(sel(du, HO_SRC))[0], np.where(sel(dv, HO_TGT))[0]
        pf, ph = ft[nn_pairs(Xu[fs], Xv[ft])], ht[nn_pairs(Xu[hs], Xv[ht])]
        maps = {"nn_affine": fit_affine_paired(Zs[fs], Zt[pf]), "nn_orth": fit_procrustes_paired(Zs[fs], Zt[pf])}
        score = {"resid": {a: resid(*maps[a], Zs[hs], Zt[ph]) for a in maps},
                 "resid_vm": {a: resid(*var_match(*maps[a], Zs[fs], Zt[pf]), Zs[hs], Zt[ph]) for a in maps},
                 "cos": {a: cos(*maps[a], Zs[hs], Zt[ph]) for a in maps}}
        pick = {c: CRIT[c](score[c], key=score[c].get) for c in CRIT}
        vk = {a: var_kept(*maps[a], Zs[hs], Zt[ht]) for a in maps}
        cl = s["success"]
        win = "nn_affine" if cl["nn_affine"] >= cl["nn_orth"] else "nn_orth"
        pct = {a: 100 * cl[a] / cl["affine"] for a in maps}
        rows.append(dict(group=s["group"], name=s["name"], score=score, var_kept=vk, pick=pick, closed_loop_winner=win,
                         pct_ceiling=pct, success=cl))
        print(f"{s['name']:40s} " + " | ".join(f"{c} aff {score[c]['nn_affine']:.3f} orth {score[c]['nn_orth']:.3f} -> {pick[c][3:]}" for c in CRIT)
              + f" | var kept aff {vk['nn_affine']:.2f} orth {vk['nn_orth']:.2f} | closed loop {win[3:]} "
              f"({pct['nn_affine']:.0f}% vs {pct['nn_orth']:.0f}%)", flush=True)

    out = R / f"{date.today():%Y%m%d}_stepR_map_select"
    out.mkdir(parents=True, exist_ok=True)
    ms = lambda v: f"{np.mean(v):.0f} ± {np.std(v, ddof=1):.0f}%"
    lines = ["| group | n | closed-loop winner nn_affine / nn_orth | always nn_affine | always nn_orth | var kept nn_affine / nn_orth | "
             + " | ".join(f"{c}: agrees, picks aff / orth, selected" for c in CRIT) + " |", "|" + "---|" * (6 + len(CRIT))]
    summ = {}
    for g in ("matrix", "goal", "embodiment", "all"):
        G = [x for x in rows if g == "all" or x["group"] == g]
        n_aff_win = sum(x["closed_loop_winner"] == "nn_affine" for x in G)
        summ[g] = dict(n=len(G), closed_loop_affine_wins=n_aff_win,
                       always_affine=float(np.mean([x["pct_ceiling"]["nn_affine"] for x in G])),
                       always_orth=float(np.mean([x["pct_ceiling"]["nn_orth"] for x in G])),
                       oracle_pick=float(np.mean([max(x["pct_ceiling"].values()) for x in G])), crit={})
        cells = []
        for c in CRIT:
            sel = [x["pct_ceiling"][x["pick"][c]] for x in G]
            agree = sum(x["pick"][c] == x["closed_loop_winner"] for x in G)
            n_aff = sum(x["pick"][c] == "nn_affine" for x in G)
            summ[g]["crit"][c] = dict(agree=agree, picks_affine=n_aff, selected=float(np.mean(sel)))
            cells.append(f"{agree} / {len(G)}, {n_aff} / {len(G) - n_aff}, {ms(sel)}")
        lines.append(f"| {g} | {len(G)} | {n_aff_win} / {len(G) - n_aff_win} | {ms([x['pct_ceiling']['nn_affine'] for x in G])} | "
                     f"{ms([x['pct_ceiling']['nn_orth'] for x in G])} | {np.mean([x['var_kept']['nn_affine'] for x in G]):.2f} / "
                     f"{np.mean([x['var_kept']['nn_orth'] for x in G]):.2f} | " + " | ".join(cells) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    json.dump(dict(rows=rows, summary=summ, fit=[[FIT_SRC.start, FIT_SRC.stop], [FIT_TGT.start, FIT_TGT.stop]],
                   heldout=[[HO_SRC.start, HO_SRC.stop], [HO_TGT.start, HO_TGT.stop]]), open(out / "metrics.json", "w"), indent=1)
    print("\n".join(lines))
