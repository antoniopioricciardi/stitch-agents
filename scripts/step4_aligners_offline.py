# Step 4, offline on exported Mario latents: identity / SAPS / action_pairs vs GW and action-pair + fused GW.
# Metric: on held-out frames, agreement of the stitched agent (source encoder -> map -> other controller) with
# the native agent of the played domain, exactly as mario/anchors_offline.py:
#   versions: 1-1 v_u encoder (seed s) + v_v controller (seed s+1); native = v_v agent (seed s+1) on the same
#             frames in v_v. Unpaired fits use disjoint episodes (source TRAIN_EPS[:5], target TRAIN_EPS[5:]);
#             SAPS uses all paired training frames.
#   levels:   level a encoder (seed s) + level b controller (seed s+1), played on a; native = level a agent
#             (seed s). Unpaired fits on the full training sets.
#   halves:   (full mode, bc / scil) v0 -> v1 as in versions, but source frames only from the first half of 1-1
#             (x-position below the training median) and target frames only from the second half, so the two
#             clouds no longer share a state distribution. Evaluated on all held-out frames. No SAPS.
# Modes:
#   repro  cheap aligners on all 6 version permutations + levels, compared with 20260929_anchors_offline*
#   pilot  seed 0, bc / scil, one version pair and one level direction, one draw of GW / FGW per epsilon
#   full   v1 version pairs + levels, all aligners
# Run: uv run python scripts/step4_aligners_offline.py {repro|pilot|full}
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stitch.align import fit_action_pairs, fit_action_pairs_fgw, fit_gw, fit_identity, fit_procrustes_paired

MODE = sys.argv[1]
ROOT = Path(__file__).resolve().parent.parent / "results"
LAT = ROOT / "20260929_step4_latents"
OUT = ROOT / f"20260929_step4_aligners_offline_{MODE}"
SEEDS = [0, 1, 2] if MODE != "pilot" else [0]
METHODS = ["bc", "scil", "scil_taco3"] if MODE != "pilot" else ["bc", "scil"]
TRAIN_EPS = [1, 2, 3, 4, 5, 8, 9, 11, 12, 13]  # mario/data.py, 1-1
SRC_EPS, TGT_EPS = TRAIN_EPS[:5], TRAIN_EPS[5:]
ALL_VERSION_PAIRS = list(itertools.permutations([0, 1, 2], 2))
VERSION_PAIRS = {"repro": ALL_VERSION_PAIRS, "pilot": [(0, 1)], "full": [(0, 1), (1, 0), (1, 2), (2, 1)]}[MODE]
LEVEL_PAIRS = [("1-1", "1-2"), ("1-2", "1-1")] if MODE != "pilot" else [("1-1", "1-2")]  # (played, controller)
EPS = [0.0005, 0.001, 0.005]  # entropic regularisation on [0, 1]-scaled costs; chosen by plan sharpness (pilot)
N_DRAWS, N_DRAWS_GW = 5, 3 if MODE != "pilot" else 1
GW = MODE != "repro"
HALF_METHODS = ["bc", "scil"] if MODE == "full" else []
REF = {"bc": (.718, .630, .682), "scil": (.762, .754, .844), "scil_taco3": (.768, .750, .827)}  # saps, pairs v / l

_cache = {}


def load(method, seed, level, version):
    k = (method, seed, level, version)
    if k not in _cache:
        _cache[k] = dict(np.load(LAT / f"{method}_s{seed}_{level}_v{version}.npz"))
    return _cache[k]


def act(ctrl, Z):
    return (Z @ ctrl["W"].T.astype(np.float64) + ctrl["b"]).argmax(1)


def cases(method, seed):
    # Yields (setting, pair, Zs, ys, Zt, yt, Zs_va, ctrl, native, paired) with float64 latents.
    cs = (seed + 1) % len([0, 1, 2])
    for u, v in VERSION_PAIRS:
        a, b = load(method, seed, "1-1", u), load(method, cs, "1-1", v)
        src, tgt = np.isin(a["ep_tr"], SRC_EPS), np.isin(b["ep_tr"], TGT_EPS)
        Za, Zb = a["Z_tr"].astype(np.float64), b["Z_tr"].astype(np.float64)
        yield ("versions", f"v{u}->v{v}", Za[src], a["y_tr"][src], Zb[tgt], b["y_tr"][tgt],
               a["Z_va"].astype(np.float64), b, act(b, b["Z_va"].astype(np.float64)), (Za, Zb))
    if method in HALF_METHODS:
        a, b = load(method, seed, "1-1", 0), load(method, cs, "1-1", 1)
        mid = np.median(a["x_tr"])
        src = np.isin(a["ep_tr"], SRC_EPS) & (a["x_tr"] < mid)
        tgt = np.isin(b["ep_tr"], TGT_EPS) & (b["x_tr"] >= mid)
        yield ("halves", "v0->v1 first half -> second half", a["Z_tr"][src].astype(np.float64), a["y_tr"][src],
               b["Z_tr"][tgt].astype(np.float64), b["y_tr"][tgt], a["Z_va"].astype(np.float64), b,
               act(b, b["Z_va"].astype(np.float64)), None)
    for la, lb in LEVEL_PAIRS:
        a, b = load(method, seed, la, 0), load(method, cs, lb, 0)
        Za_va = a["Z_va"].astype(np.float64)
        yield ("levels", f"play {la} ctrl {lb}", a["Z_tr"].astype(np.float64), a["y_tr"],
               b["Z_tr"].astype(np.float64), b["y_tr"], Za_va, b, act(a, Za_va), None)


def plan_stats(diag, ys, yt):
    # label accuracy of the plan (labels used for evaluation only), its chance level (independent coupling
    # with the same marginals), plan spread (effective partners per source row, exp of the row entropy;
    # n = uniform plan, 1 = a matching; label-free), and the marginal error (Sinkhorn convergence check)
    P, ls, lt = diag["P"], ys[diag["i_s"]], yt[diag["i_t"]]
    same = ls[:, None] == lt[None, :]
    n = len(P)
    rows = P / P.sum(1, keepdims=True)
    return {"label_acc": float((P * same).sum() / P.sum()), "label_chance": float(same.mean()),
            "partners": float(np.exp(-(rows * np.log(rows + 1e-300)).sum(1)).mean()),
            "marg_err": float(np.abs(P.sum(1) - 1 / n).sum() + np.abs(P.sum(0) - 1 / n).sum()),
            "nan": bool(np.isnan(P).any())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, seed in itertools.product(METHODS, SEEDS):
        for setting, pair, Zs, ys, Zt, yt, Zs_va, ctrl, native, paired in cases(method, seed):
            def rec(name, fit_fns):
                # fit_fns: list of zero-arg callables returning (R, b, diag); one per draw
                agree, secs, stats = [], [], []
                for f in fit_fns:
                    t0 = time.perf_counter()
                    R, b, diag = f()
                    secs.append(time.perf_counter() - t0)
                    agree.append(float((act(ctrl, Zs_va @ R.T + b) == native).mean()))
                    if diag:
                        stats.append(plan_stats(diag, ys, yt))
                row = {"method": method, "seed": seed, "setting": setting, "pair": pair, "aligner": name,
                       "agree": float(np.mean(agree)), "agree_draws": agree, "secs": float(np.mean(secs))}
                for k in (stats[0] if stats else {}):
                    row[k] = float(np.mean([s[k] for s in stats]))
                rows.append(row)
                print(f"{method:10s} s{seed} {pair:18s} {name:22s} agree {row['agree']:.3f}  {row['secs']:6.2f}s"
                      + (f"  label_acc {row['label_acc']:.3f} (chance {row['label_chance']:.3f})  partners {row['partners']:.0f}"
                         f"  marg_err {row['marg_err']:.1e}" if stats else ""), flush=True)

            def no_diag(fn):
                return lambda: (*fn(), None)

            def with_diag(fn, **kw):
                def f():
                    d = {}
                    R, b = fn(Zs, Zt, diag=d, **kw)
                    return R, b, d
                return f

            rec("identity", [no_diag(lambda: fit_identity(Zs, Zt))])
            if paired is not None:
                rec("saps", [no_diag(lambda: fit_procrustes_paired(*paired))])
            for pc in (100, 5):
                fns = [no_diag(lambda d=d: fit_action_pairs(Zs, Zt, ys, yt, np.random.default_rng(1000 * seed + d),
                                                           per_class=pc)) for d in range(N_DRAWS)]
                rec(f"pairs{pc}", fns)
                if GW:
                    rec(f"pairs{pc}_{N_DRAWS_GW}draws", fns[:N_DRAWS_GW])
            if not GW:
                continue
            for eps in EPS:
                rec(f"gw_eps{eps}", [with_diag(fit_gw, rng=np.random.default_rng(1000 * seed + 100 + d), eps=eps)
                                     for d in range(N_DRAWS_GW)])
            for pc, eps in itertools.product((100, 5), EPS):
                # same rng seeds as action_pairs draws, so the initial map R0 is the same draw
                rec(f"fgw{pc}_eps{eps}", [with_diag(fit_action_pairs_fgw, ys=ys, yt=yt,
                                                    rng=np.random.default_rng(1000 * seed + d), eps=eps,
                                                    per_class=pc, all_labels=pc == 100)
                                          for d in range(N_DRAWS_GW)])

    # summary: mean over seeds and pairs, per (method, pair type, aligner); repro also uses all 6 version pairs
    summary = {}
    for method, setting in itertools.product(METHODS, ["versions", "levels", "halves"]):
        rs = [r for r in rows if r["method"] == method and r["setting"] == setting]
        if not rs:
            continue
        summary[f"{method}/{setting}"] = {
            a: {k: float(np.mean([r[k] for r in rs if r["aligner"] == a]))
                for k in ("agree", "secs", "label_acc", "label_chance", "partners", "marg_err") if k in next(r for r in rs if r["aligner"] == a)}
            for a in dict.fromkeys(r["aligner"] for r in rs)}
    for k, v in summary.items():
        print(k, "  ".join(f"{a} {s['agree']:.3f}" for a, s in v.items()))
    if MODE == "repro":
        for m in METHODS:
            got = (summary[f"{m}/versions"]["saps"]["agree"], summary[f"{m}/versions"]["pairs100"]["agree"],
                   summary[f"{m}/levels"]["pairs100"]["agree"])
            print(f"repro {m:10s} saps / pairs versions / pairs levels: got " + " / ".join(f"{g:.3f}" for g in got)
                  + "  ref " + " / ".join(f"{r:.3f}" for r in REF[m]))
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    (OUT / "config.json").write_text(json.dumps({
        "MODE": MODE, "METHODS": METHODS, "SEEDS": SEEDS, "VERSION_PAIRS": VERSION_PAIRS, "LEVEL_PAIRS": LEVEL_PAIRS,
        "EPS": EPS, "N_DRAWS": N_DRAWS, "N_DRAWS_GW": N_DRAWS_GW, "n_subsample": 1000, "alpha": 0.5,
        "HALF_METHODS": HALF_METHODS, "SRC_EPS": SRC_EPS, "TGT_EPS": TGT_EPS, "per_class": [100, 5], "latents": str(LAT)}, indent=2))


if __name__ == "__main__":
    main()
