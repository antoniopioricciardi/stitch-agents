# Within-action information probe (Step 0c follow-up F2): does the latent keep information *inside* an
# action cluster? Within each large cluster (current action NOOP, R, R+A, R+B), a linear probe from the latent
# (fit on training frames of that cluster, scored on held-out frames of that cluster) predicts:
#   (a) velocity   Mario's x displacement over the next decision (SKIP frames), R^2
#   (b) future     the action 3 decisions later, accuracy (next to the within-cluster majority baseline)
#   (c) jump       decisions until the next jump onset (A pressed, not pressed one decision earlier), R^2
# Nature CNN, 1-1 v0 and 1-2 v0, 3 seeds.
# Run: uv run --project mario python mario/probe_within_action.py
import itertools
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from agents import ROOT, embed, load_model
from data import CLASS_NAMES, EPISODES, SKIP, load_episodes

METHODS = ["bc", "scil", "scil_taco3"]
DOMAINS = [("1-1", 0), ("1-2", 0)]
SEEDS = [0, 1, 2]
CLUSTERS = [0, 1, 2, 3]  # NOOP, R, R+A, R+B
JUMP = np.array([c in (2, 4, 5, 7) for c in range(len(CLASS_NAMES))])  # classes that press A
FUTURE, HORIZON = 3, 40  # decisions ahead for (b); max decisions searched for (c)
MAX_VEL = 20  # px per decision; larger jumps are x_pos resets (pipes), not motion
OUT = ROOT / "20260929_step0c_probe"


def targets(y, e, xpos):
    # per frame t (arrays of length N; NaN / -1 where the target leaves the episode)
    N = len(y)
    same = lambda k: np.r_[e[k:] == e[:N - k], np.zeros(k, bool)]  # frame t+k in the same episode
    vel = np.full(N, np.nan)
    vel[same(SKIP)] = (xpos[SKIP:] - xpos[:-SKIP])[same(SKIP)[:N - SKIP]]
    vel[np.abs(vel) > MAX_VEL] = np.nan
    fut = np.full(N, -1)
    k = FUTURE * SKIP
    fut[same(k)] = y[k:][same(k)[:N - k]]
    jmp = np.full(N, np.nan)
    J = JUMP[y]
    for d in range(HORIZON, 0, -1):  # smallest d wins: go from far to near
        a, b = d * SKIP, (d - 1) * SKIP
        ok = same(a)
        onset = np.zeros(N, bool)
        onset[:N - a] = J[a:] & ~J[b:N - a + b]
        jmp[ok & onset] = d
    return vel, fut, jmp


def probe(Zt, tt, Zv, tv, kind):
    # linear probe on centred (not rescaled: near-dead ReLU units would blow up) latents, regularisation
    # chosen by cross-validation on the training frames; rows with a missing target are dropped
    if kind == "cls":  # keep future actions with >= 10 training frames (CV folds need every class)
        keep = [c for c in np.unique(tt[tt >= 0]) if (tt == c).sum() >= 10]
        mt, mv = np.isin(tt, keep), np.isin(tv, keep)
    else:
        mt, mv = ~np.isnan(tt), ~np.isnan(tv)
    sc = StandardScaler(with_std=False).fit(Zt[mt])
    if kind == "cls":
        clf = LogisticRegressionCV(Cs=np.logspace(-4, 0, 4), cv=3, max_iter=1000).fit(sc.transform(Zt[mt]), tt[mt])
        base = (tv[mv] == np.bincount(tt[mt]).argmax()).mean()
        return float((clf.predict(sc.transform(Zv[mv])) == tv[mv]).mean()), float(base)
    reg = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Zt[mt]), tt[mt])
    return float(r2_score(tv[mv], reg.predict(sc.transform(Zv[mv])))), None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for L, v in DOMAINS:
        tr_eps, va_eps = EPISODES[L]
        xt, *meta_t = load_episodes(tr_eps, v, L)  # frames (tensor) + actions, episode ids, x positions
        xv, *meta_v = load_episodes(va_eps, v, L)
        yt, et, pt = (a.numpy() for a in meta_t)
        yv, ev, pv = (a.numpy() for a in meta_v)
        Tt, Tv = targets(yt, et, pt), targets(yv, ev, pv)
        for method, seed in itertools.product(METHODS, SEEDS):
            enc, _ = load_model("nature", method, seed, L, v)
            Zt, Zv = embed("nature", enc, xt), embed("nature", enc, xv)
            for c in CLUSTERS:
                it, iv = np.where(yt == c)[0][::2], np.where(yv == c)[0]  # every 2nd training frame
                r = {"domain": f"{L} v{v}", "method": method, "seed": seed, "cluster": CLASS_NAMES[c]}
                r["vel_r2"], _ = probe(Zt[it], Tt[0][it], Zv[iv], Tv[0][iv], "reg")
                r["future_acc"], r["future_base"] = probe(Zt[it], Tt[1][it], Zv[iv], Tv[1][iv], "cls")
                r["jump_r2"], _ = probe(Zt[it], Tt[2][it], Zv[iv], Tv[2][iv], "reg")
                rows.append(r)
            print(f"done {L} {method} s{seed}", flush=True)

    summary = {}
    for method in METHODS:
        for dom in ["1-1 v0", "1-2 v0", "all"]:
            rs = [r for r in rows if r["method"] == method and (dom == "all" or r["domain"] == dom)]
            summary[f"{method}/{dom}"] = {k: float(np.mean([r[k] for r in rs]))
                                          for k in ["vel_r2", "future_acc", "future_base", "jump_r2"]}
    for k, s in summary.items():
        print(f"{k:22s} velocity R2 {s['vel_r2']:.3f} | action t+3 acc {s['future_acc']:.3f} "
              f"(majority {s['future_base']:.3f}) | jump R2 {s['jump_r2']:.3f}")
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    (OUT / "config.json").write_text(json.dumps({"METHODS": METHODS, "DOMAINS": DOMAINS, "SEEDS": SEEDS,
                                                  "CLUSTERS": [CLASS_NAMES[c] for c in CLUSTERS], "FUTURE": FUTURE,
                                                  "HORIZON": HORIZON, "probe": "RidgeCV / LogisticRegressionCV on centred latents", "MAX_VEL": MAX_VEL}, indent=2))


if __name__ == "__main__":
    main()
