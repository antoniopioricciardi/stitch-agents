# Step 0b: stitched agents playing in the game. Encoder from version u (it sees u's frames, so the
# game runs in u) + map + controller from version v. Only pairs involving v1, the one shift where
# unchanged agents fail (see EXPERIMENTS.md). Maps are fitted exactly as in stitch_offline.py.
# Run: uv run --project mario python mario/stitch_play.py
import itertools
import json

import numpy as np
import torch

from align import fit_procrustes_paired, fit_prototypes
from data import TRAIN_EPS, load_episodes
from evaluate import rollout
from latent_checks import DEVICE, ROOT, embed, load_model
from models import to_input
from stitch_offline import PROTO_SRC_EPS, PROTO_TGT_EPS, subsample

ORACLES = "20260929_step0_oracles_v{v}_e50"  # which oracle set to stitch (results folder, {v} = version)
PAIRS = [(0, 1), (1, 0), (1, 2), (2, 1)]  # (encoder / played version u, controller version v)
METHODS = ["bc", "scil"]
SEEDS = [0, 1, 2]
ALIGNERS = ["saps", "prototypes_all", "prototypes_5"]
EVAL_EPISODES = 10
OUT = ROOT / "20260929_step0b_stitch_play"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    versions = sorted({v for p in PAIRS for v in p})
    x_tr = {v: load_episodes(TRAIN_EPS, v)[0] for v in versions}  # same frames, rendered per version
    _, y_tr, e_tr, _ = load_episodes(TRAIN_EPS, 0)
    y_tr, e_tr = y_tr.numpy(), e_tr.numpy()
    src, tgt = np.isin(e_tr, PROTO_SRC_EPS), np.isin(e_tr, PROTO_TGT_EPS)

    rows = []
    for method in METHODS:
        models = {(v, s): load_model(method, s, v, ORACLES) for v in versions for s in SEEDS}
        Z = {k: embed(m[0], x_tr[k[0]]) for k, m in models.items()}  # train latents on own version
        for seed, (u, v) in itertools.product(SEEDS, PAIRS):
            cs = (seed + 1) % len(SEEDS)  # controller from another seed: no shared initialisation
            enc, ctrl = models[u, seed][0], models[v, cs][1]
            Zu, Zv = Z[u, seed], Z[v, cs]
            for aligner in ALIGNERS:
                if aligner == "saps":
                    R, b = fit_procrustes_paired(Zu, Zv)
                else:
                    n = None if aligner == "prototypes_all" else 5
                    rng = np.random.default_rng(1000 * seed)
                    Zs, ys = subsample(Zu[src], y_tr[src], n, rng)
                    Zt, yt = subsample(Zv[tgt], y_tr[tgt], n, rng)
                    R, b = fit_prototypes(Zs, Zt, ys=ys, yt=yt)
                R_t = torch.from_numpy(R).float().to(DEVICE)
                b_t = torch.from_numpy(b).float().to(DEVICE)
                # z_mapped = z @ R.T + b; default args pin this iteration's modules and map
                policy = lambda x, enc=enc, ctrl=ctrl, R_t=R_t, b_t=b_t: ctrl(enc(to_input(x)) @ R_t.T + b_t)
                max_x, flags = rollout(policy, u, EVAL_EPISODES, seed, DEVICE)
                rows.append({"method": method, "seed": seed, "u": u, "v": v, "aligner": aligner,
                             "max_x": float(max_x.mean()), "flag_rate": float(flags.mean())})
                print(f"{method} s{seed} enc v{u} -> ctrl v{v} {aligner}: max x {max_x.mean():.0f}, "
                      f"flags {flags.mean():.0%}", flush=True)

    # summary per (method, aligner): mean over pairs x seeds; % of native = max x / native max x of
    # an agent trained in the played version u (from the oracle run)
    native = {v: json.loads((ROOT / ORACLES.format(v=v) / "metrics.json").read_text()) for v in versions}
    summary = {}
    for method, aligner in itertools.product(METHODS, ALIGNERS):
        rs = [r for r in rows if r["method"] == method and r["aligner"] == aligner]
        pct = [r["max_x"] / np.mean(native[r["u"]][f"{method}_s{r['seed']}"]["max_x"]) for r in rs]
        summary[f"{method}/{aligner}"] = {"max_x": float(np.mean([r["max_x"] for r in rs])),
                                          "pct_native": float(np.mean(pct)),
                                          "flag_rate": float(np.mean([r["flag_rate"] for r in rs]))}
        print(f"{method}/{aligner}: " + ", ".join(f"{k} {v:.3f}" for k, v in summary[f'{method}/{aligner}'].items()))
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    (OUT / "config.json").write_text(json.dumps({"ORACLES": ORACLES, "PAIRS": PAIRS, "METHODS": METHODS,
                                                  "SEEDS": SEEDS, "ALIGNERS": ALIGNERS,
                                                  "EVAL_EPISODES": EVAL_EPISODES}, indent=2))


if __name__ == "__main__":
    main()
