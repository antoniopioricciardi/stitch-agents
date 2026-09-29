# Cross-level stitching (1-1 <-> 1-2, both in v0). Encoder from level a (it sees a's frames, so the
# game is level a) + map + controller from level b. There are no paired frames across levels, so
# SAPS is not applicable: the only map is prototypes (action-class centroids of encoder_a on level-a
# frames vs encoder_b on level-b frames). Compared with the native agent of level a and with the
# level-b agent run unchanged on level a. Offline (held-out frames of level a) and in-game.
# Run: OMP_NUM_THREADS=1 uv run --project mario python mario/stitch_levels.py <arch>
import itertools
import json
import sys

import numpy as np
import torch

from agents import DEVICE, ROOT, SIZE, embed, game_policy, inputs, load_model, oracle_dir
from align import fit_identity, fit_prototypes
from data import EPISODES
from evaluate import rollout
from stitch_offline import acc, ctrl_predict, subsample

ARCH = sys.argv[1]
LEVELS = ["1-1", "1-2"]
VERSION = 0
METHODS = ["bc", "scil"]
SEEDS = [0, 1, 2]
N_DRAWS = 5  # offline draws for prototypes with 5 per class (in-game uses draw 0)
EVAL_EPISODES = 10
OUT = ROOT / f"20260929_stitch_levels_{ARCH}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for L in LEVELS:
        tr, va = EPISODES[L]
        X_tr, y_tr, _ = inputs(ARCH, L, VERSION, tr)
        X_va, y_va, _ = inputs(ARCH, L, VERSION, va)
        data[L] = (X_tr, y_tr.numpy(), X_va, y_va.numpy())

    rows = []
    for method in METHODS:
        models = {(L, s): load_model(ARCH, method, s, L, VERSION) for L in LEVELS for s in SEEDS}
        Z = {k: (embed(ARCH, m[0], data[k[0]][0]), embed(ARCH, m[0], data[k[0]][2])) for k, m in models.items()}
        for seed, (a, b) in itertools.product(SEEDS, itertools.permutations(LEVELS, 2)):
            cs = (seed + 1) % len(SEEDS)  # controller from another seed (no shared initialisation)
            enc_a, ctrl_a = models[a, seed]
            enc_b, ctrl_b = models[b, cs]
            (Za_tr, Za_va), (Zb_tr, _) = Z[a, seed], Z[b, cs]
            y_a_tr, y_a_va, y_b_tr = data[a][1], data[a][3], data[b][1]
            native_pred = ctrl_predict(ctrl_a, Za_va)
            base = {"method": method, "seed": seed, "played": a, "ctrl_level": b}

            def offline(pred):
                return {"bal_acc": acc(pred, y_a_va)[1], "agree_native": float((pred == native_pred).mean())}

            # native agent of level a (in-game numbers come from its training run)
            nat = json.loads((oracle_dir(ARCH, a, VERSION) / "metrics.json").read_text())[f"{method}_s{seed}"]
            rows.append({**base, "setting": "native", **offline(native_pred),
                         "max_x": float(np.mean(nat["max_x"])), "flag_rate": nat["flag_rate"]})
            # level-b agent run unchanged on level a
            Zb_on_a = embed(ARCH, enc_b, data[a][2])
            mx, fl = rollout(game_policy(ARCH, enc_b, ctrl_b), VERSION, EVAL_EPISODES, seed, DEVICE, a, SIZE[ARCH])
            rows.append({**base, "setting": "unchanged", **offline(ctrl_predict(ctrl_b, Zb_on_a)),
                         "max_x": float(mx.mean()), "flag_rate": float(fl.mean())})
            # stitches: encoder a + map + controller b
            maps = {"no_map": [fit_identity(Za_tr, Zb_tr)], "prototypes_all": [fit_prototypes(Za_tr, Zb_tr, ys=y_a_tr, yt=y_b_tr)]}
            maps["prototypes_5"] = []
            for draw in range(N_DRAWS):
                rng = np.random.default_rng(1000 * seed + draw)
                Zs, ys = subsample(Za_tr, y_a_tr, 5, rng)
                Zt, yt = subsample(Zb_tr, y_b_tr, 5, rng)
                maps["prototypes_5"].append(fit_prototypes(Zs, Zt, ys=ys, yt=yt))
            for setting, fits in maps.items():
                off = [offline(ctrl_predict(ctrl_b, Za_va @ R.T + bb)) for R, bb in fits]
                R, bb = fits[0]
                R_t, b_t = torch.from_numpy(R).float().to(DEVICE), torch.from_numpy(bb).float().to(DEVICE)
                mx, fl = rollout(game_policy(ARCH, enc_a, ctrl_b, R_t, b_t), VERSION, EVAL_EPISODES, seed, DEVICE, a, SIZE[ARCH])
                rows.append({**base, "setting": setting, **{k: float(np.mean([o[k] for o in off])) for k in off[0]},
                             "max_x": float(mx.mean()), "flag_rate": float(fl.mean())})
            for r in rows[-5:]:
                print(f"{method} s{seed} play {a} (ctrl {b}) {r['setting']:15s} bal {r['bal_acc']:.3f} "
                      f"agree {r['agree_native']:.3f} max x {r['max_x']:.0f} flags {r['flag_rate']:.0%}", flush=True)

    # summary per (method, played level, setting): mean over seeds; % of native per seed
    summary = {}
    settings = ["native", "unchanged", "no_map", "prototypes_all", "prototypes_5"]
    for method, a in itertools.product(METHODS, LEVELS):
        nat = {r["seed"]: r["max_x"] for r in rows if r["method"] == method and r["played"] == a and r["setting"] == "native"}
        for setting in settings:
            rs = [r for r in rows if r["method"] == method and r["played"] == a and r["setting"] == setting]
            summary[f"{method}/play_{a}/{setting}"] = {
                k: float(np.mean([r[k] for r in rs])) for k in ["bal_acc", "agree_native", "max_x", "flag_rate"]}
            summary[f"{method}/play_{a}/{setting}"]["pct_native"] = float(np.mean([r["max_x"] / nat[r["seed"]] for r in rs]))
            print(f"{method} play {a} {setting:15s} " + "  ".join(f"{k} {v:.3f}" for k, v in summary[f'{method}/play_{a}/{setting}'].items()))
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    (OUT / "config.json").write_text(json.dumps({"ARCH": ARCH, "LEVELS": LEVELS, "VERSION": VERSION, "METHODS": METHODS,
                                                  "SEEDS": SEEDS, "N_DRAWS": N_DRAWS, "EVAL_EPISODES": EVAL_EPISODES,
                                                  "EPISODES": EPISODES}, indent=2))


if __name__ == "__main__":
    main()
