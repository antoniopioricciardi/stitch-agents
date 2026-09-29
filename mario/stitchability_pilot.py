# Pilot of the stitchability score (PLAN Step 7) on existing Mario results: does offline agreement (stitched
# agent vs the native agent of the played domain, on held-out frames) predict in-game performance
# (% of native max x)? Collects every agent that has both numbers:
#   versions  Nature CNN stitches of Step 0b + anchor-recipe runs (offline agreement from 0a / anchors_offline)
#   levels    cross-level stitches, all encoders (unchanged / no map / prototypes)
#   shift     agents run unchanged in another ROM version, all encoders
# Run: uv run --project mario python mario/stitchability_pilot.py
import json
from statistics import mean

import numpy as np
from scipy.stats import pearsonr, spearmanr

from agents import ROOT, oracle_dir

load = lambda name: json.loads((ROOT / name).read_text())
OUT = ROOT / "20260929_stitchability_pilot"


def native_x(arch, method, seed, level, version):
    return mean(json.loads((oracle_dir(arch, level, version) / "metrics.json").read_text())[f"{method}_s{seed}"]["max_x"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pts = []  # (group, arch, label, offline agreement, in-game % of native)

    # versions: in-game from Step 0b and the anchor-recipe runs; offline from 0a and anchors_offline
    off = {}
    for r in load("20260929_step0a_stitch_offline_e50/rows.json"):
        name = {"saps": "saps", "prototypes": {None: "prototypes_all", 5: "prototypes_5"}.get(r["n"])}.get(r["aligner"])
        if name:
            off.setdefault((r["method"], r["seed"], r["u"], r["v"], name), []).append(r["agree"])
    for r in load("20260929_anchors_offline/rows.json"):
        if r["arch"] == "nature" and r["setting"] == "versions" and r["aligner"].startswith("pairs"):
            u, v = int(r["pair"][1]), int(r["pair"][-1])
            off[r["method"], r["seed"], u, v, r["aligner"]] = [r["agree"]]
    games = [(r["method"], r["seed"], r["u"], r["v"], r["aligner"], r["max_x"]) for r in load("20260929_step0b_stitch_play/rows.json")]
    games += [(r["method"], r["seed"], int(r["enc"][-1]), int(r["ctrl"][-1]), r["aligner"], r["max_x"])
              for r in load("20260929_anchors_play_versions/rows.json")]
    for m, s, u, v, a, x in games:
        pts.append(("versions", "nature", f"{m} v{u}->v{v} {a}", mean(off[m, s, u, v, a]), x / native_x("nature", m, s, "1-1", u)))

    # levels: rows carry both numbers
    for arch in ["nature", "resnet18", "dinov2"]:
        for r in load(f"20260929_stitch_levels_{arch}/rows.json"):
            if r["setting"] != "native":
                pts.append(("levels", arch, f"{r['method']} play {r['played']} {r['setting']}", r["agree_native"],
                            r["max_x"] / native_x(arch, r["method"], r["seed"], r["played"], 0)))

    # shift: unchanged agents in another version
    for arch, name in [("nature", "20260929_step0_shift_nostitch_e50"), ("resnet18", "20260929_shift_nostitch_resnet18"),
                       ("dinov2", "20260929_shift_nostitch_dinov2")]:
        for r in load(f"{name}/rows.json"):
            if r["train"] != r["test"]:
                pts.append(("shift", arch, f"{r['method']} v{r['train']} in v{r['test']}", r["agree_native"],
                            r["max_x"] / native_x(arch, r["method"], r["seed"], "1-1", r["test"])))

    agree, pct = np.array([p[3] for p in pts]), np.array([p[4] for p in pts])
    summary = {"all": {"n": len(pts), "spearman": float(spearmanr(agree, pct)[0]), "pearson": float(pearsonr(agree, pct)[0])}}
    for key, sel in [(g, [p[0] == g for p in pts]) for g in ["versions", "levels", "shift"]] + \
                    [(a, [p[1] == a for p in pts]) for a in ["nature", "resnet18", "dinov2"]]:
        sel = np.array(sel)
        summary[key] = {"n": int(sel.sum()), "spearman": float(spearmanr(agree[sel], pct[sel])[0]),
                        "pearson": float(pearsonr(agree[sel], pct[sel])[0])}
    for k, v in summary.items():
        print(f"{k:9s} n={v['n']:3d}  spearman {v['spearman']:.2f}  pearson {v['pearson']:.2f}")
    # biggest disagreements: high agreement but poor play, and the reverse
    order = np.argsort(pct - np.poly1d(np.polyfit(agree, pct, 1))(agree))
    print("\nplays much worse than its agreement predicts:")
    for i in order[:5]:
        print(f"  {pts[i][1]:9s} {pts[i][2]:32s} agree {agree[i]:.2f}  in-game {pct[i]:.0%}")
    print("plays much better than its agreement predicts:")
    for i in order[-5:]:
        print(f"  {pts[i][1]:9s} {pts[i][2]:32s} agree {agree[i]:.2f}  in-game {pct[i]:.0%}")
    (OUT / "points.json").write_text(json.dumps(pts, indent=1))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for g, mk in [("versions", "o"), ("levels", "s"), ("shift", "^")]:
        for a, col in [("nature", "C0"), ("resnet18", "C1"), ("dinov2", "C2")]:
            sel = [i for i, p in enumerate(pts) if p[0] == g and p[1] == a]
            if sel:
                ax.scatter(agree[sel], pct[sel], s=18, marker=mk, c=col, alpha=0.6, label=f"{g}, {a}")
    ax.set_xlabel("offline agreement with the native agent")
    ax.set_ylabel("in-game max x, % of native")
    ax.set_title(f"Stitchability pilot: Spearman {summary['all']['spearman']:.2f} (n={len(pts)})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "agreement_vs_ingame.png", dpi=120)


if __name__ == "__main__":
    main()
