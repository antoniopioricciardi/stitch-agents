"""Step B tables: PeS vs nn_affine vs the paired affine ceiling (matrix + goal shift), and GW on the matrix.

Usage: uv run python scripts/stepB_table.py {pes|pesc|gw}   (pesc: the centred PeS variant, runs stepB_pesc_*, tag _centred)
Sources (success_once, 250 episodes): ceiling and identity = Step 1b matrix (labelstitch-1b/results/20261003_step1b_matrix);
nn_affine on own-data motion pairs = results/20261013_step5b_motion_matrix_<shift>_<pair> and, for the goal shift,
results/20261013_step5b_motion_stepR_goal_<pair> (nn_affine_motion_own; its "affine" is the goal ceiling).
PeS: results/*_stepB_pes_{matrix,goal}_*_k<K>; PeS native = mean success_once of the last 3 evals (40k / 45k / 50k)
in each agent's training log; an agent below the oracle bar (.3) is flagged with *.
Prints markdown tables and the decision-rule numbers (d = nn_affine - method in % of the ceiling, 2 SE = 2 std / sqrt(n)).
"""
import glob
import json
import re
import sys

import numpy as np

SHIFTS, PAIRS, DIRS = ("look1", "look2", "look2light1", "cam3"), ("p1", "p2", "p3"), ("fwd", "bwd")
SEED_OF = {"p1": (1, 2), "p2": (2, 3), "p3": (3, 1)}  # (cam0 seed, shift seed); goal: (look 2 seed, goal cam0 seed)
CEIL = json.load(open("/home/ricc/projects/labelstitch-1b/results/20261003_step1b_matrix/metrics.json"))["success_once"]


def last(pattern):
    f = sorted(glob.glob(pattern))
    return json.load(open(f[-1])) if f else None


def native(dom, seed, K=256):
    f = sorted(glob.glob(f"results/*_stepB_{PRE}_k{K}_{dom}_s{seed}/log.txt"))
    if not f:
        return None
    s = [float(x) for x in re.findall(r"^success_once: ([0-9.]+)", open(f[-1]).read().replace("\r", "\n"), re.M)]
    return float(np.mean(s[-3:])) if len(s) >= 11 else None  # 11 evals = training finished


def rule(d):
    d = np.array(d)
    return f"mean d = {d.mean():+.1f} pts, 2 SE = {2 * d.std() / np.sqrt(len(d)):.1f} (n = {len(d)})"


def fmt(x, ref=None):
    return "—" if x is None else (f"{x:.3f}" if ref is None else f"{x:.3f} ({100 * x / ref:.0f}%)")


if __name__ == "__main__":
    MODE = sys.argv[1]
    K = 256
    PRE, TAG = ("pesc", "_centred") if MODE == "pesc" else ("pes", "")
    if MODE in ("pes", "pesc"):
        print("| shift | pair | dir | identity | PeS stitch | nn_affine | ceiling | PeS native (enc / ctrl) |\n|---|---|---|---|---|---|---|---|")
        d_all = []
        for sh in SHIFTS:
            for p in PAIRS:
                pes, nn = last(f"results/*_stepB_pes_matrix_{sh}_{p}{TAG}_k{K}/metrics.json"), last(f"results/20261013_step5b_motion_matrix_{sh}_{p}/metrics.json")
                s0, s1 = SEED_OF[p]
                for dr in DIRS:
                    c = CEIL[sh][p][dr]
                    e, ct = ((native("cam0", s0), native(sh, s1)) if dr == "fwd" else (native(sh, s1), native("cam0", s0)))
                    x = pes["stitch"][dr]["success_once"] if pes and dr in pes["stitch"] else None
                    y = nn["stitch"][dr]["nn_affine_motion"]["success_once"]
                    flag = "*" if any(v is not None and v < .3 for v in (e, ct)) else ""
                    print(f"| {sh} | {p} | {dr} | {fmt(c['identity'], c['affine'])} | {fmt(x, c['affine'])}{flag} | {fmt(y, c['affine'])} | {c['affine']:.3f} | {fmt(e)} / {fmt(ct)} |")
                    if x is not None and not flag:
                        d_all.append(100 * (y - x) / c["affine"])
        print("\nmatrix, nn_affine - PeS:", rule(d_all) if d_all else "no cells yet")
        print("\n| goal pair | PeS stitch | nn_affine (own, motion) | ceiling | PeS native (look 2 enc / goal ctrl) |\n|---|---|---|---|---|")
        d_goal = []
        for p in PAIRS:
            pes, ref = last(f"results/*_stepB_pes_goal_{p}{TAG}_k{K}/metrics.json"), last(f"results/20261013_step5b_motion_stepR_goal_{p}/metrics.json")
            c, y = ref["aligners"]["affine"]["success_once"], ref["aligners"]["nn_affine_motion_own"]["success_once"]
            e, ct = native("look2", SEED_OF[p][0]), native("goal_cam0", SEED_OF[p][1])
            x = pes["stitch"]["goal"]["success_once"] if pes else None
            flag = "*" if any(v is not None and v < .3 for v in (e, ct)) else ""
            print(f"| {p} | {fmt(x, c)}{flag} | {fmt(y, c)} | {c:.3f} | {fmt(e)} / {fmt(ct)} |")
            if x is not None and not flag:
                d_goal.append(100 * (y - x) / c)
        print("\ngoal, nn_affine - PeS:", rule(d_goal) if d_goal else "no cells yet")
        print("\nPeS native, last 3 (K = 256):", {f"{d}_s{s}": native(d, s) for d in ("cam0", "look1", "look2", "look2light1", "cam3", "goal_cam0") for s in (1, 2, 3)})
        print("PeS native, last 3 (K = 64):", {f"{d}_s{s}": native(d, s, 64) for d in ("cam0", "look2") for s in (1, 2, 3)})
        k64 = [(last(f"results/*_stepB_pes_matrix_look2_{p}{TAG}_k64/metrics.json"), last(f"results/*_stepB_pes_matrix_look2_{p}{TAG}_k256/metrics.json")) for p in PAIRS]
        print("look 2, K = 64 vs 256 (success_once per pair and dir):",
              [(p, dr, a["stitch"][dr]["success_once"], b["stitch"][dr]["success_once"]) for p, (a, b) in zip(PAIRS, k64) if a and b for dr in DIRS])
    else:
        print("| shift | pair | dir | identity | gw_orth | gw_affine | nn_affine | ceiling |\n|---|---|---|---|---|---|---|---|")
        d = {"gw_orth": [], "gw_affine": []}
        for sh in SHIFTS:
            for p in PAIRS:
                gw, nn = last(f"results/*_stepB_gw_matrix_{sh}_{p}/metrics.json"), last(f"results/20261013_step5b_motion_matrix_{sh}_{p}/metrics.json")
                for dr in DIRS:
                    c = CEIL[sh][p][dr]
                    y = nn["stitch"][dr]["nn_affine_motion"]["success_once"]
                    g = {k: gw["stitch"][dr][k]["success_once"] if gw and dr in gw["stitch"] else None for k in d}
                    print(f"| {sh} | {p} | {dr} | {fmt(c['identity'], c['affine'])} | {fmt(g['gw_orth'], c['affine'])} | {fmt(g['gw_affine'], c['affine'])} | {fmt(y, c['affine'])} | {c['affine']:.3f} |")
                    for k in d:
                        if g[k] is not None:
                            d[k].append(100 * (y - g[k]) / c["affine"])
        for k, v in d.items():
            print(f"nn_affine - {k}:", rule(v) if v else "no cells yet")
