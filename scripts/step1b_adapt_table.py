"""Step 1b items 2a-2c: success vs fine-tuning iterations per arm, plus encoder geometry / map residuals.

Usage: uv run python scripts/step1b_adapt_table.py
Curves from each run's log.txt: evals at iterations 0, 1k, ..., 4k (100 episodes) and 5k (250 episodes) for the 5k
runs; every 5k (250 episodes) for the 50k scratch references. Geometry / residuals from the runs' geom stitch outputs.
Writes results/<date>_step1b_adapt/{metrics.json, table.md}.
"""
import glob
import json
import re
from datetime import date
from pathlib import Path

R = Path("results")


def curve(run):
    # -> [(episodes, success_once, success_at_end)] in eval order
    t = open(glob.glob(f"results/2026100?_{run}/log.txt")[0], errors="ignore").read()
    ev = re.findall(r"Evaluated (\d+) episodes\s*\n?.*?success_once: ([0-9.]+)\s*\n?.*?success_at_end: ([0-9.]+)", t, re.S)
    return [(int(n), float(a), float(b)) for n, a, b in ev]


def geom(name, domain):
    f = glob.glob(f"results/2026100?_step1b_stitch_{name}_{domain}/metrics.json")
    if not f:
        return None
    m = json.load(open(f[0]))
    g = m["geometry"][domain]
    a = next(iter(m["stitch"].values()))["aligners"]
    return dict(nc1=g["nc1"], eff_rank=g["eff_rank"], **{f"resid_{k}": v["z_residual"] for k, v in a.items()})


out, rows = {}, []
rows += ["**(2a / 2b) cam0 s1 policy fine-tuned on N look 2 demos** (success_once at 0, 1k, 2k, 3k, 4k (100 ep.) | 5k (250 ep.))", "",
         "| arm | N | curve | final | encoder NC1 / eff. rank | residual: identity / affine paired / pca16 (N demos) |", "|---|---|---|---|---|---|"]
for part in ("all", "encoder"):
    for n in (5, 10, 25):
        run = f"step1b_ft_{part}_cam0s1_look2_n{n}"
        c = curve(run)
        g = geom(f"geom_{part}_n{n}", "look2")
        out[run] = dict(curve=c, geometry=g)
        rows.append(f"| {'end-to-end' if part == 'all' else 'encoder only'} | {n} | " + " ".join(f"{x[1]:.2f}" for x in c[:-1])
                    + f" | **{c[-1][1]:.3f}** | {g['nc1']:.2f} / {g['eff_rank']:.1f} | "
                    + f"{g['resid_identity']:.2f} / {g['resid_affine']:.2f} / {g['resid_nn_affine_pca16']:.2f} |")
rows += ["", "**(2c) goal shift: look 2 default encoder s2 → cam0 goal controller s2, goal variant in look 2** (reference: goal look 2 oracle .685)", "",
         "| arm | N | curve 0–4k (100 ep.) | 5k (250 ep.) |", "|---|---|---|---|"]
for n in (10, 25):
    for arm, label in (("i_map", "(i) stitched, map only"), ("ii_all", "(ii) stitched, all"),
                       ("iii_identity", "(iii) identity map, all"), ("iv_scratch5k", "(iv) from scratch, 5k")):
        run = f"step1b_ftgoal_{arm}_n{n}"
        c = curve(run)
        out[run] = dict(curve=c)
        rows.append(f"| {label} | {n} | " + " ".join(f"{x[1]:.2f}" for x in c[:-1]) + f" | **{c[-1][1]:.3f}** |")
    run = f"step1b_ftgoal_iv_scratch50k_n{n}"
    c = curve(run)
    out[run] = dict(curve=c, geometry=dict(scratch5k=geom(f"geom_iv_scratch5k_n{n}", "goal_look2"), scratch50k=geom(f"geom_iv_scratch50k_n{n}", "goal_look2")))
    so = [x[1] for x in c]
    rows.append(f"| (iv) from scratch, 50k (every 5k, 250 ep.) | {n} | " + " ".join(f"{x:.2f}" for x in so) + f" | final {so[-1]:.3f} (last-3 {sum(so[-3:]) / 3:.3f}) |")

OUT = R / f"{date.today():%Y%m%d}_step1b_adapt"
OUT.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT / "metrics.json", "w"), indent=1)
json.dump(dict(source="each run's log.txt and geom stitch outputs"), open(OUT / "config.json", "w"), indent=1)
open(OUT / "table.md", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
