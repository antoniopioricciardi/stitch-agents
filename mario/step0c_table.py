# Step 0c summary table: collapse, native play and label-only alignability for BC, SCIL, TACO and SCIL+TACO
# (Nature CNN). Collects results already on disk; alignment = random same-action pairs (<=100 per action,
# full training set), the adopted recipe.
# Run: uv run --project mario python mario/step0c_table.py
import json
from statistics import mean

from agents import ROOT, oracle_dir

METHODS = ["bc", "scil", "taco1", "taco3", "scil_taco1", "scil_taco3"]
DOMAINS = [("1-1", 0), ("1-1", 1), ("1-1", 2), ("1-2", 0)]
OUT = ROOT / "20260929_step0c_table"
load = lambda name: json.loads((ROOT / name).read_text())


def native(method, seed, level, version):
    return mean(json.loads((oracle_dir("nature", level, version) / "metrics.json").read_text())[f"{method}_s{seed}"]["max_x"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    collapse = load("20260929_collapse_checks_nature_" + "_".join(METHODS) + "/metrics.json")
    offline = {**load("20260929_anchors_offline/metrics.json"),
               **load("20260929_anchors_offline_nature_taco1_taco3_scil_taco1_scil_taco3/metrics.json")}
    play = {s: [r for f in ["", "_taco1_taco3", "_scil_taco1_scil_taco3", "_taco1_taco3_scil_taco1_scil_taco3"]
                if (ROOT / f"20260929_anchors_play_{s}{f}/rows.json").exists()
                for r in load(f"20260929_anchors_play_{s}{f}/rows.json") if r["aligner"] == "pairs_full"]
            for s in ["versions", "levels"]}
    table = {}
    for m in METHODS:
        nat = [json.loads((oracle_dir("nature", L, v) / "metrics.json").read_text())[m] for L, v in DOMAINS]
        ingame = {}
        for s in ["versions", "levels"]:
            rs = [r for r in play[s] if r["method"] == m]
            ingame[s] = (mean(r["max_x"] for r in rs),
                         mean(r["max_x"] / native(m, r["seed"], r["enc"][:3], int(r["enc"][-1])) for r in rs))
        table[m] = {
            "nc1": collapse[f"nature/{m}"]["nc1"], "eff_rank": collapse[f"nature/{m}"]["eff_rank"],
            "var_span": collapse[f"nature/{m}"]["var_span"],
            "native_x_1-1": mean(n["max_x"][0] for n in nat[:3]), "native_x_1-2": nat[3]["max_x"][0],
            "native_flags": mean(n["flag_rate"][0] for n in nat),
            "agree_versions": offline[f"nature/versions/{m}"]["pairs_full"],
            "agree_levels": offline[f"nature/levels/{m}"]["pairs_full"],
            "ingame_versions_x": ingame["versions"][0], "ingame_versions_pct": ingame["versions"][1],
            "ingame_levels_x": ingame["levels"][0], "ingame_levels_pct": ingame["levels"][1],
        }
        t = table[m]
        print(f"{m:11s} NC1 {t['nc1']:.2f}  rank {t['eff_rank']:5.1f}  span {t['var_span']:.0%}  "
              f"native 1-1 {t['native_x_1-1']:.0f} 1-2 {t['native_x_1-2']:.0f} flags {t['native_flags']:.0%}  "
              f"agree {t['agree_versions']:.3f}/{t['agree_levels']:.3f}  "
              f"in-game v1 {t['ingame_versions_x']:.0f} ({t['ingame_versions_pct']:.0%})  "
              f"levels {t['ingame_levels_x']:.0f} ({t['ingame_levels_pct']:.0%})")
    (OUT / "metrics.json").write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
