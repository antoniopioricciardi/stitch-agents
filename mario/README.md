# Mario (Step 0 / 0c) — frozen

Results and conclusions: `EXPERIMENTS.md` (repo root). Figures: `notebooks/latent_tour.ipynb`.

## Handoff

- **Checkpoints** (git-ignored, under `results/`): `{method}_s{seed}.pt` (keys `encoder`, `controller`). Nature CNN on 1-1 in `20260929_step0_oracles_v{0,1,2}_e50/`; everything else (Nature 1-2, ResNet18, DINOv2) in `20260929_oracles_{arch}_{level}_v{version}/`. `agents.oracle_dir(arch, level, version)` returns the right folder. Methods: `bc, scil, scilproj, taco1, taco3, scil_taco1, scil_taco3` (Nature; frozen backbones only `bc, scil`).
- **Latents are not saved**; they are recomputed from checkpoints (fast). Only frozen-backbone features are cached: `data/feats/{resnet18,dinov2}/{level}_v{version}_ep{NNN}.pt`. Demos: `data/demos/` (v0), `data/demos_v1/`, `data/demos_v2/` (git-ignored).
- **Load and embed** (`agents.py`):
  `enc, ctrl = load_model(arch, method, seed, level, version)` · `X, y, ep = inputs(arch, level, version, eps)` · `Z = embed(arch, enc, X)` → `(N, 512)` numpy. Episode lists: `data.EPISODES[level] = (train_eps, heldout_eps)`. Playing: `evaluate.rollout(agents.game_policy(arch, enc, ctrl, R, b), ...)`.
- **Aligners** (`align.py`): `fit_action_pairs` (adopted recipe: random same-action pairs, ≤100 per action), `fit_prototypes`, `fit_procrustes_paired` (SAPS); map is `z @ R.T + b`.
- **Offline agreement**: `anchors_offline.py` (adopted recipe; version pairs and cross-level; stitched vs native action on held-out frames). Older: `stitch_offline.py` (Step 0a), `stitch_levels.py` (cross-level, also in-game).
- **Seed pairing**: every stitching script uses encoder seed `s` with controller seed `cs = (seed + 1) % len(SEEDS)`. Same-seed models share their initialisation and stay partly aligned, which inflates "no map". The notebook uses `SEED = {"1-1": 0, "1-2": 1}`.
