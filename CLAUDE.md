# CLAUDE.md

This is a research codebase for **label-aligned encoder–controller stitching**: independently trained encoders and controllers are recombined through a map fitted from action labels alone, without paired frames. The other docs:
- `PROJECT.md`: the science
- `PLAN.md`: the roadmap and the evaluation protocol
- `EXPERIMENTS.md`: the lab log

## How to write code here

This is research code. Optimize for reading it and changing it quickly.

- **Simple and flat.** Prefer functions. Use a class only when it holds state, e.g. an `nn.Module`.
- **Readable.** Use clear names and short files. Comment the *why*: the math, tensor shapes (`z: (N, d)`), and non-obvious choices. Don't comment the obvious.
- **No guard rails.** No input validation, no try/except, no defensive checks, no fallbacks, no logging frameworks. Let it crash.
- **No over-expansion.** Implement only what the current step asks for. Avoid:
  - config systems
  - registries
  - abstract base classes
  - CLIs with many flags
  - "future-proof" options
  Hard-coded constants at the top of a script are fine.
- **One script per experiment** beats a generic framework. Copying 10 lines is better than adding a premature abstraction.
- **Standard components:** for policies, baselines and datasets, use the published implementation and modify it minimally. Write from scratch only what is new in this project (alignment, scores, analysis).
- Don't touch unrelated files. Don't reformat. Ask before adding a dependency.
- If something is ambiguous, ask before writing a lot of code.

## Stack

- Python and PyTorch.
- ManiSkill3 (PickCube) is the main environment. CARLA comes later.
- Policies: ManiSkill's **Diffusion Policy** baseline, vendored unmodified in `third_party/maniskill_diffusion_policy/` (v3.0.1). Changes go in our glue code, not in the vendored files, unless a minimal patch is unavoidable (then say so in the commit).
- POT for optimal transport and Gromov–Wasserstein.
- DINOv2/v3 via `torch.hub` or `timm`.
- matplotlib for plots.

## Environment (uv)

`uv` owns the venv and the dependencies. Never use `pip`, `pip install`, `python -m venv` or `conda`.

- `pyproject.toml` + `uv.lock` are the source of truth; the venv is `.venv/`, created by `uv sync`.
- Add a dependency with `uv add <pkg>` (dev-only tools: `uv add --dev <pkg>`); remove with `uv remove <pkg>`. Never edit the dependency list by hand.
- Run everything through uv: `uv run python scripts/<name>.py`.
- Still ask before adding a dependency.
- Pins to keep: `numpy<2` (mani_skill pins `mplib==0.1.1`, which segfaults under numpy 2); `gymnasium==0.29.1` (the DP baseline reads `final_info`, which gymnasium 1.x vector envs no longer return).

**Two separate uv projects** (not a workspace — the lockfiles must stay independent):
- Root (`./`): Python 3.12. Main env (ManiSkill, Diffusion Policy, alignment, analysis).
- `mario/`: Python 3.10, `gym==0.25.2`, `gym-super-mario-bros`, `nes-py`, `numpy<2`, torch. Mario needs the old gym/numpy stack, so it lives apart. Run with `uv run --project mario python mario/<script>.py`; add deps with `uv add --project mario <pkg>`. Mario code does not import `stitch/` (separate env, Python 3.10); copy the few functions it needs. Mario is frozen: its checkpoints and the latent export are documented in `mario/README.md` (Handoff).
- `nes-py` compiles from source, so `build-essential` (g++) must be installed on the machine.

## Layout

```
stitch/
  envs.py       # ManiSkill env variants + variation axes (visual, task, embodiment, goal marker)
  align.py      # aligners (see conventions)
  score.py      # stitchability score components (Step 7, not written yet)
scripts/        # one entry point per experiment step (run_dp.sh, export_dp_demos.py, eval_dp_blind.py, ...)
third_party/    # vendored published code (Diffusion Policy baseline)
mario/          # frozen Mario project (own uv env)
results/        # one folder per run (git-ignored except each run's config.json / metrics.json)
```

Add files only when a step needs them. Split a file only when it becomes hard to read.

## Conventions

- **Agent.** An agent is a Diffusion Policy trained on one (visual domain, task, robot). Its **encoder** is the observation encoder `agent.visual_encoder` (PlainConv): RGB `(B, 3, 128, 128)` → `z: (B, 256)` per frame. Its **controller** is the conditional denoiser, which reads `[z_{t-1}, z_t, state]` (observation horizon 2). The state (qpos, qvel, tcp_pose, goal_pos) is never mapped.
- **Stitch.** A stitch is `C_v(T(E_u(obs)))`, with T applied to z per frame.
  - T is always an `(R, b)` pair, with `z_mapped = z @ R.T + b`.
  - The only exception is the few-shot MLP map.
- **Aligners** (`stitch/align.py`). Every aligner has the same signature: `fit_<name>(Zs, Zt, **info) -> (R, b)`.
  - `Zs: (N, d)` are source latents and `Zt: (M, d)` are target latents.
  - Unpaired means N ≠ M and there is no row correspondence.
  - `info` carries action labels or foundation-model features when an aligner needs them.
  - Implemented: `identity`, `procrustes_paired` (SAPS, the paired ceiling), `action_pairs` (the default: ≤100 random same-label frame pairs per label, then Procrustes), `gw`, `action_pairs_fgw`. Planned: `dynamics`, `fm_anchors`, `fewshot`.
- **Seed pairing.** Always stitch an encoder of seed s with a controller of seed s+1. Same-seed models share their initialisation and stay partly aligned, which makes "no map" look falsely good.
- **Label budget.** When comparing aligners, every method gets the same labelled frames. Label-free methods (e.g. `gw`) must not use labels anywhere, including for subsampling.
- **Data.** Demos: states + actions per (task, robot) in `.npz` (re-renderable in any visual variant, which gives paired frames for the SAPS ceiling); training files in the DP baseline's `.h5` format, exported by `scripts/export_dp_demos.py`.
- **Evaluation protocol.** Final checkpoint (mean of the last 3 in brackets), never the best checkpoint; 250 evaluation episodes; success at any step and at the end.
- **Metrics.**
  - Success rate.
  - % of oracle recovered = stitched / oracle on the target combo.
  - % of paired ceiling = stitched / SAPS-paired.
  - Offline agreement: distance between the stitched and native agents' predicted action chunks on held-out frames. Report "no map" next to its chance level.
- **Seeds.** Always pass and record the seed. Report mean ± std over at least 3 seeds for paper numbers; single-seed runs are pilots.
- **Runs.** Each run writes to `results/<YYYYMMDD>_<step>_<name>/`: `config.json` holds the exact settings, `metrics.json` holds the numbers, and plots go alongside.

## Workflow

- Before running an experiment, write its hypothesis in `EXPERIMENTS.md`. Afterwards, add the result and a one-line takeaway. Don't draw conclusions from a pilot (one seed, one draw); wait for the full run.
- **Oracles first.** Oracle agents must work in every domain before any stitching number means anything.
- **The frozen-DINO + controller baseline** belongs in every main comparison from Step 3 on. It's the main "why stitch at all?" check.
- **One GPU, one training job at a time**, launched as a capped systemd service (see the Step 2 handoff in `EXPERIMENTS.md`).

## Working in parallel (several agents)

- Each agent works in its **own git worktree, on its own branch**. The main project folder stays on `main`.
- Shared inputs (demos, checkpoints) are read by absolute path and never modified. Each agent writes its outputs to its own worktree's `results/`.
- In `EXPERIMENTS.md`, add only your own dated entries at the top. Don't edit other entries.
- Merge into `main` when a step closes. On conflicts: keep both sides of `pyproject.toml`, then regenerate the lockfile with `uv lock` (never hand-merge `uv.lock`); keep both sides of the docs.
