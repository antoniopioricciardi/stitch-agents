# CLAUDE.md

This is a research codebase for **anchor-free encoder–controller stitching**. The other docs:
- `PROJECT.md`: the science
- `PLAN.md`: the roadmap
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
- Don't touch unrelated files. Don't reformat. Ask before adding a dependency.
- If something is ambiguous, ask before writing a lot of code.

## Stack

- Python and PyTorch.
- ManiSkill3 is the main environment. CARLA comes later.
- POT for optimal transport and Gromov–Wasserstein.
- DINOv2/v3 via `torch.hub` or `timm`.
- matplotlib for plots.

## Environment (uv)

`uv` owns the venv and the dependencies. Never use `pip`, `pip install`, `python -m venv` or `conda`.

- `pyproject.toml` + `uv.lock` are the source of truth; the venv is `.venv/`, created by `uv sync`.
- Add a dependency with `uv add <pkg>` (dev-only tools: `uv add --dev <pkg>`); remove with `uv remove <pkg>`. Never edit the dependency list by hand.
- Run everything through uv: `uv run python scripts/<name>.py`.
- Still ask before adding a dependency.

**Two separate uv projects** (not a workspace — the lockfiles must stay independent):
- Root (`./`): Python 3.12, `numpy<2` (mani_skill pins `mplib==0.1.1`, which segfaults under numpy 2). Main env (ManiSkill, alignment, analysis).
- `mario/`: Python 3.10, `gym==0.25.2`, `gym-super-mario-bros`, `nes-py`, `numpy<2`, torch. Mario needs the old gym/numpy stack, so it lives apart. Run with `uv run --project mario python mario/<script>.py`; add deps with `uv add --project mario <pkg>`. Mario code does not import `stitch/` (separate env, Python 3.10); copy the few functions it needs.
- `nes-py` compiles from source, so `build-essential` (g++) must be installed on the machine.

## Layout

```
stitch/
  envs.py       # env builders + variation axes (visual, task, embodiment)
  data.py       # demo collection / loading
  models.py     # Encoder, Controller (plain nn.Modules)
  align.py      # alignment functions (see conventions)
  score.py      # stitchability score components
  evaluate.py   # closed-loop rollout of C(T(E(obs)))
scripts/        # one entry point per PLAN step
results/        # one folder per run
```

Start with these files. Split a file only when it becomes hard to read.

## Conventions

- **Agent.** An agent is an encoder `E: pixels -> z ∈ R^d` plus a controller `C: z [+ proprio] -> action`. We train one agent per (visual domain u, task v).
- **Stitch.** A stitch is `C_v(T(E_u(obs)))`.
  - T is always an `(R, b)` pair, with `z_mapped = z @ R.T + b`.
  - The only exception is the few-shot MLP map.
- **Aligners.** Every aligner has the same signature: `fit_<name>(Zs, Zt, **info) -> (R, b)`.
  - `Zs: (N, d)` are source latents and `Zt: (M, d)` are target latents.
  - Unpaired means N ≠ M and there is no row correspondence.
  - `info` carries actions, labels or foundation-model features when an aligner needs them.
  - Planned aligners: `identity`, `procrustes_paired` (SAPS), `prototypes`, `fm_anchors`, `fgw`, `fewshot`.
- **Datasets.** Store as `.npz` / `.h5` with keys `obs, action, proprio, domain, task, episode`.
- **Seeds.** Always pass and record the seed. Report mean ± std over at least 3 seeds.
- **Metrics.**
  - Success rate.
  - % of oracle recovered = stitched / oracle on the target combo.
  - % of paired ceiling = stitched / SAPS-paired.
- **Runs.** Each run writes to `results/<YYYYMMDD>_<step>_<name>/`: `config.json` holds the exact settings, `metrics.json` holds the numbers, and plots go alongside.

## Workflow

- Before running an experiment, write its hypothesis in `EXPERIMENTS.md`. Afterwards, add the result and a one-line takeaway.
- **Oracles first.** Oracle agents must work in every domain before any stitching number means anything.
- **The frozen-DINO + controller baseline is in every comparison.** It's the main "why stitch at all?" check.
