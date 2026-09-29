# PLAN.md — step-by-step discovery plan

Each step answers **one question**. For each step you get:
- a **prompt** to paste into Claude Code (or into chat),
- an **exit criterion**,
- a **decision** to make before moving on.

Don't start a step until the previous exit criterion is met. If a result surprises you, stop and discuss it; the plan is allowed to change.

**Rough timeline, targeting ICML 2027 (late January):**
- Oct: Steps 0c–3 (including 1b, the action-labelling question)
- Nov: Steps 4–5
- Dec: Steps 6–8 + ablations
- Jan: writing

Step 9 (CARLA) can move to the NeurIPS/CoRL version.

---

## Step 0 — Mario sanity check (done)

**Questions:**
- Do action-class prototypes give a map as good as SAPS with paired anchors?
- Does it work post hoc, or only on SCIL-trained encoders?

**Findings (2026-09-28/29, details in `EXPERIMENTS.md`):**
- SCIL does not make agents robust; it makes them **alignable without paired data**. Offline, SCIL + prototypes ≈ SCIL + SAPS (paired); BC + prototypes is clearly weaker.
- In-game (0b, v1 pairs): SCIL + label-only anchors 88–91% of native, same as SCIL + SAPS; BC + label-only 46–49%.
- Across levels (1-1 ↔ 1-2, no paired frames exist): SCIL + label-only lets the other level's controller play (Nature CNN 74–77% of native on 1-1, flags included); BC 28–39%.
- Anchor recipe: random same-action frame pairs (≤100 per action) ≡ action centroids.
- Mechanism: **NC1 (each action collapses onto its centroid) holds; NC2 (regular simplex) does not.** SupCon on a projection head removes the collapse and label-only alignment falls back to about BC level. Collapse is the mechanism.
- Frozen ResNet18 / DINOv2 do not remove the visual-shift problem and play worse (weak test on NES frames).
- Stitchability pilot: offline agreement vs in-game ρ ≈ 0.69 for well-trained agents, ≈ 0 for weak ones.

## Step 0c — Mario: does a TACO-style temporal loss keep or break alignability? (optional, ~half a day)

**Question:** before continuous control, check the two TACO roles we care about on known ground.

> Read CLAUDE.md, PROJECT.md and the latest EXPERIMENTS.md entries. Write the hypotheses in EXPERIMENTS.md first:
> (H1) a TACO-only encoder (temporal InfoNCE: latent of frame t + embedded actions t..t+K−1 ↔ latent of frame t+K, K ∈ {1, 3}; BC on the controller) does not collapse and aligns from labels about as badly as BC;
> (H2) SCIL + TACO (SupCon on the latent + TACO on its own projection heads) keeps SCIL's collapse and alignability while native play is equal or better.
> Train Nature CNN agents for BC, SCIL, TACO, SCIL+TACO on 1-1 v0/v1/v2 and 1-2 v0 (3 seeds, 50 epochs, same recipe as before). Report: NC1, effective rank, variance in the centroid span, native play, and label-only alignability (offline agreement for version pairs and levels; in-game v1 pairs and cross-level), with the adopted anchor recipe. Keep the code minimal; TACO heads are small MLPs, batch as large as memory allows.

- **Exit:** one table in `EXPERIMENTS.md`.
- **Decision:**
  - If SCIL+TACO keeps alignability, it goes into the Step 1b candidate list as the leading hybrid.
  - If TACO destroys collapse even as an auxiliary term, drop the hybrid and keep TACO only as a labelling function and dynamics aligner.

## Step 1 — ManiSkill3 environment with variation axes

**Question:** can we generate controlled visual, task and embodiment variants, plus expert demos, cheaply?

> Read CLAUDE.md and PROJECT.md. In `stitch/envs.py`, write `make_env(visual, task, robot)` for ManiSkill3, starting from a single task (PickCube or PushCube). The axes are:
> - visual: camera pose ×3, texture/colour ×3, lighting ×2
> - task: default vs changed friction/mass, and one goal variant
> - robot: Panda, plus a second arm or a locked joint
>
> Then write `scripts/check_envs.py`, which saves one RGB frame per variant to `results/` so I can inspect them, and prints the rendering FPS. Keep it minimal.

> Write `scripts/collect_demos.py`, which collects N motion-planning expert demos per (visual, task, robot) combination and saves them in the dataset format defined in CLAUDE.md.

- **Exit:** frames look right, and there are ≥100 successful demos per combination.
- **Decision:** confirm ManiSkill3 as the main environment. If the visual axes are too weak, look at Colosseum V2.

## Step 1b — Action labelling for continuous control (the key design question)

**Question:** which action labelling / objective keeps encoders alignable from labels alone **and** keeps enough within-action information for fine control?

> On one ManiSkill3 task, two visual domains (e.g., two camera poses) and the default robot, train encoders + controllers with each candidate:
> 1. BC (reference, no collapse)
> 2. SupCon on per-dimension action bins (SCIL mixed-radix)
> 3. SupCon on k-means clusters of actions (K ∈ {8, 16, 32})
> 4. SupCon on k-means clusters of action chunks (e.g., 4–8 steps)
> 5. SupCon on clusters in a TACO action space (action encoder trained on actions pooled from both domains, so labels are shared)
> 6. Rank-N-Contrast on continuous actions
> 7. Fixed simplex (ETF) head on the cluster labels
> 8. Hybrid: best of 2–5 + TACO temporal term (only if Step 0c supports it)
> 9. Temporal anchors: anchor pairs matched on action + steps-to-grasp (or task phase). Test only if the probe-transfer test below shows a drop.
>
> For each: native success rate, NC1 / effective rank, label-only alignability (offline action agreement and closed-loop success of the cross-domain stitch, vs SAPS paired), and a within-action information probe (regress the continuous action from the latent within each cluster; report R²). Also:
> - **Probe-transfer test:** fit the within-action probe on the target encoder's latents and apply it to mapped source latents, T(E_u(o)). A drop relative to the target encoder itself means within-action information exists but isn't aligned by the map.
> - **Probe recipe** (from Mario F2): centre the latents only (no per-unit scaling; near-dead ReLU units blow up), and choose the regularisation by cross-validation.

- **Exit:** one table: candidate × {native success, stitched success, % of paired ceiling, within-cluster R²}.
- **Decision:** pick the labelling/objective for the rest of the project. If nothing keeps both native performance and alignability, reframe: label-only alignment for coarse control, few-shot map refinement for fine control.

## Step 2 — Agents and oracles

**Question:** do our agents solve each combination on their own?

> In `stitch/models.py`, write a small CNN (or ViT-S) Encoder with latent dim d = 64–256, and an MLP Controller that takes z and optionally proprioception. Write `scripts/train_bc.py`, which trains one agent per combination with BC plus the encoder objective chosen in Step 1b. Write `stitch/evaluate.py`, which runs closed-loop success-rate evaluation. Use 3 seeds.

- **Exit:** every oracle reaches at least ~80% success, or the best achievable for that task.
- **Decision:** fix the architecture and latent dimension for the rest of the project.

## Step 3 — The "why stitch?" baseline

**Question:** does a frozen foundation backbone already solve the visual shift?

> Train controllers on frozen DINOv2 and DINOv3 features: (a) raw features, (b) features plus a small trainable adapter. Train on one visual domain, evaluate on all the others, and compare with our trained encoders plus naive retraining.

- **Exit:** a table comparing frozen vs trained encoders across the visual axes.
- **Decision (important):**
  - If frozen+adapter matches the oracles on the visual shifts, move the emphasis to task and embodiment shift, and to reusing legacy encoders.
  - Otherwise, the motivation section writes itself.

## Step 4 — Visual shift, unpaired stitching (core result)

**Question:** how much of the paired ceiling can we recover with no paired frames?

> In `stitch/align.py`, implement `fit_identity`, `fit_procrustes_paired` (SAPS), `fit_action_pairs` (random same-action pairs, then Procrustes; the Mario recipe), `fit_gw` (entropic GW with POT, then barycentric Procrustes) and `fit_action_pairs_fgw` (action-pair initialisation plus fused-GW refinement). Write `scripts/stitch_matrix.py`, which evaluates every encoder_u × controller_u' pair for the same task and saves an N×N success matrix per method.

- **Exit:** success matrices per method, and the % of paired ceiling.
- **Decision:** pick the default aligner. Target: at least ~70–80% of the paired ceiling. Add the HGA / Latent Functional Maps baselines here.

## Step 5 — Task shift

**Question:** when action semantics differ, what replaces them: latent dynamics consistency, foundation-mined anchors, or a few demos?

> Implement three aligners:
> - `fit_dynamics`: train a latent forward model (TACO-style: z_t + action embedding → z_{t+K}) on the target domain; fit T so that mapped source transitions (T z_t, a_t) predict T z_{t+K} under the target model. T linear/orthogonal, initialised from action pairs when available.
> - `fit_fm_anchors`: DINO features for frames from both domains (optionally foreground crops); keep mutual nearest neighbours that pass a cycle-consistency filter; robust Procrustes on the corresponding latents, then FGW.
> - `fit_fewshot`: train T by BC through the frozen controller on K target demos.
>
> Evaluate separately on goal/reward variants (same dynamics) and physics variants (friction/mass), for K ∈ {0, 1, 5, 20}. Compare with retraining the controller using the same K, and with Dynamics Cycle-Consistency (Zhang et al. 2021, code `sjtuzq/Cycle_Dynamics`).

- **Exit:** a K-shot curve per task shift, with the zero-shot aligners at K = 0.
- **Decision:** decide which task shifts are zero-shot and which are few-shot, and which correspondence source works where. This goes into the claim.

## Step 6 — Embodiment

**Question:** can we stitch across robots if the controller owns the proprioception?

> Stitch encoders across the Panda and second-robot (or locked-joint) variants, keeping each robot's own controller with proprio input. Use the best aligner from Steps 4–5.

- **Exit:** a results table.
- **Decision:** does embodiment stay in the main claim, or move to the appendix?

## Step 7 — Stitchability score

**Question:** can we predict stitch success without doing rollouts?

Mario pilot: offline agreement predicts in-game play for well-trained agents (ρ ≈ 0.69) but not for weak ones (ρ ≈ 0).

> In `stitch/score.py`, compute: held-out anchor residual, cycle error, GW distortion, CKA, and action agreement on a few labelled target samples. Over all pairs from Steps 4–6, report the Spearman ρ of each component and of their combination against success, plus the ROC of a deploy / don't-deploy threshold. Exclude or flag pairs whose native agents are weak.

- **Exit:** a correlation plot.
- **Decision:** is the score a headline contribution, or an analysis section?

## Step 8 — Controller swap mid-episode (trajectory-stitching demo)

**Question:** can one encoder drive two stitched controllers in sequence?

> Take a two-stage task (e.g., reach, then grasp/push). Run controller A, then switch to controller B at the hand-off point, with both reading the same encoder through their aligned maps. Measure hand-off success, and check whether the stitchability score predicts it.

- **Exit:** one figure plus a video.
- **Decision:** include it in the paper, or keep it as a teaser for the follow-up.

## Step 9 — CARLA (secondary; can move to the NeurIPS/CoRL version)

> Using a Bench2Drive subset with a PDM-Lite expert, vary weather ×4 and camera placement ×3, plus target speed and vehicle dynamics. Repeat Steps 2, 3, 4 and 5 at a smaller scale. Add A Stitch in Time (arXiv 2606.21509) as a baseline.

## Step 10 — Ablations and writing

- **Ablations:**
  - correspondence source (action pairs / dynamics / foundation anchors / GW-only)
  - map class (orthogonal / affine / MLP)
  - encoder objective (BC / SCIL / SupCon on a projection head / TACO / SCIL+TACO / fixed simplex)
  - choice of foundation model F
  - number of anchors
  - K
  - latent dimension d
- **Writing:**
  - The intro builds on the limitations stated by Policy Stitching and Perception Stitching, and positions against Zhang et al. 2021 (unpaired, but trained translators) and A Stitch in Time (paired or task-supervised).
  - The mechanism section: action collapse (NC1), the projection-head ablation, and why NC2 is not needed.
  - The main table has three column groups: visual | task | embodiment.
  - Paper-quality Mario rows: more episodes and confidence intervals.
  - Post to arXiv early, because of the scoop risk.
