# PLAN.md — step-by-step discovery plan

Each step answers **one question**. For each step you get:
- a **prompt** to paste into Claude Code (or into chat),
- an **exit criterion**,
- a **decision** to make before moving on.

Don't start a step until the previous exit criterion is met. If a result surprises you, stop and discuss it; the plan is allowed to change.

**Rough timeline, targeting ICML 2027 (late January):**
- Oct: Steps 1b–3 (Steps 0, 0c and 1 are done; Step 2 oracles in progress)
- Nov: Steps 4–5
- Dec: Steps 6, 6b, 7, 8 + ablations
- Jan: writing

Step 9 (CARLA) can move to the NeurIPS/CoRL version.

**Evaluation protocol (all ManiSkill steps):** report the final checkpoint (mean of the last 3 in brackets), never the best checkpoint; 250 evaluation episodes for oracle and stitched numbers; success at any step and at the end.

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

## Step 0c — Mario: does a TACO-style temporal loss keep or break alignability? (done)

**Findings (2026-09-29, details in `EXPERIMENTS.md`):**
- TACO alone does not collapse and aligns from labels as badly as BC or worse.
- **SCIL + TACO (K = 3) keeps SCIL's label-only alignability** (matches SCIL in-game) and **recovers the native play lost to collapse** (1-2: 654 → 1207, 18% flags).
- Within-action probe on 1-1: SCIL < BC ≈ SCIL+TACO (future-action accuracy: SCIL at the majority baseline). Supportive, but the absolute signal is weak.
- **Decision:** SCIL + TACO (K = 3) is the leading hybrid for Step 1b. Open control for the paper: the TACO term saw 1024 extra transitions per step (batch confound).

## Step 1 — ManiSkill3 environment with variation axes (done, 2026-09-30)

**Findings (2026-09-29/30, details in `EXPERIMENTS.md`):** ManiSkill3 confirmed as the main environment.
- Task: **PickCube** (supports both robots and has working planner solutions for both).
- Axes: camera ×3 (cam0 → cam2 mild, cam0 → cam1 strong), look ×3, light ×2; task: default, **actuation** (EE-delta bounds ×0.5), goal (y ∈ [0.15, 0.25]); robot: Panda, xArm6 + Robotiq. The locked joint is not supported in this version.
- The friction/mass variant was dropped: the open-loop planner ignores physics, so expert actions didn't change.
- Goal marker: a **lollipop** (magenta 1.25 cm sphere on a 12 mm pole, non-colliding). The original solid sphere covered ~10% of the cube in cam0 and cost ~0.2 success.
- Demos: ~500 per (task, robot), position-only control (`pd_ee_delta_pos`, 4-d on both robots), states stored so any trajectory can be re-rendered in any visual variant (paired frames for the SAPS ceiling). 500 planner attempts per (task, robot): ≥ 99% planner success, ~100% conversion (496–498 demos saved per combination); 0.3–1.4 s per demo. Our demos are identical to ManiSkill's official PickCube demos for the same seeds.
- Known limitations: at some goal positions the hand hides the marker from cam0; low goals are only a few pixels.

## Step 1b — Action labelling for continuous control (the key design question)

**Question:** which action labelling / objective keeps encoders alignable from labels alone **and** keeps enough within-action information for fine control?

**Policy class:** ManiSkill's Diffusion Policy (see Step 2). SupCon goes on the observation-encoder output (= z); the conditional denoiser is the controller. Agreement between stitched and native agents is measured as a distance between predicted action chunks.

> On one ManiSkill3 task, two visual domains (e.g., two camera poses) and the default robot, train encoders + controllers with each candidate:
> 1. BC (reference, no collapse)
> 2. SupCon on per-dimension action bins (SCIL mixed-radix)
> 3. SupCon on k-means clusters of actions (K ∈ {8, 16, 32})
> 4. SupCon on k-means clusters of action chunks (e.g., 4–8 steps)
> 5. SupCon on clusters in a TACO action space (action encoder trained on actions pooled from both domains, so labels are shared)
> 6. Rank-N-Contrast on continuous actions
> 7. Fixed simplex (ETF) head on the cluster labels
> 8. Hybrid: best of 2–5 + TACO temporal term (leading candidate after Step 0c)
> 9. Temporal anchors: anchor pairs matched on action + steps-to-grasp (or task phase). Test only if the probe-transfer test below shows a drop.
>
> For each: native success rate, NC1 / effective rank, label-only alignability (offline action agreement and closed-loop success of the cross-domain stitch, vs SAPS paired), and a within-action information probe (regress the continuous action from the latent within each cluster; report R²). Also:
> - **Probe-transfer test:** fit the within-action probe on the target encoder's latents and apply it to mapped source latents, T(E_u(o)). A drop relative to the target encoder itself means within-action information exists but isn't aligned by the map.
> - **Probe recipe** (from Mario F2): centre the latents only (no per-unit scaling; near-dead ReLU units blow up), and choose the regularisation by cross-validation.

- **Exit:** one table: candidate × {native success, stitched success, % of paired ceiling, within-cluster R²}.
- **Decision:** pick the labelling/objective for the rest of the project. If nothing keeps both native performance and alignability, reframe: label-only alignment for coarse control, few-shot map refinement for fine control.

## Step 2 — Agents and oracles (core recipe done 2026-10-01; oracles for the other combinations to do)

**Question:** do our agents solve each combination on their own?

**Recipe:** ManiSkill's published **Diffusion Policy** baseline (RGB, vendored unmodified in `third_party/`), not a hand-written BC. Plain single-step BC on planner demos failed (0/100), even from the true state, because of planner rest steps and ambiguity; chunked DP is the standard answer.

**Status (2026-10-01): core recipe done, fixed only for default task / Panda / cam0, one seed (seed 1).**
- Reproduction on ManiSkill's own env: 0.77 success (their run: 0.81).
- Our env, goal marker hidden: 0.776. With the lollipop: 0.716 at 30k, 0.772 (last-3 0.789) at 50k.
- **Core recipe:** DP baseline unmodified, 100 demos, `pd_ee_delta_pos`, 50k iterations, lollipop visible,
  state = qpos, qvel, tcp_pose, **goal_pos** (no `is_grasped`): **0.82 final / 0.79 last-3 mean** (250 episodes).
  Blind check (visual feature zeroed / shuffled): ≈ 0.05, so vision is essential (the cube is only in the image).
- The 256-d visual feature (PlainConv, per frame) is the z we stitch; the conditional denoiser is the controller.
- Goal from pixels (no `goal_pos`, no `is_grasped`) is parked until Step 6b: 0.02 with 100 demos, 0.12–0.19 with 498;
  likely an encoder limitation (next idea: spatial-softmax ResNet18).
- How to train, load and read the oracles: the "Step 2 handoff" section in `EXPERIMENTS.md`.

> Next: with the objective chosen in Step 1b, train oracles for every combination needed by Steps 4–6, 3 seeds each. Write `stitch/evaluate.py` wrappers around the vendored evaluation where needed.

- **Exit:** every oracle reaches a stable success rate under the protocol above (no fixed threshold: stitched agents are compared with native agents in the same setting, not with ManiSkill's published number).
- **Decision:** fix the architecture and latent dimension for the rest of the project.
- **Design constraint from Step 6b:** if we run Step 6b, the source agents there must take privileged information (e.g. `goal_pos`) **through the encoder**, not through the controller's state input.

## Step 3 — The "why stitch?" baseline

**Question:** does a frozen foundation backbone already solve the visual shift?

> Train controllers on frozen DINOv2 and DINOv3 features: (a) raw features, (b) features plus a small trainable adapter. Train on one visual domain, evaluate on all the others, and compare with our trained encoders plus naive retraining.

- **Exit:** a table comparing frozen vs trained encoders across the visual axes.
- **Decision (important):**
  - If frozen+adapter matches the oracles on the visual shifts, move the emphasis to task and embodiment shift, and to reusing legacy encoders.
  - Otherwise, the motivation section writes itself.

## Step 4 — Visual shift, unpaired stitching (core result)

**Question:** how much of the paired ceiling can we recover with no paired frames?

**Already done offline on Mario (2026-09-30, branch `step4-aligners`):** `stitch/align.py` exists with identity, SAPS, action_pairs, GW and action_pairs + FGW. `action_pairs` is the default aligner (fast, robust at 5 pairs per action). Label-free GW is unreliable: it fails by swapping action clusters, and its objective gives no failure signal. FGW helps only non-collapsed (BC) encoders.

> Write `scripts/stitch_matrix.py`, which evaluates every encoder_u × controller_u' pair for the same task (closed-loop, protocol above) and saves an N×N success matrix per method: identity, SAPS (paired ceiling), action_pairs (≤100 and 5 per action), GW, action_pairs + FGW. Include pairs whose state distributions differ (e.g. different demo subsets or initial-state ranges), not only matched ones: in Mario, geometry sometimes worked when the two domains shared their state distribution.

- **Exit:** success matrices per method, and the % of paired ceiling.
- **Decision:** confirm the default aligner. Target: at least ~70–80% of the paired ceiling. Add the HGA / Latent Functional Maps baselines here.

## Step 5 — Task shift

**Question:** when action semantics differ, what replaces them: latent dynamics consistency, foundation-mined anchors, or a few demos?

> Implement three aligners:
> - `fit_dynamics`: train a latent forward model (TACO-style: z_t + action embedding → z_{t+K}) on the target domain; fit T so that mapped source transitions (T z_t, a_t) predict T z_{t+K} under the target model. T linear/orthogonal, initialised from action pairs when available.
> - `fit_fm_anchors`: DINO features for frames from both domains (optionally foreground crops); keep mutual nearest neighbours that pass a cycle-consistency filter; robust Procrustes on the corresponding latents, then FGW.
> - `fit_fewshot`: train T by BC through the frozen controller on K target demos.
>
> Evaluate separately on goal/reward variants (same dynamics) and actuation variants (EE-delta bounds ×0.5), for K ∈ {0, 1, 5, 20}. Compare with retraining the controller using the same K, and with Dynamics Cycle-Consistency (Zhang et al. 2021, code `sjtuzq/Cycle_Dynamics`).

- **Exit:** a K-shot curve per task shift, with the zero-shot aligners at K = 0.
- **Decision:** decide which task shifts are zero-shot and which are few-shot, and which correspondence source works where. This goes into the claim.

## Step 6 — Embodiment

**Question:** can we stitch across robots if the controller owns the proprioception?

> Stitch encoders across the Panda and xArm6 variants, keeping each robot's own controller with proprio input. Use the best aligner from Steps 4–5.

- **Exit:** a results table.
- **Decision:** does embodiment stay in the main claim, or move to the appendix?

## Step 6b — Privileged-to-deployable transfer (sim-to-sim proxy for sim-to-real)

**Question:** can a controller trained with privileged information be reused, through stitching, by an encoder that must read that information from pixels in a harder domain, **with fewer target demos than training a new policy**?

**Why:** this is the application story. In sim-to-real, the source (simulation) can use privileged information and plenty of demos; the target (real) has only pixels and few, expensive demos. We have no real robot, so this is a **sim-to-sim proxy**, and the paper must call it that.

**Design:**
- **Source agent:** privileged information enters **through the encoder**: E_src(image, goal_pos) → z, controller C(z, proprio). If the privileged input went to the controller directly, stitching could not replace it.
- **Target domain:** harder than the source: different look / light / camera, no `goal_pos` and no `is_grasped` in any input, optionally sensor effects (noise, blur). The goal is visible only through the marker.
- **Target encoder:** E_tgt(image) only, trained with the Step 1b objective on N target demos, then stitched to C with `action_pairs`.

> Build the source agent (encoder with the privileged input) and the target domain above. For N ∈ {10, 25, 50, 100} target demos, 3 seeds each, compare closed-loop success of:
> 1. **stitched:** E_tgt (trained on N demos) + action_pairs map + frozen source controller C;
> 2. **target from scratch:** a full policy trained on the same N target demos;
> 3. **fine-tuned source:** the source policy fine-tuned on the N target demos, with the privileged input removed (the standard adaptation baseline);
> 4. **paired ceiling:** the same stitch with SAPS on paired frames (re-rendered from stored states; not available on a real robot).
> Write the hypotheses first: stitched > target-from-scratch at small N, with the gap closing as N grows.

- **Exit:** a data-efficiency curve: success vs N for the four methods.
- **Decision:**
  - If stitching clearly beats training from scratch and fine-tuning at small N, this becomes the paper's application section.
  - If the curves overlap, drop it: the utility argument doesn't hold, and it's better to know early.
- **Related work to position against:** Latent Adaptation of Foundation Policies for Sim-to-Real (ICLR 2026; small target data), A Stitch in Time (perception updates, paired or task-supervised), privileged teacher → student distillation (e.g. Learning by Cheating). Our difference: label-only alignment, no paired data, and the data-efficiency comparison.

## Step 7 — Stitchability score

**Question:** can we predict stitch success without doing rollouts?

Mario pilot: offline agreement predicts in-game play for well-trained agents (ρ ≈ 0.69) but not for weak ones (ρ ≈ 0). Mario Step 4: the GW objective is not a usable failure signal.

> In `stitch/score.py`, compute: held-out anchor residual, cycle error, GW distortion, CKA, and action agreement on a few labelled target samples. Over all pairs from Steps 4–6b, report the Spearman ρ of each component and of their combination against success, plus the ROC of a deploy / don't-deploy threshold. Exclude or flag pairs whose native agents are weak.

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
  - batch-size control for the TACO term (Step 0c confound)
- **Writing:**
  - The intro builds on the limitations stated by Policy Stitching and Perception Stitching, and positions against Zhang et al. 2021 (unpaired, but trained translators) and A Stitch in Time (paired or task-supervised).
  - The mechanism section: action collapse (NC1), the projection-head ablation, why NC2 is not needed, and why label-free geometry is unreliable (cluster swaps, no failure signal).
  - The main table has three column groups: visual | task | embodiment.
  - The application section (if Step 6b holds): privileged-to-deployable transfer, framed as a sim-to-sim proxy.
  - Paper-quality Mario rows: more episodes and confidence intervals.
  - Post to arXiv early, because of the scoop risk.
