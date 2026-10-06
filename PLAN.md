# PLAN.md — step-by-step discovery plan

_Rewritten 4 Oct 2026, after Step 1b closed. Target: ICML 2027 (deadline ~late January 2027, unofficial; verify)._

Each remaining step answers **one question**, with a **prompt** for Claude Code, an **exit criterion** and a **decision**. Don't start a step until the previous exit criterion is met. If a result surprises you, stop and discuss it; the plan is allowed to change. Details of finished steps are in `EXPERIMENTS.md`; decisions are in `PROJECT.md`.

**Protocols (all ManiSkill steps):**
- Evaluation: final checkpoint (mean of the last 3 in brackets), never the best checkpoint; 250 episodes; success at any step and at the end.
- Oracles: accepted if last-3 success_once ≥ 0.3 (clearly above blind ≈ 0.05) plus the three checks (blind, fake goal, cube/goal probe). Hypothesis levels are reported, never used as gates.
- Job chains: a failed gate blocks only the jobs that depend on that model; independent jobs keep running (order chains so independent phases come first).
- Stitching: encoder seed s with controller seed s+1; label fits on disjoint demo halves (source 0–49, target 50–99); headline metric = % of the affine paired ceiling, per seed pair.
- Fine-tuning: a fixed iteration budget chosen in advance is the headline (5k for map-only fine-tuning; final checkpoint for DP from scratch); early-stopped checkpoints (sampled-chunk error on 10 held-out validation demos) are secondary numbers only. Never select on evaluation success. Every low-data arm gets N training demos + the same 10 validation demos. (DP's held-out denoising loss is not a proxy for success: it rises while success rises.)
- Pilots (one seed) are never conclusions; paper numbers use 3 seeds.

---

## Where we are (done)

- **Step 0 / 0c — Mario:** SCIL (SupCon on actions) makes encoders alignable from labels alone (NC1 collapse; the projection-head ablation removes it). SCIL + TACO keeps alignability and recovers native play. Label-free GW is unreliable.
- **Step 1 — ManiSkill3 PickCube env:** cameras, looks, lights; actuation and goal variants; Panda and xArm6; lollipop goal marker; ~500 demos per (task, robot), re-renderable from stored states.
- **Step 2 — Oracles:** ManiSkill's Diffusion Policy, unmodified; `goal_pos` in the state, `is_grasped` removed; 50k iterations (100k for the xArm). Working second domains: look 1, look 2, look 2 + light 1, cam3 (15° rotation). Limitations: cam1, cam2, light 1.
- **Step 1b / Step 4 — Labels and the main matrix:**
  - SupCon on chunk clusters is a measured tradeoff (branch closed).
  - Plain DP is already compressed (regression collapse); `nn_affine` is the default aligner: **77% ± 10% of the paired ceiling** over 4 shifts × 3 seed pairs × 2 directions, above paired SAPS (64%). The affine paired ceiling reaches ~89% of native.
- **First combinations and low-data adaptation (one-seed pilots, superseded by Step R below):**
  - goal shift: label-only 60–64% of the ceiling;
  - embodiment Panda → xArm: 83% of the ceiling;
  - map-only fine-tuning from a stitched start: 66% / 77% of the reference oracle with 10 / 25 demos (scratch: 0.01 / 0.08);
  - encoders trained on 5–25 demos are unstructured and can't be stitched.

- **Step R — Combinations and low-data adaptation, 3 seed pairs (done, 6 Oct):**
  - goal shift: nn_affine 69% ± 5% of the affine ceiling (orthogonal maps 45–51%, identity 7%);
  - embodiment Panda → xArm: ceiling 56% of the xArm reference; nn_affine and nn_orth both work, no consistent winner (1000 episodes);
  - map-only fine-tuning: 60% / 74% of the target oracle with 10 / 25 demos (DP from scratch 0–16%), goal shift = same task;
  - no label-only criterion selects the map class; nn_affine stays the default.

**Rough timeline for what remains (about 16 weeks):**
- Weeks 1–2 (to ~18 Oct): Step R (done), Step F, Step 3
- Weeks 3–4 (to ~1 Nov): Steps 5b and B
- Weeks 5–6 (to ~15 Nov): Steps 7 and RL
- Weeks 7–8 (to ~29 Nov): Step 6b, plus Step 8 if time allows
- Weeks 9–10 (to ~13 Dec): ablations, paper-quality figures, Mario rows with CIs
- Weeks 11–15 (to ~18 Jan): writing, internal review, arXiv
- Buffer: about one week

CARLA moves to the NeurIPS/CoRL version. Genre-level transfer across games is a separate follow-up paper.

---

## Step R — Replicate the combinations and the low-data adaptation (3 seeds) — done 6 Oct

_Done: see "Where we are" and the Step R entry in EXPERIMENTS.md. **Split confirmed (6 Oct):** main paper = the goal-shift row, a short embodiment row with both map classes ("both work, no consistent winner"), and the data-efficiency figure (map-only fine-tuning vs DP from scratch, goal shift and same task); appendix = per-pair tables, early-stopping numbers, oracle checks, the s2 → s2 pilots, the map-selection test._


**Question:** do the one-seed results (goal shift, embodiment, map-only fine-tuning) hold over seeds?

> For seeds 1 and 3 (seed 2 exists), train the agents the combinations need: cam0 goal-variant and look 2 goal-variant agents; cam0 xArm (100k iterations) and look 2 xArm agents. Three checks each. Then, per seed pair (s → s+1):
> 1. goal shift: look 2 default encoder → cam0 goal controller, on the goal variant in look 2;
> 2. embodiment: look 2 Panda encoder → cam0 xArm controller, on the xArm in look 2;
> with identity, the affine ceiling, SAPS, nn_orth, nn_affine and nn_affine-PCA16.
> Then the low-data adaptation on the goal-shift combination, for every seed pair: map-only fine-tuning from the stitched start, N ∈ {5, 10, 25}, 5k-iteration budget, early stopping on held-out demos, plus from-scratch DP on the same N.
> Add the same-task counterpart (look 2 encoder → cam0 controller, on look 2): map-only fine-tuning from the stitched start, N ∈ {5, 10, 25}, so the same-task data-efficiency figure is complete.
> Hypotheses first. Report mean ± std over 3 seed pairs.

- **Exit:** combination table and data-efficiency curves with 3 seeds.
- **Decision:** which combination results go into the main paper and which into the appendix.

## Step F — Which part to adapt with few demos (small ablation)

**Question:** given what changed (visuals vs task), is fine-tuning only the map always best, or does adapting the encoder or (lightly) the controller as well help?

Known so far: from a stitched start, map-only fine-tuning is the best arm; end-to-end fine-tuning destroys encoder structure; adapting a non-matching encoder alone (cam0 encoder → look 2) is weak.

> From the stitched start (nn_affine-PCA16, as in Step R), N ∈ {10, 25}, fixed 5k budget, seed pair 1 first:
> (a) map + encoder (controller frozen);
> (b) map + a light controller adaptation (last denoiser layers only, or a LoRA adapter; propose which, keeping the vendored change minimal);
> compared with map-only (Step R numbers, same pair).
> On two combinations: same task (look 2 enc → cam0 ctrl, a visual change) and goal shift (look 2 enc → cam0 goal ctrl, a task change). Log NC1 / effective rank of the encoder after fine-tuning. Hypotheses first: (a) ≤ map-only on both; (b) > map-only on the goal shift only. If a variant beats map-only by more than the eval noise, replicate on all 3 seed pairs.

- **Exit:** a small table: {map only, map + encoder, map + light controller} × {visual change, task change} × N.
- **Decision:** a rule for the paper ("what to adapt, given what changed"), or confirmation that map-only is the right default everywhere. Ablation-sized: runs in a gap before Step 3's trainings.

## Step 3 — "Why stitch?" and pretrained encoders

**Question:** does a frozen or fine-tuned foundation encoder remove the need for stitching, or provide the structure that few demos can't?

> On cam0 and look 2 (default task, Panda, seeds 1–3), train DP agents whose visual encoder is a pretrained DINOv2 (or DINOv3) ViT-S:
> (a) frozen backbone + a small trainable adapter;
> (b) backbone fine-tuned on the full demos;
> (c) backbone fine-tuned on only N ∈ {5, 10, 25} demos.
> For each: native success, the three checks, NC1 / effective rank of z.
> Then: (1) run the cam0 agent unchanged in look 2 (does a shared pretrained backbone already transfer?); (2) stitch cam0 ↔ look 2 with identity, the affine ceiling and nn_affine; (3) for (c), stitch the N-demo agent's encoder to the cam0 controller (does pretraining give the structure that 5–25 demos alone couldn't?).
> Hypotheses first. Keep the vendored DP change minimal and clearly marked (encoder swap only).

- **Exit:** table of frozen / fine-tuned / few-demo pretrained encoders × {native, unchanged in the other domain, stitched}.
- **Decision:**
  - If frozen + adapter transfers across our shifts as well as stitching, the paper emphasises combinations (task, embodiment) and the reuse of existing specialised agents.
  - If (c) stitches well, it becomes the low-data route: pretrained encoder + a few demos, then stitch.

## Step 5b — Actuation combination (motion-based pairs)

**Question:** when commanded actions change meaning (EE-delta bounds ×0.5), can pairs built from the robot's actual motion replace action labels?

> Train the actuation-variant agents in cam0 and look 2 (seed 2 first, three checks each). Stitch look 2 default encoder → cam0 actuation controller, on the actuation variant in look 2. Label maps from three pairings:
> (i) commanded-action chunks (expected to mismatch);
> (ii) motion chunks: the change in TCP position over 8 steps, from the stored proprio;
> (iii) motion chunks + gripper state.
> Plus identity and the affine ceiling. If neither motion pairing works, the fallbacks from the original plan are latent dynamics consistency (a TACO-style forward model on the target side) and few-shot map fine-tuning; propose before implementing. If (ii) or (iii) works, replicate on 3 seeds, and also re-run goal shift and embodiment with motion pairs to see whether motion pairing is a general replacement for action pairing.

- **Exit:** actuation table; one sentence on whether motion pairing generalises.
- **Decision:** does "align by what the robot did" enter the method as the general pairing rule, or stay a variant for actuation shift?

## Step B — Baselines

**Question:** how does label-only stitching compare with the closest published methods?

> 1. **Perception Stitching (Jian et al., TMLR 2024; code: generalroboticslab/PerceptionStitching).** Preferred: run our label-only stitching on their robomimic setup (their tasks and camera configurations, their BC policies, retrained if needed), against their reported PeS numbers and their anchor requirement (trajectory replay). Fallback: implement PeS (relative representations + disentanglement loss) as a baseline in our ManiSkill setup. Propose which before starting.
> 2. **Unpaired geometric alignment** on the main matrix (one seed pair per shift is enough): GW, and HGA or Latent Functional Maps if code is available. Same label budget rule (label-free methods use no labels anywhere).
> 3. **Retraining baselines**, already partly in Step R: controller retrained on N demos; DP from scratch.
> 4. **For task/actuation shift:** Dynamics Cycle-Consistency (Zhang et al., ICLR 2021; code: sjtuzq/Cycle_Dynamics) on the actuation or goal combination, if its code runs on our setup within a few days; otherwise cite and explain. ILA (latent distribution matching at deployment) on one visual shift.

- **Exit:** a baseline table for the paper.
- **Decision:** which baselines go in the main table, which in the appendix.

## Step 7 — Stitchability score

**Question:** can we predict, without rollouts, whether a stitch will work and which map to deploy?

Evidence so far: the held-out z residual ranks map types correctly across domains; encoder structure (NC1, effective rank) separates stitchable from unstitchable encoders; offline chunk agreement overrates label maps; the GW objective is no failure signal.

> In `stitch/score.py`, compute for every stitch evaluated in Steps 4, R, 3 and 5b: held-out z residual, encoder NC1 and effective rank (both sides), cycle error, CKA, and offline chunk distance on vision-sensitive frames. Report the Spearman ρ of each with closed-loop % of ceiling and with absolute success, pooled and per shift type, plus the ROC of a deploy / don't-deploy rule. Exclude or flag pairs whose native agents fail the oracle checks.

- **Exit:** a correlation figure and one recommended score.
- **Decision:** headline contribution or analysis section.

## Step RL — Appendix: label-based stitching for RL agents

**Question:** is the method specific to imitation learning?

> Take the PPO agents from SAPS (CarRacing, existing checkpoints). Collect rollouts with actions from each agent in its own domain, fit nn_affine from action pairs (no paired frames), and compare with SAPS (paired) and identity, in closed loop. Then a fine-tuning curve: PPO from the stitched start vs PPO from scratch (the LunarLander observation, made systematic).

- **Exit:** one appendix table + one learning-curve figure.

## Step 6b — Privileged-to-deployable transfer (sim-to-sim proxy), redesigned

**Question:** can a controller trained with privileged information be reused in a harder target domain with few demos?

Redesign after Step 1b: an encoder trained from scratch on N demos is unstructured and can't be stitched. So the target encoder must come from an existing competent encoder (fine-tuned on N target demos, encoder-only, controller frozen; or a pretrained encoder from Step 3), and adaptation is map-only from the stitched start.

> Source agent: privileged information (`goal_pos`) enters through the encoder. Target domain: harder look/light, no `goal_pos` in any input (the goal is only visible through the marker; may need the spatial-softmax encoder parked in Step 2). For N ∈ {10, 25, 50}, compare: stitched start + map-only fine-tuning; source policy fine-tuned end-to-end; target from scratch; paired ceiling.

- **Exit:** a data-efficiency curve for the four arms.
- **Decision:** application section, or drop. This depends on goal-from-pixels working at all, so it's the most at-risk step; cut it first if time runs short.

## Step 8 — Controller swap mid-episode (optional)

> Two-stage task (reach, then grasp/carry). Run controller A, switch to controller B at the hand-off, both reading the same encoder through their aligned maps. Measure hand-off success; check whether the Step 7 score predicts it.

- **Exit:** one figure plus a video, or skip.

## Step 10 — Ablations and writing

- **Ablations:**
  - correspondence source: action pairs, motion pairs, K-cluster pairs, GW-only;
  - map class: orthogonal, affine, PCA-16 affine, MLP;
  - encoder objective: plain DP, SupCon λ sweep (done), pretrained (Step 3);
  - number of label pairs / demos;
  - latent dimension.
- **Paper-quality extras:** Mario rows with more episodes and confidence intervals; the SupCon tradeoff figure; the map-class figure (orthogonal vs affine ceiling).
- **Writing:**
  - Intro: reuse of competent modules without paired frames; positions against Policy/Perception Stitching (anchors), Zhang et al. 2021 (trained translators), A Stitch in Time (paired or task-supervised), MVD (training-time multi-view).
  - Mechanism: action collapse (discrete) and regression collapse (continuous) make latents alignable from actions; the collapse-vs-control tradeoff; structure requires competent encoders.
  - Results: main matrix (visual shifts), combinations (goal, embodiment, actuation), low-data adaptation, baselines, stitchability score.
  - Limitations: strong viewpoint changes and low-contrast domains break the oracles themselves; goal-from-pixels; one task family (PickCube); no real robot.
  - Post to arXiv early, because of the scoop risk.
