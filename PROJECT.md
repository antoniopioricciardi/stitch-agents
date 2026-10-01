# Anchor-free model stitching for autonomous agents — project brief

_Last updated: 30 Sep 2026. Owner: Antonio._

## Thesis (one line)

Reuse encoders and controllers that were trained separately: recombine them zero-shot or few-shot, **without paired observations**, instead of retraining.

## Where we come from

- **R3L** (arXiv 2404.12917, ours): relative representations for RL.
  - Training-time method: the anchors are fixed before training.
- **SAPS** (arXiv 2503.01881, ours): a post-hoc orthogonal/affine map between latent spaces, estimated from semantically aligned anchor frames.
- **Shared weakness of R3L and SAPS:** both need paired or aligned frames across environments.
- **SCIL** (Celemin et al., Sony, arXiv 2509.11880): a SupCon loss on discretized action classes, used in imitation learning.
  - It organizes latent spaces by action semantics, which gives correspondences without paired frames.
  - Works on SuperMario in IL.
  - Online RL (PPO) showed gradient conflicts.
- **Action-contrastive representations predate SCIL:** ACO (Zhang, Peng, Zhou, ECCV 2022; driving pretraining from YouTube, positives = similar actions) and ADAT (Kim et al., 2022). Neither is used for alignment or stitching. Cite both.
- **Closest robotics work:**
  - Policy Stitching (Jian et al., CoRL 2023): low-dimensional states, k-means anchors.
  - Perception Stitching (TMLR 2024): pixels, anchors from trajectory replay, needs 256–512 anchors.
  - Both list anchor-free alignment as future work.
- **Unpaired cross-domain correspondence for control already exists:** Dynamics Cycle-Consistency (Zhang, Xiao, Efros, Pinto, Wang, ICLR 2021 oral). It learns correspondences across modality (vision vs state), physics (mass, friction) and morphology from unpaired, randomly collected data, then transfers the policy without fine-tuning.
  - *How we differ:* they train translation networks with adversarial and cycle losses on top of a pretrained forward model, and report shortcut/collapse issues when training end to end. We fit a closed-form linear map between modules that already exist.
  - *Consequence for the claim:* we cannot claim "first unpaired transfer in control". Our claim is post-hoc, closed-form, label-only alignment of independently trained modules, with a mechanism (action collapse).
  - Code: `sjtuzq/Cycle_Dynamics`. Baseline for task/actuation shift.
- **Compatible representation learning** (image retrieval): CoReS gets representations that stay compatible across model updates by fixing the classifier weights to the vertices of a regular polytope, so no mapping between representations is ever learned. A 2026 follow-up shows that d-Simplex fixed classifiers give compatibility in expectation, and combines cross-entropy with a contrastive loss.
  - *How we differ:* they coordinate the trainings (every model shares the same fixed prototypes). Ours are trained independently, share only the label set, and the map is found afterwards. They work on retrieval; we work on closed-loop control.
  - *Tool we can borrow:* a fixed simplex (ETF) head would enforce the equal-angle geometry (NC2) that our encoders lack. Test it as a recipe, and as a "coordinated training" upper bound.
- **Action Collapse** (arXiv 2509.02737): in policy-gradient networks, last-layer features of states sharing the same optimal action collapse to that action's mean, and the means form a simplex ETF. They use it to improve training, not for stitching. **Must cite; we do not claim to have discovered action collapse.** It supports our mechanism.
- **Neural collapse under class imbalance** (Dang et al., arXiv 2401.02058): with imbalanced classes, class means move away from the simplex ETF. This is the textbook explanation for the missing NC2 in our Mario encoders.
- **TACO** (Zheng et al., NeurIPS 2023, arXiv 2306.13229): temporal contrastive loss that learns state and action representations together (current state + action sequence ↔ future state); theoretically sufficient for Q*; its action encoder groups actions by effect.
  - Its positives are temporal, not "same action", so on its own it should **not** produce action collapse, and hence should not give label-only alignability (to be checked in Mario, Step 0c).
  - Possible roles for us:
    1. the information-preserving half of a hybrid loss (SupCon on coarse action labels + TACO), to keep within-action detail for continuous control;
    2. its action space as the labelling function (cluster actions by effect; train on actions pooled across domains, so labels are shared for free);
    3. its latent forward model as a label-free alignment signal under goal/reward shift (fit T so mapped source transitions obey the target's latent dynamics; a latent, linear version of Zhang et al. 2021).
  - Cost: large batches (1024 in the paper), ~3.6× slower than DrQ-v2. Acceptable offline.
- **A Stitch in Time Saves Nine** (arXiv 2606.21509, June 2026): linear and convolutional stitchers that restore compatibility between updated perception modules and frozen driving policies (nuScenes → CARLA: >91% of the no-shift score, adaptation 22.18 h → 0.91 h). The linear stitcher needs paired anchors; the convolutional one is trained with task labels through the frozen decoder, which is essentially our few-shot fallback. They note similar latent clustering between models trained with the same supervision but never use it for label-only alignment.

## Working claim

Independently trained visuomotor agents can be recombined without paired frames. Training encoders with action-supervised contrastive learning collapses each action onto its centroid, so the map between two encoders can be fitted **post hoc and in closed form from action labels alone** (**frame-free, label-aligned**). We study why this works, how to keep it working in continuous control, where collapse discards fine within-action information, and how to predict whether a stitch will work before deployment.

Scope of the variations between the two agents:

- **Visual variation:** textures, colours, lighting, camera pose, sensors.
- **Task variation:** actuation/dynamics, goals, rewards, embodiment/joints.

Where action labels do not share a meaning across domains, correspondences come from latent dynamics consistency, frozen foundation-model features, or K demos (few-shot).

## Regimes

| Regime | What the two domains share | Correspondence source |
|---|---|---|
| Visual shift, same task | Action semantics | Random same-action frame pairs (≈ action-class prototypes) |
| Goal / reward shift (main task-shift result) | Dynamics (same physics), not action meaning | Latent dynamics consistency (TACO-style forward model); else foundation-mined anchors; else K demos |
| Actuation / dynamics shift (EE-delta bounds ×0.5) | Neither exactly | Foundation-mined anchors (mutual NN in DINO space + cycle filter), approximate dynamics consistency; else K demos |
| Embodiment shift | — | The controller gets proprioception (Policy-Stitching style); only perception is stitched |

Note: a friction/mass variant was dropped. It did not change the expert's actions, because the motion planner is open-loop and ignores physics, so from the controller's point of view there was no task shift.

## Method sketch

**Setting.** We have an encoder E_u trained on domain u and a controller C_v trained on domain v. The goal is a map T such that C_v(T(E_u(o))) acts well.

0. **Encoder training (part of the method).** Encoders are trained with an action-supervised contrastive loss (SCIL: SupCon directly on the latent the controller reads). Plain BC encoders do not align from labels alone, and neither does SupCon on a projection head (Mario Step 0).
   - **Open for continuous control:** how to define the action labels, and how to keep within-action information. Candidates: per-dimension bins, k-means on actions or action chunks, clusters in TACO's action space, Rank-N-Contrast, fixed simplex (ETF) head, SupCon + TACO hybrid.
1. **Correspondences.**
   - Visual shift: random pairs of frames that share the same action (≤100 per action), equivalent to action-class prototypes.
   - Goal/reward shift: latent dynamics consistency.
   - Actuation / dynamics shift: foundation-mined pseudo-anchors.
2. **Map.** Orthogonal Procrustes by default, affine as an ablation. Refine with fused Gromov–Wasserstein, combining latent geometry, action agreement and optionally DINO similarity. An MLP map is used only in the few-shot setting.
3. **Few-shot.** Fit T (or a low-rank correction to it) with BC through the frozen C_v on K target demos, K ∈ {0, 1, 5, 20}.
4. **Stitchability score** (computed without rollouts). It combines:
   - anchor residual
   - cycle-consistency error
   - GW distortion
   - CKA
   - action agreement (Mario pilot: offline agreement vs in-game ρ ≈ 0.69 for well-trained agents, ≈ 0 for weak ones)

   It is validated by Spearman correlation against closed-loop success over all encoder × controller pairs.

## Baselines

- **Must have:**
  - identity stitch (lower bound)
  - oracle trained on the target combination (upper bound)
  - controller retraining with N demos
  - SAPS with paired anchors (the ceiling)
  - frozen DINOv2/v3 (+ small adapter) + controller
  - ILA
  - unpaired geometric alignment: GW, HGA, Latent Functional Maps, vec2vec-style
  - Dynamics Cycle-Consistency (Zhang et al. 2021), for task/actuation shift
- **Should have:** Perception Stitching, R3L, DrQ-v2/SVEA, A Stitch in Time (CARLA part).
- **Nice to have:** HPT / Devin et al., PoCo, fixed-simplex coordinated training (upper bound for training-time protocols).

## Environments (recommended)

- **Main: ManiSkill3.**
  - Visual axes: camera pose, textures/colours, lighting.
  - Task axes: actuation (EE-delta bounds ×0.5), goal variants.
  - Embodiment: Panda vs xArm, and a locked joint.
  - Experts: motion-planning demos.
  - GPU-parallel, so it runs on one node.
- **Secondary: CARLA** (Bench2Drive subset, PDM-Lite expert).
  - Visual axes: weather, camera placement.
  - Task axes: target speed, vehicle dynamics, route.
- **Continuity:** Mario / CarRacing, one row each.

## Metrics

- Success rate.
- % of oracle recovered = stitched / oracle on the target combination.
- % of paired ceiling = stitched / SAPS-paired.
- Compute saved versus retraining.

## Risks and falsifiers

- **Collapse vs fine control.** Collapse is what makes label-only alignment work, but it discards within-action information that continuous control needs. If no labelling/objective keeps both alignability and native performance in ManiSkill, the method is limited to coarse/discrete control. Even if within-action information is kept, label anchors do not align it; stitched performance can lag native performance for this reason.
- **Frozen DINO + adapter matches our method on most axes.** Then we reposition the paper around task/embodiment shift and reuse of legacy encoders. Published evidence (OpenVLA; DINOv3 diffusion policy on PushT, 0.39 frozen vs 0.84 fine-tuned) and our Mario frozen-backbone runs suggest frozen backbones aren't free performance.
- **Unpaired alignment stays well below the paired ceiling** (under ~70%) under visual shift.
- **The stitchability score doesn't correlate** with closed-loop success.
- **Task shift:** action prototypes are misaligned by construction, and DINO anchors can be fooled by texture. Mitigation: foreground crops, cycle filtering, dynamics consistency.

## Closest competitors (possible scoop)

- A Stitch in Time Saves Nine (arXiv 2606.21509, June 2026): stitching perception updates to frozen driving policies; paired anchors or task-label training. Baseline for the CARLA part; its paired-frame requirement is our entry point.
- CoReS / d-Simplex compatible representations: compatibility without mappings, but with coordinated training (shared fixed prototypes).
- HGA (LLNL, arXiv 2608.28840, Aug 2026): unsupervised stitching, not yet applied to control.
- The Latent Functional Maps / latent-communication group.
- Boyuan Chen's lab (Policy Stitching / Perception Stitching authors).
- LAST (ICML 2026): GW alignment in VLAs.

Novelty check (29 Sep 2026, ~10 targeted searches): no paper found that aligns independently trained visuomotor encoders post hoc from action labels alone. Good coverage, not a guarantee. Post an arXiv version early.

## Venue and timeline

- ICLR 2027 deadline has passed.
- **Target: ICML 2027** (around late January 2027; unofficial date, verify).
- Fallback: NeurIPS / CoRL 2027 (around late May 2027).
- CARLA and embodiment results can go into the NeurIPS/CoRL version if needed.

## Out of scope for now

- Offline-RL trajectory stitching: follow-up paper.
- Online RL: appendix at most.
- Swapping controllers mid-episode: a single demo experiment.

## Open questions

1. Does prototype alignment work post hoc on plain BC encoders, or is SupCon training required? *(Answered 2026-09-29 in Mario: SupCon on the latent is required; see Decisions.)*
2. How many samples per action class are needed? *(Partly answered in Mario: 5 per class gives 73% of native in-game vs 88% with all; ≤100 random pairs per action is the adopted recipe.)*
3. Does frozen DINO + adapter close the gap on its own? *(Mario: no, but upscaled NES frames are a weak test; redo in ManiSkill, Step 3.)*
4. Are foundation-mined anchors reliable under task shift, or do we need few-shot?
5. Is an orthogonal map enough, or do we need affine/MLP?
6. **How should continuous actions be labelled so that encoders stay alignable without losing within-action information?** (Step 1b.)
7. Does a TACO-style temporal term preserve or destroy label-only alignability when combined with SupCon? (Step 0c.) *(Answered 2026-09-29 in Mario: the temporal term preserves alignability; see Decisions.)*
8. Is latent dynamics consistency a reliable correspondence signal under goal/reward shift? (Step 5.)
9. Label anchors only fix the map on the span of the action centroids. Is the within-action structure added by TACO aligned across encoders, or scrambled by the map? Candidate fixes: finer labels (action chunks, more clusters), temporal anchors (pair by action + steps-to-event).
10. When does label-free geometry work? Mario: GW fails on mismatched halves of a level for SCIL, but works on mismatched halves and across levels for SCIL+TACO (every seed), while failing on some matched version pairs. So matched state distributions are neither necessary nor sufficient. Shared dynamics structure from the temporal term, or optimisation luck (which local optimum the solver finds)?

## Decisions

- 2026-09-26: rewrite the paper and the code from scratch, keeping only the philosophy ("reuse encoders and controllers when possible").
- 2026-09-28: follow the direction of the research report (anchor-free stitching). Environment choice to be confirmed after Step 1 of PLAN.md.
- 2026-09-29 (Step 0, Mario): SupCon is part of the method: a training-time recipe, not post hoc. Unpaired action anchors match the paired SAPS ceiling on SCIL encoders but not on BC encoders (answers open question 1). SCIL does not make agents robust; it makes them alignable without paired data.
- 2026-09-29: anchors = random pairs of frames sharing the same action (≤100 per action, full training set), as in the old repo. Equivalent to action centroids in all tests; the map keeps the latent dimension (512).
- 2026-09-29 (mechanism, Mario): **NC1 holds, NC2 does not.** SCIL latents collapse each action onto its centroid (within/between spread 0.47 vs 4.97 for BC; effective rank ~7), but the centroids are not an equiangular simplex (no closer than BC's, likely minority collapse from imbalanced actions). With SupCon on a projection head the latent does not collapse and label-only alignment drops to about BC level. So collapse is the mechanism, and in Mario we align the controller's decision space.
- 2026-09-29: working claim reworded to "frame-free, label-aligned": we use no paired frames, but we do use action labels in both domains.
- 2026-09-29 (novelty check): unpaired correspondence for control exists (Zhang et al. 2021), so the claim is now "post-hoc, closed-form, label-only alignment of independently trained modules, explained by action collapse". Added Zhang et al. 2021 as a baseline, and ACO/ADAT/TACO/imbalanced-NC to related work.
- 2026-09-29 (TACO): not a replacement for SCIL (no action collapse expected). Candidate roles: hybrid loss for continuous control, labelling function, and latent dynamics-consistency aligner for goal/reward shift.
- 2026-09-29 (Step 0c, Mario): SCIL + TACO (K=3) keeps SCIL's label-only alignability (matches SCIL in-game) and recovers native play lost to collapse (1-2: 654 → 1207). The within-action probe on 1-1 supports SCIL < BC ≈ SCIL+TACO (future-action accuracy: SCIL at the majority baseline, 0.624 vs 0.621; SCIL+TACO 0.654; BC 0.664), but the absolute signal is weak. SCIL+TACO K=3 is the leading candidate for Step 1b. Mario is frozen.
- 2026-09-29 (F1): no-map agreement across levels is at chance (the controller gets stuck on "right"); always report it next to its chance level.
- 2026-09-29: physics variant replaced by actuation; planner is open-loop, so friction/mass left expert actions unchanged.
- 2026-09-30 (Step 4 offline, Mario): action_pairs is the default aligner (fast, robust at 5 pairs/action). Label-free GW is unreliable: it fails by swapping action clusters, and seed/pair-dependent. The GW objective gives no usable label-free failure signal. FGW helps only non-collapsed (BC) encoders; kept as a candidate for open question 9.
- 2026-10-01: core experiments keep goal_pos in the state and remove is_grasped (DP baseline, 100 demos, 50k iterations: 0.82 final / 0.79 last-3; blind ≈0.05, so vision is essential because the cube is only in the image). Goal-from-pixels is parked until Step 6b (498 demos: 0.12–0.19; likely an encoder limitation; next idea: spatial-softmax ResNet18).
