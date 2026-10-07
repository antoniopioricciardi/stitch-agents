# Label-aligned model stitching for autonomous agents — project brief

_Last updated: 7 Oct 2026. Owner: Antonio._

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
- **Neural Regression Collapse** (Andriopoulos et al., NeurIPS 2024, arXiv 2409.04180): multivariate regression, including imitation learning, has its own collapse: last-layer features collapse onto a subspace whose dimension equals the target (action) dimension. Relevance: plain continuous BC/DP encoders may already have a low-dimensional, action-aligned structure, a possible route to aligning continuous actions without discretising. Measured in Step 1b (variance in the top-n principal components of z).
- **Regression collapse hurts generalisation** (arXiv 2510.01105): unlike in classification, collapse in multivariate regression consistently degrades performance and correlates with higher test error. Literature support for our "collapse vs fine control" risk, and for why the SCIL+TACO hybrid (which restored native performance in Mario) matters.
- **Neural collapse under class imbalance** (Dang et al., arXiv 2401.02058): with imbalanced classes, class means move away from the simplex ETF. This is the textbook explanation for the missing NC2 in our Mario encoders.
- **TACO** (Zheng et al., NeurIPS 2023, arXiv 2306.13229): temporal contrastive loss that learns state and action representations together (current state + action sequence ↔ future state); theoretically sufficient for Q*; its action encoder groups actions by effect.
  - Its positives are temporal, not "same action", so on its own it does **not** produce action collapse and does not give label-only alignability (confirmed in Mario, Step 0c). Combined with SupCon (SCIL + TACO, K=3) it keeps alignability and recovers native performance (see Decisions).
  - Possible roles for us:
    1. the information-preserving half of a hybrid loss (SupCon on coarse action labels + TACO), to keep within-action detail for continuous control;
    2. its action space as the labelling function (cluster actions by effect; train on actions pooled across domains, so labels are shared for free);
    3. its latent forward model as a label-free alignment signal under goal/reward shift (fit T so mapped source transitions obey the target's latent dynamics; a latent, linear version of Zhang et al. 2021).
  - Cost: large batches (1024 in the paper), ~3.6× slower than DrQ-v2. Acceptable offline.
- **A Stitch in Time Saves Nine** (arXiv 2606.21509, June 2026): linear and convolutional stitchers that restore compatibility between updated perception modules and frozen driving policies (nuScenes → CARLA: >91% of the no-shift score, adaptation 22.18 h → 0.91 h). The linear stitcher needs paired anchors; the convolutional one is trained with task labels through the frozen decoder, which is essentially our few-shot fallback. They note similar latent clustering between models trained with the same supervision but never use it for label-only alignment.

## Working claim

Independently trained visuomotor agents can be recombined without paired frames. The map between two agents' encoders is fitted **post hoc and in closed form from action correspondences alone** (**frame-free, label-aligned**): frames whose action chunks are nearest neighbours are paired across domains, then an affine map is fitted. This works because imitation-trained encoders are organised by action: in discrete control this has to be induced (SupCon → action collapse, Mario); in continuous control standard Diffusion Policy training already produces it (regression collapse). Pushing collapse further with SupCon costs control (measured tradeoff). Stitching reuses **competent** modules: encoders trained on too few demos are unstructured and can't be stitched. With few target demos, the best adaptation is to keep the modules frozen and fine-tune only the map.

Scope of the variations between the two agents:

- **Visual variation:** textures, colours, lighting, camera pose, sensors.
- **Task variation:** actuation/dynamics, goals, rewards, embodiment/joints.

Where action labels do not share a meaning across domains, correspondences come from latent dynamics consistency, frozen foundation-model features, or K demos (few-shot).

## Regimes

| Regime | What the two domains share | Correspondence source | Status |
|---|---|---|---|
| Visual shift, same task | Action semantics | Nearest action-chunk pairs, affine map (`nn_affine`) | 77% ± 10% of the affine paired ceiling, 4 shifts × 3 seed pairs |
| Goal / reward shift (main task-shift result) | Reach/grasp behaviour; dynamics | Nearest action-chunk pairs (all frames or pre-grasp only) | nn_affine 69% ± 5% of the ceiling (3 seed pairs); map-only fine-tuning: 60% / 74% of the reference with 10 / 25 demos (scratch 0–16%) |
| Embodiment shift (Panda ↔ xArm, shared EE-delta actions) | Action semantics (same EE control) | Nearest action-chunk pairs | ceiling 56% of the xArm oracle; nn_affine / nn_orth 77% / 82% of the ceiling, no consistent winner (3 seed pairs) |
| Actuation / dynamics shift (EE-delta bounds ×0.5) | The robot's motion, not the commands | Planned: motion-chunk pairs (change in TCP position from proprio); fallbacks: latent dynamics consistency, foundation-mined anchors, K demos | Step 5b |

Note: a friction/mass variant was dropped. It did not change the expert's actions, because the motion planner is open-loop and ignores physics, so from the controller's point of view there was no task shift.

## Method sketch

**Setting.** We have an encoder E_u trained on domain u and a controller C_v trained on domain v. The goal is a map T such that C_v(T(E_u(o))) acts well.

0. **Encoders.** No special training in continuous control: plain imitation-trained encoders (Diffusion Policy) are already compressed onto a few action-relevant directions. In discrete control (Mario), SupCon on actions is needed to induce this structure. SupCon on chunk clusters in ManiSkill was tested and closed (it erases the cube position: collapse-vs-control tradeoff).
1. **Correspondences.** Pair each source frame with the target frame whose z-scored 8-step action chunk is nearest (label fits on disjoint demo halves). Planned for actuation shift: pairs on the robot's actual motion instead of commanded actions.
2. **Map.** Affine least squares (`nn_affine`, the default); affine in the top-16 principal directions of z when only a few demos are available (`nn_affine-PCA16`). Orthogonal maps (SAPS, `nn_orth`) are systematically worse, especially across tasks and viewpoints.
3. **Few-shot.** Fine-tune only the map with BC through the frozen controller on N target demos, encoder and controller frozen (best arm in Step 1b; end-to-end fine-tuning destroys encoder structure). Early stopping on held-out demos.
4. **Stitchability score** (computed without rollouts). Candidates: held-out z residual (ranks map types correctly across domains), encoder structure (NC1, effective rank: separates stitchable from unstitchable encoders), cycle error, CKA, offline chunk agreement on vision-sensitive frames (overrates label maps). The GW objective is not a failure signal.

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

- **Main: ManiSkill3, PickCube** (confirmed in Step 1).
  - Visual axes: camera pose ×3, look (textures/colours) ×3, lighting ×2.
  - Task axes: actuation (EE-delta bounds ×0.5), goal variant. Goal marker: a "lollipop" (small sphere on a thin pole).
  - Embodiment: Panda vs xArm6 + Robotiq (a locked joint is not supported in this ManiSkill version).
  - Experts: motion-planning demos (identical to ManiSkill's official ones), position-only control (`pd_ee_delta_pos`).
  - Policies: ManiSkill's Diffusion Policy baseline, unmodified; z = its 256-d visual feature. Core setup keeps `goal_pos` in the state; the cube is only visible in the image (blind check ≈ 0.05).
- **Secondary: CARLA** (Bench2Drive subset, PDM-Lite expert).
  - Visual axes: weather, camera placement.
  - Task axes: target speed, vehicle dynamics, route.
- **Continuity:** Mario / CarRacing, one row each.

## Metrics

- Success rate.
- % of oracle recovered = stitched / oracle on the target combination.
- % of paired ceiling = stitched / affine map fitted on paired frames (the headline metric, per seed pair).
- Compute saved versus retraining.

## Risks and falsifiers

- **Collapse vs fine control** (observed and measured: SupCon on chunk clusters erases the cube position; plain DP avoids it). Collapse is what makes label-only alignment work, but it discards within-action information that continuous control needs. If no labelling/objective keeps both alignability and native performance in ManiSkill, the method is limited to coarse/discrete control. Even if within-action information is kept, label anchors do not align it; stitched performance can lag native performance for this reason.
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

## Application (if Step 6b holds)

- **Privileged-to-deployable transfer, as a sim-to-sim proxy for sim-to-real:** reuse a controller trained with privileged information (entering through its encoder), with an encoder that must read that information from pixels in a harder domain. Measured as a data-efficiency curve (success vs number of target demos) against training from scratch and fine-tuning. No real robot yet, so never call it sim-to-real.
- **Redesign after Step 1b:** an encoder trained from scratch on N target demos is unstructured and can't be stitched, so the target encoder must come from an existing competent (or pretrained) encoder, adapted with map-only fine-tuning. Depends on goal-from-pixels working; the most at-risk step.

## Out of scope for now

- Offline-RL trajectory stitching: follow-up paper.
- Online RL: appendix at most.
- Swapping controllers mid-episode: a single demo experiment.
- CARLA: moved to the NeurIPS/CoRL version.
- Genre-level transfer across games (e.g. platformers, arcade driving): new encoder per game + reused controller via alignment + short fine-tune. Separate follow-up paper, after ICML.

## Open questions

1. Does prototype alignment work post hoc on plain BC encoders, or is SupCon training required? *(Answered 2026-09-29 in Mario: SupCon on the latent is required; see Decisions.)*
2. How many samples per action class are needed? *(Partly answered in Mario: 5 per class gives 73% of native in-game vs 88% with all; ≤100 random pairs per action is the adopted recipe.)*
3. Does frozen DINO + adapter close the gap on its own? *(Mario: no, but upscaled NES frames are a weak test; redo in ManiSkill, Step 3.)* Also: does a pretrained encoder fine-tuned on a few demos provide the structure that 5–25 demos alone can't? *(Answered 2026-10-07 in ManiSkill: no; off-the-shelf DINOv2 ViT-S stays below the oracle bar in our DP setup, so the few-demo question is not testable here. See Decisions.)*
4. Are foundation-mined anchors reliable under task shift, or do we need few-shot? *(Partly superseded: goal shift and embodiment work with action-chunk pairs; actuation will test motion pairs first.)*
5. Is an orthogonal map enough, or do we need affine/MLP? *(Answered 2026-10-04: affine. The affine paired ceiling reaches ~89% of native vs ~58% for orthogonal SAPS; an MLP adds nothing over affine.)*
6. **How should continuous actions be labelled so that encoders stay alignable without losing within-action information?** (Step 1b.) *(Answered 2026-10-04 in ManiSkill: no encoder-side labels; plain DP encoders + label-only maps on continuous action pairs. See Decisions.)*
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
- 2026-10-04 (Step 1b, ManiSkill, closed): **default aligner nn_affine** (each source frame paired with the target frame whose z-scored 8-step action chunk is nearest, source demos 0–49 vs target demos 50–99; affine least squares): 77% ± 10% of the affine paired ceiling over 4 shifts (look 1, look 2, look 2 + light 1, cam3) × 3 seed pairs × 2 directions (SAPS 64%, nn_orth 61%, K=16 action pairs 58%, identity 10%). **nn_affine-PCA16** (affine map in the top-16 PCA subspace) for N ≤ 10 demos, where plain affine is underdetermined. **The paired ceiling is affine** (it recovers 80–104% of native; orthogonal SAPS only ~60%).
- 2026-10-04 (Step 1b): **SupCon on action-chunk clusters is a measured tradeoff, branch closed**: as λ grows (0, .01, .1, 1), relative alignability rises (SAPS 65% → 96% of the ceiling, nn_affine 67% → 82%, λ 0 → .01) while native control falls (cam0 .79 → .30 → .05 → .05): collapse erases the cube position from z, and at λ ≥ .1 the agents fall back on goal_pos. The main line uses plain-DP encoders.
- 2026-10-04 (Step 1b): **label alignment needs competent, structured encoders.** Encoders trained on 5–25 demos are unstructured (NC1 ≈ 9) and cannot be stitched (stitched success at blind level). Stitching reuses competent modules; it does not rescue a domain from scratch.
- 2026-10-04 (Step 1b): **low-data adaptation: freeze the competent modules, fine-tune only the map.** On the goal-shift combination, map-only fine-tuning from a stitched start reaches 66% / 77% of the reference oracle with 10 / 25 demos (from scratch: .01 / .08); end-to-end fine-tuning destroys encoder structure (NC1 ~1 → ~10). One seed so far.
- 2026-10-04 (Step 1b): **first visual × task combinations work** (one seed): goal shift (look 2 default encoder → cam0 goal-variant controller; affine ceiling 81% of the reference, label-only 60–64% of the ceiling; orthogonal maps fail there, ~30%) and embodiment Panda → xArm (label-only nn_affine 83% of the ceiling with a 100k-iteration xArm controller; the ceiling is 60% of the xArm reference).
- 2026-10-04 (protocol): future fine-tuning runs choose early stopping on held-out demos (validation action loss), never on evaluation success.
- 2026-10-04 (Step 1b): **large seed variance across oracles** (e.g. cam0 .79 / .43 / .49 over seeds 1–3): report % of the affine ceiling per seed pair, never raw success alone. Oracle acceptance = success + three checks (blind, fake goal, cube/goal probe). Strong viewpoint changes (cam1, cam2) and a low-contrast light (light 1) give no working DP oracle: a stated limitation.
- 2026-10-06 (Step R, 3 seed pairs, closed): the one-seed combinations and low-data adaptation replicate. Goal shift: nn_affine 69% ± 5% of the affine ceiling (orthogonal maps 45–51%; identity 7%: the same-seed pilot's 15% was inflated by shared initialisation). Embodiment Panda → xArm: ceiling 56% of the xArm oracle; nn_affine and nn_orth both work with no consistent winner (77% vs 82% of the ceiling at 1000 episodes; the 250-episode "flip" was one pair plus eval noise). Map-only fine-tuning from a stitched start: 60% / 74% of the target oracle with 10 / 25 demos vs 0–16% for DP from scratch, goal shift and same task alike.
- 2026-10-06: **nn_affine stays the default map; no label-only map-class selection.** Raw, variance-matched and cosine criteria on held-out label pairs all pick nn_affine on 29–30 of 30 stitches (always-affine 76% of the ceiling vs 80% for a perfect per-stitch pick). The "noisy pairs make affine shrink" explanation is refuted (nn_affine keeps the same variance share in every shift).
- 2026-10-06: **Diffusion Policy's denoising loss on held-out demos is not a proxy for closed-loop success** (it rises while success rises; sampled-chunk error is a weak selector too). Everything is judged in closed loop.
- 2026-10-06 (protocol): **fine-tuning uses a fixed iteration budget chosen in advance** (5k for map-only; the final checkpoint for DP from scratch); early-stopped checkpoints are secondary numbers only; every low-data arm gets N training demos + the same 10 validation demos. Oracle acceptance = last-3 success ≥ 0.3 plus the three checks (hypothesis levels are not gates); a failed gate blocks only dependent jobs.
- 2026-10-07 (Step F, ablation, one seed pair): **with 10–25 demos, fine-tune only the map.** From the stitched start, adding the encoder adds variance and no gain (clearly worse once); adding the controller's last layers (last U-Net up block + output conv) is equivalent within noise. Holds for a visual change and a task change.
- 2026-10-07 (Step 3, closed, seed-1 pilot + diagnosis): **pretrained encoders do not replace task-trained ones here.** Baseline row: "Off-the-shelf DINOv2 ViT-S in ManiSkill's Diffusion Policy (frozen or fine-tuned, 126 or 224 px) stays below the oracle bar on PickCube in our setup, although its features locate the cube better than the oracle's task-trained encoder; the policy built on them is imprecise (2–4 cm from the cube vs 1.3 cm). Consistent with reports that frozen pretrained features underperform in Diffusion Policy (e.g. DINOv3-DP on PushT: 0.39 frozen vs 0.84 fine-tuned). Cause of the imprecision not diagnosed (time-boxed)." **The pretrained low-data route (a pretrained encoder fine-tuned on 5–25 demos, then stitched) is not testable in this setup**, since its 100-demo version already fails the oracle bar.
