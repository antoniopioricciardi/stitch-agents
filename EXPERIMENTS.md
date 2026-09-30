# EXPERIMENTS.md — lab log

Newest entries go on top, one entry per run or group of runs. Write the hypothesis **before** running.

## Key finding so far

**Setting.** To play an environment we use *its own* encoder (the one that can read its images) and a
controller from *another* environment, through a map fitted with SAPS. SAPS needs anchors; without paired
frames, we pair random frames that share the same action. This only works if the latent space is organised
by action, which is what SCIL provides.

**SCIL does not make agents robust to visual shift; it makes them alignable without paired data.**
Run unchanged in a new visual domain, SCIL agents fail just like BC agents. But a map fitted only on
the 8 action-class centroids (no paired frames) stitches SCIL encoders as well as SAPS with paired
frames does. For BC encoders, the same unpaired map is clearly worse. Confirmed in-game (Step 0b):
SCIL + prototypes reaches 88% of native distance, the same as SCIL + SAPS, against 46% for BC + prototypes.
Across levels (1-1 ↔ 1-2, no paired frames exist), SCIL + prototypes lets the other level's controller play
(Nature CNN: 74% of native on 1-1, flags included) where BC + prototypes reaches 28–39%.

---

## 2026-09-30 — Step 2b: switch oracles to ManiSkill's Diffusion Policy baseline

- **Why:** plain BC on motion-planning demos fails even from the true state (Step 2 below): the planner's timed rest
  pauses make the next action ambiguous from the current state, and single-step MSE BC averages over it.
  Action chunking with a multimodal (diffusion) head is the standard answer, and ManiSkill publishes a baseline for
  exactly this task. Our contribution is stitching, so oracles come from a published recipe, modified minimally.
  The rest-step and yaw-representation fixes are parked.
- **What exists:** no pretrained PickCube RGB policy from ManiSkill or LeRobot (the demo download ships state-based
  PPO checkpoints only). Reference number: ManiSkill's own wandb run of this baseline (stonet2000/ManiSkill,
  `diffusion_policy-PickCube-v1-rgb-100_motionplanning_demos-1`, seed 1): success_once 0.81, success_at_end 0.67
  at 30k iterations (0.75–0.82 from 15k on); the docs report success_once.
- **Setup:** `third_party/maniskill_diffusion_policy/` = ManiSkill v3.0.1 `examples/baselines/diffusion_policy`,
  unmodified. Their command (`baselines.sh`, RGB): 100 motion-planning demos replayed to `pd_ee_delta_pos` + RGB,
  obs horizon 2, action chunk 16 (8 executed), 30k iterations, batch 256, 100-step episodes, 100 eval episodes.
  Env changes to run it: `uv add diffusers tensorboard wandb` (wandb offline) and gymnasium pinned to 0.29.1 (the
  code reads `final_info`, removed in gymnasium 1.x). Glue: `scripts/run_dp_reference.sh`,
  `scripts/export_dp_demos.py` (our demos → ManiSkill trajectory format via `RecordEpisode`; 498/498 replay),
  `scripts/run_dp_ours.sh` (our env through `--env-id stitch.envs:StitchPickCube-v1`).
- **Hypotheses:**
  - Reference (their env, their demos, seed 1): success_once ≈ 0.81 (within ~±0.05).
  - Ours (our env: default task, Panda, cam0; our first 100 demos; same seed and settings): similar, 0.7–0.85.
    Differences: our goal sphere is visible to the camera (theirs is hidden), our demos come from our own planner
    run. Their recipe feeds `agent` + `extra` as state, which includes `goal_pos` and `is_grasped`; kept as-is here.
- **Result** (seed 1, 100 eval episodes per point, ~30 min per run on the RTX 5070 Ti;
  `results/20260930_dp_reference_pickcube_rgb/`, `results/20260930_dp_ours_default_panda_cam0/`):

  | iteration | ManiSkill's run (wandb) | reference reproduced | ours (StitchPickCube, cam0) |
  |---|---|---|---|
  | 5k | 0.05 / 0.02 | 0.05 / 0.02 | 0.04 / 0.01 |
  | 10k | 0.50 / 0.36 | 0.49 / 0.35 | 0.33 / 0.25 |
  | 15k | 0.82 / 0.67 | 0.77 / 0.64 | 0.46 / 0.35 |
  | 20k | 0.80 / 0.68 | 0.74 / 0.64 | 0.59 / 0.45 |
  | 25k | 0.75 / 0.64 | 0.68 / 0.51 | 0.54 / 0.46 |
  | 30k (final) | **0.81 / 0.67** | **0.77 / 0.64** | **0.51 / 0.39** |

  success_once / success_at_end. Demo lengths match (first 100 demos: mean 77.2 theirs, 77.4 ours; max 99 both).
- **Takeaway:**
  - **The baseline reproduces:** 0.77 vs 0.81 final, and the curve tracks theirs at every checkpoint (within ~0.07,
    i.e. about the noise of 100 episodes and one seed).
  - **On our env it works but is ~0.25 lower** (0.51 final, best 0.59 at 20k), with the same recipe, seed and
    demo lengths. One seed only; cause not yet identified. Differences to check: the goal sphere visible in the
    image (hidden in theirs), our scene rebuilt on every eval reset (`reconfiguration_freq=1`), our demo set.

---

## 2026-09-29/30 — Step 2 (minimal slice): BC oracles, default task, Panda, cam0 and cam1

- **Setup:** `stitch/models.py`: CNN encoder (4 stride-2 convs + linear, 128×128 RGB → d = 256) and MLP controller
  on [z, proprio (25-d: qpos, qvel, TCP pose; no goal)] → 7-d EE delta action. BC (MSE), 500 demos per camera,
  20k steps, batch 256, Adam 3e-4, random-shift augmentation; seeds 0, 1, 2. Evaluation: 100 episodes, seeds
  10000+ (disjoint from the demos), max 120 steps, success = PickCube success at any step. Blind checks on the same
  agents: z = 0 (zeros) and z = latent of a random training frame, redrawn every step (shuffled).
- **Hypotheses:**
  - Oracle (full z): 60–85% on both cameras; cam1 (side view, more robot occlusion) a bit lower than cam0.
  - Blind (zeros and shuffled): ≤ 5%. Proprio has no cube or goal position, and both are randomised (±10 cm cube,
    ±10 cm × 0–30 cm goal), so the controller cannot solve the task without the image.
  - ~5 min training per agent on the RTX 5070 Ti.
- **Result 1, first attempt — negative** (single-step RGB BC, 6-d rotation control; `results/20260929_step2_bc_cam{0,1}/`):
  0/100 for every seed and camera, with full z, zeros and shuffled alike (one 1/100 in shuffled). Training ~7.8 min
  per agent (two runs sharing the GPU). The policies move but never grasp (closest approach 3–16 cm).
- **Result 2, sanity check** (is it a pipeline bug?). Pipeline: replaying stored actions in the evaluation env
  succeeds 5/5; proprio matches to 1e-4; frames match except 1–2 edge pixels. Then BC from the **true state**
  (z = cube pose + goal position, same controller; `results/20260930_step2_state_bc_*/`, seed 0, 160-step limit):

  | z | control | recipe | eval seeds: success any step / at end | training seeds 0–9: any step |
  |---|---|---|---|---|
  | pixels (Result 1) | pose (7-d) | single step | 0.00 / — | 1/10 cam0, 2/10 cam1 |
  | state, cube quaternion | pose (7-d) | single step | 0.00 / 0.00 | 1/10 |
  | state, cube quaternion | pose (7-d) | chunk | 0.05 / 0.03 | 2/10 |
  | state, cube yaw as sin/cos(4·yaw) | pose (7-d) | single step | 0.00 / 0.00 | 1/10 |
  | state, cube yaw as sin/cos(4·yaw) | pose (7-d) | chunk | **0.45 / 0.26** | 5/10 |
  | state, cube quaternion | pos (4-d) | single step | 0.00 / 0.00 | 1/10 |
  | state, cube quaternion | pos (4-d) | chunk | 0.02 / 0.02 | 3/10 |

  single step = one action per step; chunk = 2-step history, 16-step chunk, 8 executed. Both standardise proprio
  and actions per dimension. The sin/cos(4·yaw) rows are a one-off diagnostic (not a committed script).
- **Why it fails:** the planner stops at rest at the end of every segment. Before descending, the expert sits
  ~5 cm above the cube for 2–3 steps with near-zero actions, then accelerates. From the state alone that looks
  like "stay still": the state-BC policies (pos, chunk) stop 4–6 cm above the cube in 19/20 episodes and close
  the gripper there, at steps 24–37, where the expert would start descending.
- **Takeaway:**
  - Not a pipeline bug: the same controller fails from the true state. The failure comes from the demos
    (timed rest pauses between planner segments) combined with plain MSE BC.
  - Chunking helps only once the rotation target is continuous (5% → 45%). Position-only control alone does not help (2%).
  - Candidate fixes (to decide): drop the rest pauses from the demos (steps with near-zero arm action and an
    unchanged gripper); a diffusion/flow head on the chunk; a time/phase input. The pixel recipe waits until state BC works.

---

## 2026-09-29 — Step 1: ManiSkill3 PickCube variants + motion-planning demos

- **Env** (`stitch/envs.py`, ManiSkill 3.0.1): PickCube, Panda / xArm6. Visual: camera ×3, look (colour/texture) ×3,
  light ×2. Task: default / actuation (EE-delta action bounds ×0.5; replaced physics = cube friction 0.3 → 0.1,
  10× mass, see Result 2) / goal (goal y ∈ [0.15, 0.25], disjoint from default). Goal sphere visible to the camera. **Missing: locked-joint robot** (mplib's screw planner can't mask joints).
- **Frame check** (`results/20260929_step1_check_envs/`): all variants render; same seed → identical initial state
  across visual variants; a state restored in another visual variant renders the same poses (paired frames for SAPS).
  Goal region inside the frame for all 3 cameras. 1 CPU env, 128×128 RGB: ~650 steps/s step+render, ~900 frames/s render only.
- **Demos:** mplib motion planner (joint targets) → converted to EE delta pose (7-d, both robots) by ManiSkill's
  replay conversion. Stored for every demo: actions, full env states, proprio, planner phase, `steps_to_grasp`, seed.
  Pixels rendered from the converted states only for default/Panda (cam0, cam1, cam2); the rest on demand later.
- **Expectations (before running):**
  - Panda, default task, N = 10: planner success ~100% (20/20 in a probe), conversion success ≥ 90%,
    ~85–95 EE steps per demo, ~2–3 s per demo (0.8 s planning + conversion + rendering).
  - Physics vs default (same seeds): joint plans identical (the planner is open-loop in geometry), EE actions
    nearly identical in every phase. If so, `physics` is not a task shift and is replaced by `actuation`.
- **Result 1, demos** (Panda, cam0 + cam1 rendered from the same states, N = 10, seeds 0–9; `results/20260929_step1_demos_<task>_panda/`):

  | task | planner success | conversion success | EE steps, mean (range) | time per demo |
  |---|---|---|---|---|
  | default | 10/10 | 10/10 | 72.9 (49–91) | 0.53 s |
  | physics (dropped) | 10/10 | 10/10 | 72.8 (49–91) | 0.50 s |
  | actuation | 10/10 | 10/10 | 73.2 (49–94) | 0.51 s |

  Stored EE actions replayed open-loop from the stored first state: 10/10 default demos succeed (max state diff ≤ 0.02).
  Every step also stores the planner phase and `steps_to_grasp` (0 when the gripper has closed, negative after).
- **Result 2, does the task variant change the expert?** Same seeds, EE-delta actions per phase vs default
  (arm |a| ratio = mean |action| variant / default over the 6 arm dims; `results/20260929_step1_compare_<task>_panda/`):

  | phase (mean length) | physics: arm \|a\| ratio, max same-seed \|Δa\| | actuation: arm \|a\| ratio, max same-seed \|Δa\| |
  |---|---|---|
  | approach (23.6) | 1.00, 0.000 | 1.39, 0.84 |
  | descend (14.1) | 1.00, 0.000 | 1.91, 0.21 |
  | grasp (6.0) | —, 0.014 | —, 0.03 |
  | carry = lift + move to goal, one screw motion (29.2) | 1.27, 0.075 | 1.91, 0.40 |

  Grasp start step: identical to default for physics in 10/10 seeds, for actuation in 9/10 (one demo +3 steps from clipping).
- **Result 3, full collection** (500 attempts per combination, seeds 0–499, 6 runs in parallel on 6 CPU cores;
  `results/20260929_step1_demos_<task>_<robot>/`):

  | task | robot | planner success | conversion success | demos saved | EE steps (mean) | time per demo |
  |---|---|---|---|---|---|---|
  | default | Panda | 99.6% | 100% | 498 | 78.3 | 0.65 s (3 cameras rendered) |
  | actuation | Panda | 99.6% | 100% | 498 | 79.0 | 0.33 s |
  | goal | Panda | 99.2% | 100% | 496 | 80.8 | 0.34 s |
  | default | xArm6 | 99.2% | 100% | 496 | 80.0 | 1.37 s |
  | actuation | xArm6 | 99.2% | 100% | 496 | 80.5 | 1.38 s |
  | goal | xArm6 | 99.4% | 100% | 497 | 83.7 | 1.39 s |

- **Takeaway (Step 1 closed, 2026-09-30):**
  - Motion-planning demos are cheap and reliable: ≥99% planner success and 100% conversion on every combination,
    0.3–1.4 s per demo. The xArm6 planner (RRT*, not seeded by ManiSkill) is not exactly reproducible from the seed.
  - **Physics (friction 0.1, 10× mass) is not a task shift for this expert**: identical actions until the grasp,
    ≤0.075 difference while carrying. Replaced by **actuation** (EE-delta bounds ×0.5), which roughly doubles the
    arm actions for the same motion. Physics could only come back with an expert that reacts to it.
  - Weak axis: camera 2 is close to camera 0 (kept on purpose as the mild shift).

---

## 2026-09-29 — Step 0c: does a TACO temporal loss keep or break label-only alignability?

- **Hypotheses:**
  - **H1:** a TACO-only encoder (temporal InfoNCE between [latent of frame t + embedded actions t..t+K−1] and the latent of frame t+K; K ∈ {1, 3}; controller trained with BC) does not collapse, and aligns from labels about as badly as BC.
  - **H2:** SCIL + TACO (SupCon directly on the latent, TACO on its own small projection heads) keeps SCIL's collapse and label-only alignability, with equal or better native play.
- **Setup:** Nature CNN, 1-1 v0/v1/v2 and 1-2 v0, 3 seeds, 50 epochs, same recipe as BC/SCIL (whose agents are reused). With skip 4, "t+K" is K decisions later (4K NES frames). TACO: action embedding (8 → 32) per step, concatenated over K steps; query head MLP([z_t, actions]) → 128, key head MLP(z_{t+K}) → 128, cosine InfoNCE (τ = 0.1), in-batch negatives, weight 1. The TACO term uses its own batch of 1024 transitions per step (from within-episode pairs), while cross-entropy / SupCon keep batch 256 and the same number of steps as the existing agents. Pilot first (SCIL+TACO, K = 3, v0, seed 0) to check the losses don't conflict.
- **Result:**
- **Takeaway:**

---

## 2026-09-29 — Why it works: neural collapse, projection head, stitchability pilot

- **Hypotheses:**
  1. *Collapse.* SCIL action centroids are close to a regular simplex (equal norms, pairwise cosines ≈ −1/7), and two independently trained SCIL encoders share the same centroid geometry up to rotation. BC centroids are not, and do not.
  2. *Projection head.* With SupCon on a projection head (`scilproj`), the latent `e` collapses less (higher effective rank). If alignability then drops towards BC, collapse is the mechanism; if it survives, the head is the recipe to carry into continuous control.
  3. *Stitchability.* Offline agreement correlates with in-game distance across all stitches (Spearman ρ > 0.6), with outliers such as DINOv2 on 1-2.
- **Setup:** 50-epoch Nature CNN agents (plus ResNet18 / DINOv2 for 1). `scilproj`: SupCon on MLP 512→512→128 on top of `e`, controller on `e`, everything else as SCIL; trained on 1-1 v0/v1/v2 and 1-2 v0. Alignment with the adopted recipe (random same-action pairs). Pilot: all existing stitches with both offline agreement and in-game max x.
- **Result 1, collapse** (training frames, mean over 4 domains × 3 seeds; `results/20260929_collapse_checks/`):

  | | within-class spread / separation (NC1; 0 = collapsed) | effective rank | variance in centroid span | simplex error (0 = regular) | geometry diff. between encoders |
  |---|---|---|---|---|---|
  | Nature BC | 4.97 | 43.8 | 23% | 0.238 | 0.125 |
  | **Nature SCIL** | **0.47** | **6.7** | **71%** | 0.233 | 0.114 |
  | Nature scilproj | 5.07 | 52.9 | 20% | 0.202 | 0.109 |
  | ResNet18 BC / SCIL | 2.83 / 0.93 | 14.1 / 13.6 | 42% / 55% | 0.346 / 0.323 | 0.163 / 0.139 |
  | DINOv2 BC / SCIL | 4.48 / 1.30 | 19.5 / 15.9 | 33% / 49% | 0.318 / 0.325 | 0.150 / 0.125 |

- **Result 2, projection head** (native play is unaffected: scilproj 1438–1676 on 1-1, 1077 on 1-2):

  | random same-action pairs | BC | scilproj | SCIL |
  |---|---|---|---|
  | offline agreement, versions / levels | .630 / .682 | .652 / .715 | **.754 / .844** |
  | in-game, v1 pairs | 703 (49%) | 814 (54%) | **1366 (91%)** |
  | in-game, play 1-1 / 1-2 with the other level's controller | 486 (31%) / 491 (45%) | 889 (65%) / 566 (52%) | **1299 (77%) / 698 (121%)** |

- **Result 3, stitchability pilot** (Spearman ρ between offline agreement and in-game % of native; `results/20260929_stitchability_pilot/`):
  all 372 agents ρ = 0.38. Nature CNN ρ = 0.69 (version pairs 0.66, levels 0.48, unchanged agents 0.44).
  Frozen ResNet18 / DINOv2 ρ ≈ 0: their native agents are so weak that "% of native" is dominated by noise
  (values up to 216%).
- **Takeaway:**
  - **The mechanism is collapse of each action onto its centroid (NC1), not a regular simplex (NC2).** SCIL
    latents collapse 10× more than BC's into a ~7-dim action subspace, but their centroids are no closer to
    equal angles than BC's (likely minority collapse from imbalanced actions), and the geometry is not clearly
    shared across encoders. Matching action centroids works because, for SCIL, that subspace holds almost
    everything the controller reads.
  - **With a projection head the latent does not collapse, and label-only alignment falls back to about BC
    level** (54% vs 91% in-game). So collapse is the mechanism, and in Mario we are effectively aligning the
    controller's decision space. For continuous control this is the key design question (see Next).
  - **Offline agreement is a usable stitchability signal for well-trained agents** (ρ ≈ 0.5–0.7), not for weak
    ones. Paper-quality Mario rows need more episodes and confidence intervals.
- **Next:** action labelling for continuous control (Step 1): bins, action-chunk k-means, or contrastive
  regression (Rank-N-Contrast), judged by alignability *and* retained within-class information.

---

## 2026-09-29 — Anchor recipe: action centroids vs random same-action pairs

- **Hypothesis:** for SCIL encoders the two recipes perform the same, because SCIL latents use only ~7–8 effective directions, which is about what 8 centroids determine. For BC encoders (18–39 effective directions), random pairs may do better, since they constrain more directions, or worse, since pairs inside an action are arbitrary and add noise.
- **Setup:** same models and settings as Steps 0a/0b and cross-level. Aligners: centroids (8 action means); random same-action pairs as in the old repo (1000-frame pool per side, ≤100 pairs per action); random pairs from the full training set (≤100 per action); SAPS paired as the ceiling where it exists. 5 random draws per random recipe. Offline: all encoders. In-game: Nature CNN, v1 pairs and cross-level.
- **Result, offline** (agreement with the native agent of the played domain; `results/20260929_anchors_offline/`):

  | | centroids | random pairs, pool 1000 | random pairs, full set | SAPS paired |
  |---|---|---|---|---|
  | Nature BC, versions / levels | .651 / .718 | .641 / .689 | .630 / .682 | .718 / — |
  | **Nature SCIL, versions / levels** | .758 / .842 | .755 / .833 | .754 / .844 | .762 / — |
  | ResNet18 SCIL, versions / levels | .694 / .804 | .682 / .781 | .683 / .782 | .693 / — |
  | DINOv2 SCIL, versions / levels | .621 / .644 | .612 / .638 | .605 / .636 | .632 / — |

- **Result, in-game** (Nature CNN; mean max x, % of native; `results/20260929_anchors_play_*`):

  | | centroids | random pairs, pool 1000 | random pairs, full set | SAPS paired |
  |---|---|---|---|---|
  | BC, v1 pairs | 696 (46%) | 615 (42%) | 703 (49%) | 1064 (72%) |
  | **SCIL, v1 pairs** | 1332 (89%) | 1298 (85%) | **1366 (91%)** | 1315 (88%) |
  | BC, play 1-1 / 1-2 | 448 / 427 | 430 / 435 | 486 / 491 | — |
  | **SCIL, play 1-1 / 1-2** | 1233 (74%) / 610 (104%) | 1149 (69%) / 492 (87%) | **1299 (77%) / 698 (121%)** | — |

- **Takeaway:** random same-action pairs (the old-repo recipe) work as well as centroids, offline and in-game;
  all differences are within noise. **We adopt random same-action pairs from the full training set** as the
  method's anchors (≤100 per action). The 1000-frame pool is slightly worse, probably because it has fewer
  anchors for rare actions.
- **Note on dimension:** the map is 512×512, so the latent keeps the architecture's dimension (512) whatever
  the number of anchors. The anchor count only changes how many directions of the rotation are fitted to data.
  SCIL latents use ~7–8 effective directions (BC 18–39), so a few actions' worth of anchors already pin down
  what the controller reads.

---

## 2026-09-29 — Cross-level stitching: play 1-1 / 1-2 with the other level's controller

No hypothesis was written before this run (my omission).

- **Setup:** v0, 50 epochs, three encoder types. The Nature CNN is trained from scratch; ResNet18 and DINOv2
  are frozen pretrained backbones, each with a trainable fc layer (SupCon is applied on that layer's output).
  To play level a, we use encoder a + map + controller of level b (seed s+1). SAPS is impossible here, because
  the two levels share no paired frames. Map = prototypes: Procrustes on the action centroids of encoder a
  (level-a frames) and encoder b (level-b frames), with all samples or 5 per class. Random same-action anchor
  pairs, as in the notebook and the old repo, are the same map on average.
  In-game: 10 episodes per stitch, 3 seeds. Offline agreement: on held-out frames of level a, how often the
  stitched agent picks the same action as the native agent of level a.
- **Agents in their own level** (in-game mean max x / offline balanced acc; 1-2 is harder: balanced acc ≈ 0.25–0.28 for all):

  | | 1-1 BC | 1-1 SCIL | 1-2 BC | 1-2 SCIL |
  |---|---|---|---|---|
  | Nature CNN | 1579 / .41 | 1710 / .44 | 1075 / .25 | 654 / .25 |
  | ResNet18 (frozen) | 1055 / .50 | 885 / .53 | 682 / .28 | 503 / .28 |
  | DINOv2 (frozen) | 809 / .50 | 972 / .48 | 368 / .27 | 440 / .27 |

- **Result, in-game** (mean max x, % of the native agent of the played level; offline agreement in brackets):

  | encoder | played level | other level's agent, unchanged | BC + prototypes | **SCIL + prototypes** |
  |---|---|---|---|---|
  | Nature CNN | 1-1 | BC 558 / SCIL 495 | 448, 28% (.69) | **1233, 74% (.85), 13% flags** |
  | Nature CNN | 1-2 | BC 180 / SCIL 181 | 427, 39% (.72) | **610, 104% (.84)** |
  | ResNet18 | 1-1 | BC 340 / SCIL 247 | 307, 31% (.57) | **716, 82% (.81)** |
  | ResNet18 | 1-2 | BC 230 / SCIL 281 | 45, 7% (.51) | **345, 60% (.80)** |
  | DINOv2 | 1-1 | BC 595 / SCIL 742 | 135, 19% (.50) | **775, 82% (.70)** |
  | DINOv2 | 1-2 | BC 452 / SCIL 300 | 40, 11% (.46) | 62, 13% (.59) |

  With 5 samples per class, SCIL + prototypes still reaches 56% (Nature, 1-1), 121% (Nature, 1-2), 64% / 87% (ResNet) and 48% / 74% (DINOv2).
- **Takeaway:** across levels, where paired frames do not exist, **SCIL + prototypes lets the other level's
  controller play**. With the Nature CNN it reaches 74% of native on 1-1, flags included; BC + prototypes stays
  at 28–39%. Offline agreement is 0.80–0.85 for SCIL and 0.51–0.72 for BC, on every encoder except DINOv2 on
  1-2 (0.59).
- **Caveats:** the frozen-backbone agents are weak players even natively (x 370–1050), so their in-game
  percentages are noisy (e.g. >100%). With a frozen backbone, "unchanged" and "no map" agents keep some skill,
  because both levels' encoders share the same pretrained features. DINOv2 + SCIL on 1-2 fails in-game
  (x 62) despite 0.59 agreement.
- Results: `results/20260929_stitch_levels_{nature,resnet18,dinov2}/`.

---

## 2026-09-29 — Frozen ResNet18 / DINOv2 encoders on 1-1 (v0 / v1 / v2)

No hypothesis was written before this run (my omission).

- **Setup:** frozen ImageNet ResNet18 or DINOv2 ViT-S/14 (224×224, same 2-frame merge) + trainable fc layer
  (`e`, SupCon here) + linear controller. Otherwise the Step 0 recipe, 50 epochs. Same no-alignment baseline and
  offline stitching as for the Nature CNN.
- **Result:**

  | | native max x (mean v0–v2) | unchanged, pairs with v1 | unchanged, v0 ↔ v2 | offline: prototypes as % of SAPS (all / 5 per class) |
  |---|---|---|---|---|
  | ResNet18 BC | 955 | 35% | 58% | 82% / 66% |
  | ResNet18 SCIL | 880 | 50% | 54% | **100% / 91%** |
  | DINOv2 BC | 862 | 43% | 30% | 85% / 64% |
  | DINOv2 SCIL | 860 | 37% | 25% | **99% / 88%** |
  | *(Nature CNN BC / SCIL, for reference)* | *1514 / 1561* | *32% / 35%* | *60% / 66%* | *90% / 68%, 99% / 97%* |

- **Takeaway:** frozen pretrained features do **not** remove the visual-shift problem. Unchanged agents still
  drop to 25–58% of native, and the frozen agents play much worse than the Nature CNN (x ≈ 860–955 vs ≈ 1500).
  The stitching result holds for all three encoders: SCIL + prototypes ≈ SCIL + SAPS, and BC + prototypes is
  clearly below.
- Results: `results/20260929_oracles_{resnet18,dinov2}_1-1_v*/`, `20260929_shift_nostitch_*`, `20260929_stitch_offline_*`.

---

## 2026-09-29 — Step 0b: stitched agents playing (in-game)

- **Hypothesis:** on the pairs involving v1, stitched agents clearly beat agents run unchanged. SCIL + prototypes (no paired frames) plays about as well as SCIL + SAPS. BC + prototypes is worse, most of all with 5 samples per class.
- **Setup:** 50-epoch oracles. Pairs v0→v1, v1→v0, v1→v2, v2→v1 (encoder / played version → controller version); encoder seed s, controller seed s+1. Maps: SAPS (paired), prototypes (all / 5 per class), fitted as in Step 0a. 10 episodes per stitch; compared with the unchanged agents of the no-alignment baseline.
- **Result:** (in-game, mean over the 4 pairs × 3 seeds; % of native = vs an agent trained in the played version; `results/20260929_step0b_stitch_play/`)

  | agent playing | mean max x | % of native | flags |
  |---|---|---|---|
  | BC, run unchanged | 486 | 32% | 0% |
  | SCIL, run unchanged | 548 | 35% | 0% |
  | BC + SAPS (paired) | 1064 | 72% | 2% |
  | BC + prototypes (all / 5 per class) | 696 / 551 | 46% / 37% | 1% / 0% |
  | SCIL + SAPS (paired) | 1315 | 87% | 7% |
  | **SCIL + prototypes (all / 5 per class)** | **1332 / 1104** | **88% / 73%** | 6% / 3% |
- **Takeaway:** confirmed in play. SCIL + prototypes (no paired frames) plays as well as SCIL + SAPS (paired), and much better than BC + prototypes and than unchanged agents.

---

## 2026-09-29 — Step 0: do the oracles improve with longer training?

- **Hypothesis:** the 10-epoch oracles are undertrained (train loss still falling). 30–50 epochs give clearly better in-game play (mean max x well above ~1350) without hurting held-out balanced accuracy.
- **Setup:** same recipe as Step 0 below, v0 only, BC and SCIL, seeds 0–2, 30 and 50 epochs.
- **Result:** (v0, in-game mean max x / flags; offline balanced acc)

  | epochs | BC | SCIL |
  |---|---|---|
  | 10 (reference) | 1321 / 7% / 0.40 | 1294 / 3% / 0.41 |
  | 30 | 1534 / 10% / 0.39 | 1541 / 0% / 0.40 |
  | 50 | 1579 / 13% / 0.41 | **1710 / 17% / 0.44** |
- **Takeaway:** longer training helps moderately (+20–30% distance), most for SCIL. Single episodes still vary a lot. **From here on, oracles are trained for 50 epochs** (`results/20260929_step0_oracles_v*_e50/`).

---

## 2026-09-28 — Step 0: Mario 1-1, BC vs SCIL, with and without alignment

**Setup**
- **Data:** 12 human wins of level 1-1 (10 for training, 2 held out). Each is replayed in 3 ROM versions
  with identical physics: **v0** original, **v1** black background, **v2** flat graphics. So every frame
  exists in all 3 versions (paired data).
- **Agent:** the SCIL/Kanervisto Atari CNN. Encoder = convs + fc → 512-d latent; controller = one linear
  layer → 8 action classes. 3 seeds per setting.
- **BC** = cross-entropy only. **SCIL** = BC + SupCon on the 512-d latent (λ=1, τ=0.07).
- **Stitch** = encoder trained in version u + map + controller trained in version v (different seeds).
  - **SAPS**: Procrustes on *paired* frames (the same frames in u and v).
  - **Prototypes**: Procrustes on the 8 action-class centroids, computed from *different* episodes in u
    and v, so **no paired frames are used**.
**How to read the tables**
- **mean max x**: how far Mario gets in the level, averaged over 10 episodes (flag at x ≈ 3160).
- **flags**: % of episodes that finish the level.
- **% of native**: mean max x divided by that of an agent trained in the version being played.
- **balanced acc** (offline): on the 2 held-out human runs, how often the agent picks the human's
  action, averaged per action class so the frequent "right" does not dominate. Chance = 0.14. An agent
  trained in the right version scores ≈ 0.39, which is the practical ceiling.
- **agreement with target agent** (offline): on the same frames, how often the stitched agent picks the
  same action as the agent trained in the target version. 1.0 = it behaves exactly like that agent.
- **5 per class / all**: how many frames per action class were used to compute the centroids.

### 1. Agents in their own version (in-game)
Hypothesis: both get well past x ≈ 1000; SCIL ≥ BC.

| | mean max x | flags | offline balanced acc |
|---|---|---|---|
| BC | 1398 | 4% | 0.39 |
| SCIL | 1339 | 3% | 0.39 |

Averaged over v0/v1/v2. **Result:** both learn to play but are weak (~40% of the level). SCIL ≈ BC.
Training loss was still falling after the 10 reference epochs, so they are possibly undertrained.

### 2. Agents run unchanged in another version (in-game, no alignment)
Hypothesis: clear drop; v2 hurts more than v1.

| | pairs involving v1 | v0 ↔ v2 |
|---|---|---|
| BC | x 552 (39% of native) | x 1076 (80% of native) |
| SCIL | x 677 (52% of native) | x 1058 (76% of native) |

**Result:** v1 is the hard shift, and v2 is mild (the opposite of the hypothesis). No flags off-diagonal.
**SCIL is not more robust than BC.**

### 3. Stitching across versions (offline only, not yet tested in-game)
Hypothesis: SCIL + prototypes reaches ≥ 80% of SAPS; BC + prototypes clearly less.

| encoder + map | balanced acc | agreement with target agent |
|---|---|---|
| BC, no map | 0.16 | 0.33 |
| SCIL, no map | 0.15 | 0.37 |
| BC + SAPS (paired) | 0.36 | 0.73 |
| **SCIL + SAPS (paired)** | 0.39 | 0.78 |
| BC + prototypes (unpaired; 5 per class / all) | 0.31 / 0.37 | 0.56 / 0.69 |
| **SCIL + prototypes (unpaired; 5 per class / all)** | 0.38 / 0.40 | **0.76 / 0.78** |

Mean over 6 version pairs × 3 seeds. The native agent scores 0.39 balanced accuracy.

**Result:**
- Without a map, stitching fails for both BC and SCIL.
- With paired frames (SAPS), SCIL aligns somewhat better than BC.
- **Without paired frames (prototypes), SCIL matches SCIL + SAPS, even with only 5 samples per class.
  BC does not** (0.56 agreement at 5 per class).

**Why (latent check, v0):** in SCIL encoders, 67% of the latent variance lies in the 7-dim span of the
action centroids (BC: 40%). Projecting SCIL latents onto that span costs the controller nothing
(0.64 → 0.65 accuracy); for BC it costs 4 points. So 8 centroids pin down what a SCIL controller reads.

**Method notes**
- Stitches pair encoder seed s with controller seed s+1. Same-seed models share their initialisation and
  stay partly aligned even across versions, which made "no map" look falsely good in a first run.
- Plain accuracy is not reported: always predicting "right" already scores 0.53.

**Open:**
- Stitched agents have not been played in-game yet (Step 0b); the pairs involving v1 are the informative ones.
- Possible undertraining of the oracles.
- Harder shifts (v3, other levels).

Results: `results/20260928_step0_*` and `results/20260928_step0a_stitch_offline/`.

---

## YYYY-MM-DD — Step N: <short name>

- **Hypothesis:**
- **Setup:** env/axes, methods, seeds, episodes, git commit
- **Result:** numbers or a small table, plus a link to `results/<folder>`
- **Takeaway:** one line
- **Next:**
