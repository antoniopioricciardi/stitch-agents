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

## 2026-10-01 — Step 1b (rows 1–2): does action collapse make DP encoders alignable from labels? (branch `step1b-labels`)

- **Question:** does the Mario mechanism (SupCon on the latent → action collapse → label-only alignment ≈ paired SAPS)
  carry over to continuous control with Diffusion Policy?
- **Setup:**
  - Core DP recipe (Step 2 handoff), default task, Panda, lollipop, 100 demos, 50k iterations. Domains: cam0
    (`StitchPickCubeLollipopNoGrasp-v1`) and cam1 (`…NoGraspCam1-v1`, left side view; strong shift).
  - **Both cameras train on the same demos 0–99** (re-rendered: identical states and actions). A feasibility test,
    so the easiest conditions; the 3-seed expansion in Step 4 trains extra agents on different demos (e.g. 100–199)
    to test independent data. Held-out frames for offline metrics: demos 400–497.
  - z = `agent.visual_encoder` output (256-d per frame); the state (incl. goal_pos) goes to the denoiser unchanged;
    T maps z per frame (both obs-horizon frames).
  - **Labels:** frame t → k-means cluster (K = 16) of its executed chunk a[t:t+8] (DP's padding included), each
    action dimension z-scored over the training actions, then flattened to 32-d. Fitted once on the chunks of demos
    0–99; actions do not depend on the camera, so the labels are shared. Torch k-means (k-means++, Lloyd, restarts).
  - **Row 1 (reference):** plain DP. cam0 seed 1 = the Step 2g oracle; cam1 seed 2 new.
  - **Row 2 (SupCon):** DP loss + λ · SupCon(z_t, chunk label), λ = 1, τ = 0.07 (Mario's SCIL settings), on the
    current frame's z (L2-normalised). cam0 seed 1, cam1 seed 2.
  - New runs use `scripts/train_dp_supcon.py`: a copy of the vendored training loop (vendored file untouched) that adds
    the SupCon term and saves the 40k / 45k / final checkpoints; row 1 cam1 runs it with λ = 0.
  - **Stitches:** cam0 encoder (s1) → cam1 controller (s2), and cam1 encoder (s2) → cam0 controller (s1); different
    seeds, so no shared initialisation. Aligners: identity, SAPS (paired frames: same demo and step in both
    cameras), action_pairs (≤100 same-label pairs per cluster, 5 draws). Fit frames: demos 0–99 of both cameras
    (same labelled frames for every aligner).
  - **Metrics:** closed-loop success of the stitched agent in the encoder's domain (250 episodes, final checkpoint
    only; natives: final (last-3) from the training logs). Offline agreement on held-out frames: L2 distance between
    the stitched and native (same encoder's own agent) 8-step chunks, with the same DDPM noise; next to it the chance
    level (native on random other frames) and the noise floor (native vs native, different noise). On every agent:
    NC1, effective rank (participation ratio), and the regression-collapse check (Andriopoulos et al., NeurIPS 2024):
    share of z's variance in its top-4 and top-16 principal components.
  - **Order / stop rule:** row 1 is trained and reported first. If SAPS with paired frames fails on plain DP, stop:
    DP latents would not be linearly stitchable even with perfect correspondences, and labels cannot fix that.
- **Hypotheses:**
  - **Row 1:** SAPS works (a large share of native success); action_pairs well below SAPS (as BC in Mario).
  - **Row 2:** action_pairs close to SAPS; native success roughly unchanged vs row 1 on the same camera; NC1 and
    effective rank much lower than row 1.
  - **Pre-registered fallback:** if row 2's native success is more than ~0.1 below row 1 on the same camera, rerun
    row 2 with λ = 0.1.
- **Preparation** (`results/20261001_dp_ours_demos_default_panda_cam1_lollipop/`, `results/20261001_step1b_labels/`):
  - cam1 demos exported with `scripts/export_dp_demos.py lollipop 1 <demos.npz>`: 498/498 replay; actions identical
    to cam0's (checked), so the labels are shared by construction.
  - Labels (7618 chunks of demos 0–99): cluster sizes 88–1772 (imbalanced; the largest, 1772, is the near-still
    closed-gripper phase incl. end padding); 6 clusters mainly gripper-open, 10 mainly closed, one of them (500) the
    grasp transition (50% open steps).
  - Smoke test (cam1, seed 2): the copied loop at λ = 0 gives exactly the vendored `train_rgbd.py` loss at each of the
    first 30 iterations. With λ = 1 SupCon dominates the encoder gradient more and more: ‖∇_enc‖ SupCon / DP = 3× at
    iteration 0, 18–80× at 50–90, 100–250× at 100–190 (the DP gradient into the encoder falls 0.23 → 0.04, SupCon's
    grows 0.7 → 9); SupCon itself only starts to fall (5.54 = chance → 5.2 at 190).
- **Result 1, cam1 oracle (row 1, seed 2, λ = 0; `results/20261001_step1b_dp_ref_cam1_s2/`): fails.**
  success_once / success_at_end every 5k from 0: .004/.004, .064/.036, .024/.012, .052/.024, .048/.032, .056/.036,
  .056/.036, .040/.036, .048/.024, .044/.028, **final .048 (last-3 .047) / .032 (.028)**: blind level (Step 2g blind
  check ≈ .05), against cam0's .82 (.79) with the same recipe. Training loss is as low as cam0's (~.001–.002 at 26k).
  Not a pipeline bug: the cam1 demo frames are pixel-identical to the cam1 eval env's frames for the same seeds.
  Likely cause (from the frames): from the left side the cube is a few dark-red pixels on the orange wooden table,
  often lined up with the lollipop pole and the gripper, and its position along the viewing axis is hard to read.
  In cam2 and in cam0 with look 1 / look 2 the cube is as visible as in cam0.
- **Decision:** stopped per "oracles first"; no stitching run on cam1 (a cam1 controller and a cam1 native at blind
  level make every stitching number meaningless). Row 2 not trained. Next domain to choose (see the takeaway).
- **Takeaway:** cam1 is not a usable domain for the DP core recipe (100 demos, PlainConv); the second domain must be one
  whose oracle works. Oracle quality limits which domains can be stitched at all.
- **To-do (cam1, kept, nothing deleted):** cam1 oracle fails (0.05, blind level) because the cube is a few pixels and
  aligned with the pole/gripper; redesign the pose (closer/higher) later to keep a strong viewpoint shift with a
  working oracle.

**Update (2026-10-01): second domains = cam0 + look 1 (appearance shift; first), then cam2 (viewpoint shift).**
Same recipe as above (λ = 0 reference, seed 2, demos 0–99; env ids `…NoGraspLook1-v1`, `…NoGraspCam2-v1`; demos
exported with `export_dp_demos.py lollipop <cam> <demos.npz> <look>`). Row 1 runs on every second domain whose oracle
works, with cam0 seed 1 as the other side; reported before any row 2 agent is trained. Labels unchanged (actions are
identical in every render). λ plan unchanged.
- **Hypotheses:** look 1 oracle ≈ cam0 (0.7–0.85: same geometry, the yellow cube on grey is more visible than red on
  wood); cam2 oracle a bit lower (0.6–0.8: low front view, depth along the viewing axis harder).
- **Offline-metric fix (before any reported number):** a code test of `step1b_stitch.py --offline-only` on cam0 / cam1
  showed that on demo frames the state alone predicts most of the chunk: native vs native on a random other frame
  (state and z both changed) = 3.15, but identity stitch = 0.10–0.18 and SAPS 0.06–0.07 even with the failing cam1
  agent. So the chance level for z now keeps the state and takes z from a random other frame (`chance_z`: 0.25 / 0.35
  in that test); the frame-level one is kept as `chance_frame`. Offline agreement on demo frames is therefore a weak
  signal for these agents; closed loop is the real test.

**cam1 diagnostics (no training; hypotheses before running).** In two eval videos the gripper goes to the goal marker,
not the cube, and never grasps. Candidates: (A) the policy cannot find the cube in cam1 and falls back on goal_pos from
the state (the goal is usually near the cube in x/y); (B) it confuses the marker with the cube.
1. **Linear probe** from frozen z (each agent on its own camera's frames; current frames; fit on demos 0–99, test on
   400–497; ridge, regularisation by CV, latents centred only) to the cube position and the goal position, R² per
   coordinate. Also on the look 1 / cam2 oracles when they finish. Hypotheses: cam0 z → cube x/y R² ≥ 0.8; cam1 z →
   cube x/y R² clearly lower (≤ 0.5) under (A), high under (B). Goal from z: moderate in both (the marker is visible
   but goal_pos is in the state, so the encoder need not encode it).
2. **Fake goal:** cam1 agent, 50 episodes, goal_pos in the state replaced by the goal of another seed's reset
   (the env's own goal distribution); the real marker stays in the image. Under (A) the gripper follows the fake goal:
   its closest approach to the fake goal is much smaller than to the cube and to the real marker. Control: 50 episodes
   with the true goal (closest approach to cube vs goal).
3. **Policy frames:** the 128×128 cam1 frames of 3 failed true-goal episodes, as a grid.
- **Results** (`results/20261001_step1b_probe_cube_goal/`, `results/20261001_step1b_cam1_fake_goal/`):
  1. Probe, first pass (test R² on demos 400–497; cube x / y on frames with the cube on the table; α grid then
     1e-3…1e3, CV picked its lower edge, so the grid is now extended to 1e-6 and the final numbers come with the
     look 1 / cam2 run): cam0 z → cube 0.78 / 0.58, cam1 z → cube 0.67 / 0.38; goal from z ≈ 0 for both (≤ 0.03).
  2. Fake goal (50 episodes each; mean closest approach of the TCP, m):

     | cam1 agent | success | grasped | to cube | to real goal | to state goal_pos | closer to state goal than to real goal |
     |---|---|---|---|---|---|---|
     | true goal | .02 | .08 | .078 | .013 | (= real) | — |
     | fake goal (16 cm from the real one on average) | .00 | .08 | .082 | .078 | **.015** | **92%** |

  3. Frame grid: the cube is visible as a red dot in the 128×128 cam1 frames; the gripper moves past it to the marker.
- **Takeaway:** **(A) supported, (B) rejected.** The cam1 agent goes wherever goal_pos in the state says (1.5 cm from a
  fake goal, 8 cm from the real marker), so it ignores the marker in the image, and it ignores the cube. Yet cam1's z
  still encodes the cube's x/y moderately (R² 0.67 / 0.38 vs cam0's 0.78 / 0.58): the information is partly there,
  but the controller learned the goal_pos shortcut instead of using it. Not a visibility problem alone.

**cam0 fake-goal test (does the core setup have the shortcut everywhere?).** `scripts/fake_goal_check.py` (the cam1
script, generalised to any domain; adds placed_fake = the cube came within 0.025 m of the fake goal). Same 50 + 50
episodes. A working agent also ends near the fake goal (carrying the cube there is the task), so the decisive measures
are the grasp rate and placed_fake.
- **Hypothesis:** cam0 finds the cube from the image: grasp rate under the fake goal ≥ 0.8× the true-goal rate,
  closest approach to the cube ≲ 2 cm in both conditions, and the cube carried to the fake goal (placed_fake ≈ the
  true-goal success rate). goal_pos is then used only for placing, as intended.
- **Decision rule (agreed):** if so, the shortcut is viewpoint-specific; keep the core setup, and every oracle gets
  three checks from now on: blind (z zeroed / shuffled), fake goal, cube/goal probe. If cam0 also partly follows the
  fake goal (grasp rate clearly lower): stop and report; the likely fix is sampling the goal outside the cube's area
  (not implemented yet).
- **Offline metric, vision-sensitive version (added to `step1b_stitch.py` before any reported number):** sensitivity
  of a held-out frame = ‖native chunk − native chunk with z from a random other frame‖ (state kept); agreement also
  reported on the 30% most sensitive frames, next to the overall number.
- **Result** (`results/20261001_step1b_fake_goal_cam0_step2g_dp_core_nograsp_50k_s1/`; 50 episodes each):

  | cam0 agent | success (real goal) | grasped | placed at state goal_pos | closest to cube (mean / median) |
  |---|---|---|---|---|
  | true goal | .80 | .90 | .80 | 1.0 / 0.6 cm |
  | fake goal | .00 | .88 | **.78** | 1.2 / 0.6 cm |

  In both conditions 89% of grasped episodes end with the cube at goal_pos; cube approach < 2 cm in 84% of episodes in both.
- **Takeaway:** **hypothesis confirmed; the shortcut is viewpoint-specific.** cam0 finds the cube from the image
  (grasp rate and approach unchanged) and uses goal_pos only for placing, as intended. Per the decision rule: keep the
  core setup; every oracle now gets three checks: blind (`eval_dp_blind.py`), fake goal (`fake_goal_check.py`),
  cube/goal probe (`probe_cube_goal.py`). Oracle acceptance needs more than a success rate: cam1 would have passed a
  "goes near the goal" eyeball test.

**Second-domain oracles (seed 2, λ = 0; `results/20261001_step1b_dp_ref_{look1,cam2}_s2/`)** — success_once /
success_at_end every 5k from 0:
- **look 1:** .000/.000, .024/.020, .480/.340, .612/.456, .668/.460, .676/.464, .680/.532, .720/.532, .712/.528,
  .696/.564, **final .676 (last-3 .695) / .508 (.533)**. Works, ~0.1 below cam0 (0.82 / 0.79) and just under the
  0.7–0.85 hypothesis. Checks: blind full .72 / zeroed .02 / shuffled .04 (needs vision); fake goal (50 + 50
  episodes) grasped .94 → .88, cube approach 1.1 → 1.4 cm, placed at state goal .80 → .68 (finds the cube from the
  image; placing drops a bit more than cam0's); probe cube x / y .76 / .59 (≈ cam0), goal ≤ .04. **Accepted.**
- **cam2: fails.** .000, .040, .032, .032, .056, .048, .060, .068, .092, .084, **final .044** (training loss .0003:
  fits the demos). Unlike cam1 the cube is clearly visible in cam2, so visibility alone does not explain it.
  Checks: blind full .08 / zeroed .02 / shuffled .03; fake goal: **follows goal_pos like cam1** (closest approach
  1.1 cm to the fake goal vs 6.8 cm to the cube; grasped .10 → .14; placed at fake goal .04); probe cube x / y
  .61 / .79 (the y coordinate better than cam0's), goal ≈ 0. So z encodes the cube, yet the controller learned the
  goal_pos shortcut: the probe does not predict this failure. No stitching on cam2. **To-do:** as cam1 (understand
  why two of three viewpoints fall into the shortcut while cam0 / look 1 do not).
- Probe with the extended α grid (α now inside the grid; cube x / y on table): cam0 .78 / .58, cam1 .66 / .39,
  look 1 .76 / .59. Unchanged from the first pass.

**Row 1, cam0 ↔ look 1, offline** (`results/20261001_step1b_stitch_row1_look1/`; 2000 held-out frames; L2 distance
between 8-step chunks, all frames / the 30% most vision-sensitive frames; action_pairs = mean of 5 draws, spread ±.002):

| | cam0 enc → look 1 ctrl | look 1 enc → cam0 ctrl |
|---|---|---|
| noise floor | .024 / .047 | .030 / .033 |
| chance for z (shuffled z, state kept) | .255 / .786 | .182 / .551 |
| identity | .126 / .207 | .144 / .220 |
| SAPS (paired) | .046 / .093 | .046 / .070 |
| action_pairs | .055 / .112 | .056 / .094 |

z geometry (training frames, 16 chunk labels): cam0 NC1 .92, effective rank 3.1, variance in top-4 / top-16 PCs
.77 / .90; look 1 1.29, 4.2, .72 / .89 (Mario BC: NC1 4.97, effective rank 44). Offline, action_pairs is already close
to SAPS on plain DP, against the row 1 hypothesis; plain DP's z is already very low-dimensional (regression collapse
without SupCon).

**Row 1, cam0 ↔ look 1, closed loop** (250 episodes, final checkpoints, success_once / success_at_end; the stitched
agent plays in its encoder's domain; action_pairs = draw 0; natives from the training logs, final (last-3)):

| | cam0 enc → look 1 ctrl (plays cam0) | look 1 enc → cam0 ctrl (plays look 1) |
|---|---|---|
| native agent of that domain | .824 (.788) / .596 | .676 (.695) / .508 |
| identity | .068 / .052 | .048 / .036 |
| SAPS (paired) | **.452 / .376** (55% of native) | **.452 / .344** (67% of native) |
| action_pairs | **.396 / .296** (48% of native, **88% of SAPS**) | **.284 / .192** (42% of native, **63% of SAPS**) |

The cam0 → look 1 direction was measured twice (the first run was OOM-killed when it opened the second domain's
eval envs while the first's were still open; fixed by closing them): identity .028, SAPS .460, action_pairs .376;
within ±.04 of the run above. Getting there took three OOM kills of the checks service: 10 GB cap too small, then two
10-worker eval-env sets alive at once.
- **Takeaway (one seed, pilot):**
  - **Stop rule passed:** a linear map stitches plain DP latents with paired frames (SAPS ≈ .45 both ways, 55–67% of
    native), and no map is at blind level (.05–.07).
  - **Row 1 hypothesis:** label-only action_pairs is 88% of SAPS one way and 63% the other. The asymmetry is treated
    as pilot noise until more seeds; not interpreted.
  - Plain DP z is already strongly collapsed onto a few directions (effective rank 3–4), which may be why labels
    already align it partly. Row 2 (SupCon) tests whether collapse onto the chunk labels closes the gap to SAPS.

**Row 2 (SupCon λ = 1): cam0 seed 1, then look 1 seed 2** (`results/20261001_step1b_dp_supcon_{cam0_s1,look1_s2}/`;
`run_dp_supcon.sh`, same recipe and demos as row 1). Hypotheses as pre-registered at the top of this entry
(action_pairs close to SAPS; native roughly unchanged; NC1 / effective rank much lower than row 1).
- **Fallback rule, made precise before the results:** compare the last-3 mean success_once with row 1 on the same
  domain (cam0 .788, look 1 .695); if either drops by more than .1, rerun row 2 with λ = 0.1.
- **cam0 seed 1, λ = 1 (`results/20261001_step1b_dp_supcon_cam0_s1/`): native collapses.** success_once /
  success_at_end every 5k from 0: .008/.004, .048/.028, .036/.024, .092/.076, .068/.064, .076/.064, .052/.040,
  .052/.048, .064/.044, .064/.048, **final .024 (last-3 .051) / .020 (.037)**, against row 1's .824 (.788): blind
  level. **The λ = 0.1 fallback triggers.** (look 1 seed 2 at λ = 1 runs anyway, as approved, since the proposal to
  skip it got no answer before cam0 finished.)
- **look 1 seed 2, λ = 1 (`results/20261001_step1b_dp_supcon_look1_s2/`): native collapses too.** .004/.004,
  .048/.020, .068/.040, .056/.040, .052/.040, .076/.052, .052/.032, .036/.028, .032/.028, .020/.012,
  **final .060 (last-3 .037) / .048 (.029)**, against row 1's .676 (.695).
- **Fallback launched as pre-registered:** row 2 at λ = 0.1 (cam0 seed 1, look 1 seed 2, `…_dp_supcon01_…`), queued
  after the eval batch, followed by the three checks and row 2 stitching on the λ = 0.1 agents. The eval batch still
  runs the λ = 1 checks / collapse measures / stitching (stitched numbers meaningless with blind-level natives; the
  collapse measures are the useful part).
- **λ = 1 eval batch results** (`results/20261001_step1b_stitch_row2_look1/`, `…_blind_step1b_dp_supcon_*`,
  `…_fake_goal_*_step1b_dp_supcon_*`, `…_probe_cube_goal/`):

  | | row 1 cam0 / look 1 | row 2 λ = 1 cam0 / look 1 |
  |---|---|---|
  | NC1 (16 chunk labels) | .92 / 1.29 | **.085 / .092** |
  | effective rank | 3.1 / 4.2 | 4.5 / 4.4 |
  | variance in top 4 / top 16 PCs | .77 / .90, .72 / .89 | .81 / **1.00**, .83 / **1.00** |
  | probe cube x / y (on table) | .78 / .58, .76 / .59 | **.19 / .14, .22 / .11** |
  | blind: full / zeroed / shuffled | .84 / .04 / .06, .72 / .02 / .04 | .05 / .04 / .05, .04 / .01 / .03 |
  | fake goal: grasped true → fake; closest to state goal | .90 → .88; —, .94 → .88; — | .10 → .10; 0.8 cm, .06 → .10; 1.2 cm |
  | stitching identity / SAPS / action_pairs (cam0 enc → look 1 ctrl) | .07 / .45 / .40 | .06 / .05 / .06 |

  Offline, row 2's chance level for z (.021 / .020) is below its noise floor (.028 / .024): the controller's chunk no
  longer depends on z at all.
- **Takeaway (λ = 1):** SupCon on the chunk labels collapses z as intended (NC1 10× lower, all variance in the
  16-class span) but **erases the cube position** from z (probe R² .78 → .19), and the controller falls back on the
  goal_pos shortcut (follows a fake goal to 0.8 cm, grasps 10%), as in cam1 / cam2. So the collapse-vs-control risk
  (PROJECT.md) shows up in its strongest form: with chunk labels the within-label information the policy needs (where
  the cube is) is exactly what collapse throws away. λ = 0.1 is training.
- **λ = 0.1 (`results/20261001_step1b_dp_supcon01_{cam0_s1,look1_s2}/`): native collapses too.** cam0: .000, .044,
  .040, .048, .048, .048, .044, .040, .060, .044, **final .048 (last-3 .051)**; look 1: .004, .028, .040, .076, .040,
  .056, .072, .068, .068, .088, **final .060 (last-3 .072)** (success_once every 5k). NC1 .10 / .10, effective rank
  3.8 / 3.2, variance in top 16 PCs 1.00; probe cube x / y .17 / .10 and .14 / .11 (row 1: .78 / .58, .76 / .59);
  blind full .04 = zeroed .06 / .03; fake goal: follows goal_pos (0.7–1.2 cm), grasps 4–16%. Stitching all
  .04–.06. Chance level for z below the noise floor again (.016 / .022 vs .023 / .027): z is ignored.
- **Takeaway (row 2):** **both pre-registered row 2 hypotheses rejected** at λ = 1 and λ = 0.1: SupCon on 16 chunk
  clusters erases the cube position from z even at a tenth of the weight, and the agents fall back on the goal_pos
  shortcut. Not running TACO / other labels yet (agreed: report first).
- **Row 2 result (logged as agreed):** SupCon on 16 chunk clusters at λ ≥ 0.1 erases the cube position; a weak-λ test
  is pending. (Plan, once the env is decided: pick λ from the smoke test's measured gradient ratio so that SupCon's
  encoder gradient is about equal to the DP term's, likely .005–.01; λ = .1 was still 10–25× stronger.)

**Map-class check, row 1 agents (closed loop, 250 episodes; `results/20261001_step1b_stitch_row1mapclass_look1/`):**

| | cam0 enc → look 1 ctrl (native cam0 .824) | look 1 enc → cam0 ctrl (native look 1 .676) |
|---|---|---|
| SAPS (orthogonal) | .480 / .380 (58% of native); z residual .49 | .388 / .272 (57%); z residual .47 |
| **affine (least squares)** | **.764 / .556 (93%)**; z residual .28 | **.580 / .452 (86%)**; z residual .24 |
| MLP (affine + 1 hidden layer) | .752 / .612 (91%); z residual .29; early stop at epoch 24 | .580 / .432 (86%); z residual .22; epoch 118 |

success_once / success_at_end. SAPS repeats across runs: cam0 enc → look 1 ctrl .460 / .452 / .480, look 1 enc → cam0
ctrl .452 / .388 (spread up to .06 between identical maps: eval noise at 250 episodes).
- **Takeaway: hypothesis rejected — the map class is the bottleneck, not the denoiser.** An unconstrained affine map
  recovers 86–93% of native, against 57–58% for the orthogonal one; the MLP adds nothing over affine. So the paired
  ceiling must be defined with the affine map, and "% of paired ceiling" for row 1 action_pairs drops to about
  .40 / .76 = 52% and .28 / .58 = 48%. The offline chunk distance did not see this (affine ≈ SAPS on all frames,
  .042 vs .046), but the held-out z residual did (.28 vs .49): the residual is the better offline proxy here.
- **Decision (2026-10-01):** from now on, the paired ceiling = affine least squares on paired frames.

**Label-based correspondence × map class, offline (row 1 agents, cam0 ↔ look 1, both directions; CPU, while λ = 0.1
trains).** Can labels alone get closer to the affine ceiling without SupCon?
- **Setup:** every label-based fit takes source frames from demos 0–49 (source domain) and target frames from demos
  50–99 (target domain), so no pair can be the same frame of the same demo (both agents trained on the same demos:
  fine labels or nearest-neighbour matching would otherwise be paired data in disguise). The same frames are
  available to every variant. Correspondences: chunk clusters K = 16 (current labels), K = 64, K = 256 (k-means on
  the same z-scored chunks of demos 0–99, as for K = 16; random same-cluster pairs, ≤100 per cluster, 5 draws);
  continuous action pairs (each source frame paired with the target frame whose z-scored 8-step chunk is nearest).
  Maps on the same pairs: orthogonal Procrustes (current action_pairs) vs affine least squares. References: identity,
  SAPS (paired, demos 0–99), affine paired = the ceiling (demos 0–99). Metrics: held-out z residual (main proxy) and
  offline chunk distance (all / top-30% vision-sensitive frames), on the same 2000 held-out frames as before.
- **Hypotheses:**
  - H1: orthogonal maps stay near or above SAPS's residual (.47–.49) whatever the correspondence: the map class limits.
  - H2: affine on random same-cluster pairs suffers regression dilution at K = 16 (the arbitrary within-cluster
    pairing shrinks the map towards the cluster means), so it is no better than orthogonal there; it improves as the
    correspondence gets finer (K = 64, 256), and continuous nearest-chunk pairs with affine are the best label-based
    variant, closing at least half of the gap between orthogonal action_pairs (.75–.79) and the affine ceiling
    (.24–.28).
  - H3: the offline chunk distance separates the variants less than the z residual.
- **Result** (`results/20261001_step1b_label_maps/`; CPU; z residual = mean of 5 draws for the k* rows, [min–max];
  chunk distance on draw 0, all / top-30%; CPU noise draws, so distances are comparable within this table only):

  | map | pairs | cam0 enc → look 1 ctrl: z residual | chunk dist | look 1 enc → cam0 ctrl: z residual | chunk dist |
  |---|---|---|---|---|---|
  | identity | — | 4.46 | .120 / .181 | 4.27 | .155 / .243 |
  | SAPS (paired, orth) | 7618 | .485 | .050 / .082 | .465 | .054 / .093 |
  | **affine paired (ceiling)** | 7618 | **.284** | .041 / .067 | **.238** | .048 / .070 |
  | k16 orth (= action_pairs) | 1318 | .826 [.81–.85] | .062 / .094 | .763 [.75–.78] | .065 / .123 |
  | k16 affine | 1318 | .711 [.69–.72] | .078 / .149 | .671 [.60–.80] | .092 / .164 |
  | k64 orth | 2389 | .697 [.68–.71] | .057 / .088 | .661 [.65–.68] | .056 / .102 |
  | k64 affine | 2389 | .542 [.52–.56] | .063 / .115 | 1.076 [.53–1.96] | .063 / .112 |
  | k256 orth | 2602 | .672 [.66–.69] | .057 / .087 | .636 [.63–.64] | .059 / .101 |
  | k256 affine | 2602 | .534 [.52–.54] | .066 / .108 | 1.333 [.84–2.25] | .061 / .126 |
  | nn orth | 3720 | .612 | **.048 / .078** | .598 | .061 / .107 |
  | **nn affine** | 3720 | **.473** | .062 / .101 | **.420** | .057 / .105 |

- **Takeaway (offline, one seed):**
  - **H1 confirmed:** orthogonal maps improve with finer correspondence (.83 → .70 → .67 → .61) but stay above SAPS (.49).
  - **H2 partly:** continuous nearest-chunk pairs + affine are the best label-based variant (.47 / .42, about SAPS's
    residual) and close 65% of the gap between k16 orth and the affine ceiling in both directions (≥ half: confirmed).
    But the dilution prediction is wrong at K = 16 (affine beats orth there), and affine on random cluster pairs is
    unstable at K = 64 / 256 in one direction (draws up to 2.2: few pairs per cluster, arbitrary pairings).
  - **H3, stronger than predicted:** chunk distance and z residual disagree. Affine on label pairs lowers the residual
    but raises the chunk distance (nn affine .062 vs nn orth .048). A plausible reason: least squares on noisy pairs
    shrinks the mapped z towards the mean (regression dilution: lower MSE, but off the target's distribution). Not
    tested. Closed loop decides: nn_orth and nn_affine are queued (250 episodes, both directions, after λ = 0.1).
- **Closed loop** (`results/20261001_step1b_stitch_row1labelmaps_look1/`; 250 episodes, success_once / at_end;
  % of the affine paired ceiling .764 / .580 from the map-class run):

  | map | cam0 enc → look 1 ctrl | look 1 enc → cam0 ctrl |
  |---|---|---|
  | affine paired (ceiling) | .764 / .556 | .580 / .452 |
  | SAPS (orth, paired; 2–3 runs) | .46 / .45 / .48 | .45 / .39 |
  | k16 orth (= action_pairs) | .396 (52%) | .284 (49%) |
  | **nn orth** | **.452 (59%)** / .320 | **.488 (84%)** / .340 |
  | **nn affine** | **.492 (64%)** / .356 | **.408 (70%)** / .316 |

- **Takeaway (one seed, eval noise ≈ ±.06):** continuous nearest-chunk pairs lift label-only stitching from about
  50% to 59–84% of the affine ceiling, about SAPS level, **without SupCon**. nn orth vs nn affine: no clear winner
  (each ahead in one direction, within noise), so neither the z residual nor the chunk distance predicted the
  ordering between them.

**cam2 pilot with the goal away from the cube (2026-10-02; GPU).** Does cam2's oracle fail because "go to goal_pos"
already brings the gripper near the cube? Env `StitchPickCubeLollipopNoGraspCam2Goal-v1` = cam2 + the Step 1 goal
variant (goal y ∈ [0.15, 0.25], disjoint from the cube's [−0.1, 0.1]; x and height as default); demos = the goal
variant's planner demos (`20260930_step1_demos_goal_panda_pos`), exported with `export_dp_demos.py lollipop 2
<demos.npz> 0 goal`. One plain-DP oracle (λ = 0, seed 2, demos 0–99, 50k), then blind, fake goal, probe. This decides
whether the core env switches (which would mean retraining all oracles), so no seeds / SupCon runs until it is done.
- **Hypothesis:** with the shortcut useless, cam2's controller must use z for the cube: the oracle works (≥ .6
  success_once last-3, about look 1 / cam0 level), blind drops it to ≤ .05, and under a fake goal it still grasps
  (≥ .8× the true-goal rate) and carries the cube to the fake goal. If it still fails, cam2's problem is not the
  shortcut but reading the cube from that view (its probe R² was .61 / .79, so this would point at the controller /
  training, not the encoder).

**Label-based maps, round 2 (offline, CPU, row 1 agents; source demos 0–49, target demos 50–99, equal frame budget).**
- (a) **Shrinkage:** per-dimension variance of mapped z vs target z (held-out), for nn_orth and nn_affine; plus two
  rescaled variants: orthogonal + per-dimension scale, affine + rescale to the target's per-dimension variance
  (scales from the fit frames). (b) **Pairing on [z-scored chunk, proprio]** (tcp pose + gripper finger qpos, z-scored;
  each block scaled by 1/√dim so chunk and proprio weigh equally), orthogonal / affine. (c) **Mutual-nearest-neighbour
  filtering** (keep only pairs that are each other's nearest neighbour), on every nearest-neighbour variant. (d) **Fair
  ceiling:** affine paired fitted on demos 0–49 only (the label-based fits' frame budget), next to the demos 0–99
  ceiling. Metrics: z residual against both ceilings, chunk distance.
- **Hypotheses:** (a) nn_affine shrinks the mapped variance (total variance ratio < .8) and nn_orth does not (≈ 1:
  orthogonal maps keep norms); rescaling affine raises its residual a bit but lowers its chunk distance. (b) adding
  proprio makes the pairs closer to true correspondences (same phase and arm pose), lowering the residual of both maps
  by ≥ .05. (c) mutual-NN filtering removes bad pairs and lowers the residual further, at the cost of fewer pairs.
  (d) the half-budget ceiling is within .03 of the full one (3.7k paired frames are plenty for a 256-d affine map).
- **After both trainings, one eval batch (no training running):** (1) row 2 oracle checks (blind, fake goal, probe),
  collapse measures, stitching cam0 ↔ look 1 (identity / SAPS / action_pairs, offline + closed loop); (2) the map-class
  check below.

**Map-class ceiling check (row 1 agents, paired frames, closed loop both directions).** SAPS recovers .45 both ways
(55–67% of native). Is the ceiling the map class or the denoiser's sensitivity to errors in z?
- Maps, all fitted on the same paired latents (current frames of demos 0–99): SAPS (orthogonal + translation, as now);
  affine least squares (z W + b, no orthogonality; `fit_affine_paired`); MLP (256 → 512 → 256, ReLU, MSE, Adam;
  early-stopped on the pairs of 10 held-out fit demos). Also reported: held-out z-space residual
  ‖T(z_s) − z_t‖² / ‖z_t − mean‖² on demos 400–497, and the offline chunk distances.
- **Hypothesis:** the ceiling comes mostly from the information mismatch plus the denoiser's sensitivity, not from the
  map class: affine within ±.05 of SAPS in closed loop, MLP at most ~.1 above, even where their z-space residual is
  clearly lower. If affine or MLP reach ≥ .6 (≈ 75–90% of native), the orthogonal map class is the bottleneck, and
  the paired ceiling (and "% of paired ceiling") must be defined with the better map.

**Next to-do (after row 2 and the map-class check; not now): cam2 pilot with the goal sampled away from the cube's
area**, so "go to goal_pos" is useless for grasping (goal y range not overlapping the cube's, as in the goal variant):
new demos, one oracle, the three checks (blind, fake goal, probe).

---

## Step 2 handoff (2026-10-01): how to train, load and read the oracles

- **Oracle recipe:** ManiSkill's Diffusion Policy baseline, unmodified (`third_party/maniskill_diffusion_policy`, v3.0.1),
  env `StitchPickCubeLollipopNoGrasp-v1` (state = qpos, qvel, tcp_pose, goal_pos; 28-d), 100 demos, 50k iterations.
- **Train:** `bash scripts/run_dp.sh stitch.envs:<env-id> <demo.h5> <seed> <name>` (env vars `TOTAL_ITERS`, `NUM_DEMOS`;
  250 eval episodes every 5k; saves the final weights as `checkpoints/49999.pt`). One run at a time: ~12 GB RAM,
  ~11 GB GPU, ~1 h; launch under `systemd-run --user -p MemoryMax=16G -p MemorySwapMax=0` (see `results/*_chain.sh`).
- **Env ids** (`stitch/envs.py`, all PickCube): `StitchPickCube-v1` (old sphere), `…HiddenGoal-v1`, `…Lollipop-v1`,
  `…LollipopNoGrasp-v1` (core), `…LollipopNoGoalState-v1`. Flags: `cam`, `look`, `light`, `task`
  (default/actuation/goal), `robot_uids`, `goal_marker` (sphere/hidden/lollipop), `goal_in_state`, `grasp_in_state`.
  The `module:EnvId` form makes the baseline's eval workers import `stitch.envs`; put the repo root and
  `third_party/maniskill_diffusion_policy` on `PYTHONPATH`.
- **Demos** (`results/` is not in git; everything is in `/home/ricc/projects/labelstitch-step1/results/`): states +
  actions per (task, robot) in `20260930_step1_demos_<task>_<robot>_pos/demos.npz`; DP format via
  `scripts/export_dp_demos.py [goal_marker]` (replays stored actions in our env with `RecordEpisode`; cam0, default
  task, Panda for now), e.g. `20260930_dp_ours_demos_default_panda_cam0_lollipop/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5`.
- **Checkpoints:** core oracle `20261001_step2g_dp_core_nograsp_50k_s1/runs/step2g_dp_core_nograsp_50k_s1/checkpoints/49999.pt`.
- **Load the final EMA agent** (as in `scripts/eval_dp_blind.py`): `import train_rgbd`; build envs with
  `make_eval_envs(env_id, 10, "physx_cpu", env_kwargs, dict(obs_horizon=2), wrappers=[FlattenRGBDObservationWrapper])`;
  `agent = train_rgbd.Agent(envs, train_rgbd.Args())`; `agent.load_state_dict(torch.load(ckpt)["ema_agent"])`.
  Guard the script with `if __name__ == "__main__"` (forkserver workers re-import it).
- **The z we stitch:** `agent.visual_encoder` (PlainConv): RGB/255 `(B, 3, 128, 128)` → `(B, 256)` per frame. In
  `Agent.encode_obs` it is concatenated with the state, per frame (obs horizon 2), into the U-Net's global condition.
  Patching `visual_encoder.forward` is how the blind check replaces z.
- **Gotchas:** gymnasium pinned to 0.29.1 (the baseline reads `final_info`); load `.npz` arrays once (they
  decompress on every access); ManiSkill's evaluation metric is success_once; report final + last-3 mean, 250 episodes.

---

## 2026-10-01 — Step 2g: core oracle setup (goal in state, no is_grasped) + blind check; goal-from-pixels with 498 demos

- **Setup:** unmodified DP recipe, lollipop visible, seed 1, 50k iterations, 250 eval episodes every 5k; final + mean
  of the last 3. Final weights saved with `--save_freq 49999` (by default the baseline keeps only "best" checkpoints).
  (1) Core setup, `StitchPickCubeLollipopNoGrasp-v1`: state = qpos, qvel, tcp_pose, goal_pos (28-d), 100 demos.
      Blind check on its final weights (`scripts/eval_dp_blind.py`, 250 episodes each): full / visual feature zeroed /
      visual feature replaced by that of a random training frame (shuffled). The state part is untouched.
  (2) Goal-from-pixels diagnostic, `StitchPickCubeLollipopNoGoalState-v1` (25-d state), **all 498 demos**.
- **Hypotheses:**
  - (1) ≈ 0.78 like Step 2f run 1 (`is_grasped` is redundant with the gripper qpos). Blind: zeroed and shuffled
    near 0 (≤ 0.05): the cube pose is not in the state, so without vision the policy cannot find the cube.
  - (2) data-limited: with 5× the goal examples, clearly above 0.02 (0.3–0.6). If still near zero: stop; next idea is
    a spatial-softmax ResNet18 encoder (robomimic style), not tuning.
- **Result** (`results/20261001_step2g_dp_core_nograsp_50k_s1/`, `results/20261001_step2g_dp_nogoalstate_498demos_50k_s1/`,
  `results/20261001_step2g_blind_step2g_dp_core_nograsp_50k_s1/`; ~58 min per run; success_once / success_at_end):

  | iteration | (1) core: goal in state, no is_grasped, 100 demos | (2) goal from pixels, 498 demos |
  |---|---|---|
  | 5k | 0.05 / 0.02 | 0.01 / 0.00 |
  | 10k | 0.55 / 0.33 | 0.05 / 0.02 |
  | 15k | 0.76 / 0.53 | 0.07 / 0.04 |
  | 20k | 0.78 / 0.54 | 0.11 / 0.06 |
  | 25k | 0.76 / 0.58 | 0.15 / 0.10 |
  | 30k | 0.81 / 0.60 | 0.19 / 0.12 |
  | 35k | 0.80 / 0.56 | 0.15 / 0.11 |
  | 40k | 0.80 / 0.60 | 0.15 / 0.10 |
  | 45k | 0.74 / 0.57 | 0.14 / 0.10 |
  | **50k (final; last-3 mean)** | **0.824 (0.788) / 0.596 (0.589)** | **0.124 (0.136) / 0.068 (0.091)** |

  Blind check on (1)'s final weights (250 episodes each): **full 0.84 / 0.62, visual feature zeroed 0.044 / 0.036,
  shuffled 0.060 / 0.044.**
- **Takeaway:**
  - **Core oracle recipe confirmed:** goal in state without `is_grasped` = 0.79 last-3 mean, same as with it (0.79).
    The blind check drops it to ~0.05: the policy needs vision (the cube pose is only in the pixels).
  - **Goal from pixels is partly data-limited but far from solved:** 498 demos lift it from 0.02 to 0.12–0.19
    (peak at 30k, then declining), below the 0.3–0.6 hypothesis. Next idea per plan: a spatial-softmax ResNet18
    encoder (robomimic style); not run.

---

## 2026-09-30 — Step 2f: lollipop at 50k iterations, with and without the goal in the state

- **Setup:** unmodified DP recipe, lollipop visible, our 100 demos (all DP runs so far used `--num-demos 100` of 498),
  seed 1, **50k iterations** (default from now on), 250 eval episodes every 5k; final + mean of the last 3.
  (1) `StitchPickCubeLollipop-v1`: state = qpos, qvel, is_grasped, tcp_pose, goal_pos (29-d).
  (2) `StitchPickCubeLollipopNoGoalState-v1` (`goal_in_state=False`): state = qpos, qvel, tcp_pose (25-d); the goal
  must be read from the lollipop. Implemented in the env's observation, not by overriding
  `build_state_obs_extractor`: the baseline's eval state comes from `FlattenRGBDObservationWrapper` (all of agent +
  extra) and ignores the extractor, while its demo loader already drops keys the env does not produce
  (`reorder_keys`), so train and eval both get 25-d with the baseline code untouched. Smoke-tested (20 iterations).
- **Hypotheses** (written after launch, before any evaluation past iteration 0):
  - (1) catches up with more training: ≈ 0.75–0.78 at 50k (the 30k run was still rising, 0.72).
  - (2) reading the goal from pixels costs 0.1–0.25 vs (1): the goal is 3-D and the lollipop is a few pixels,
    partly hidden by the hand at some positions; losing `is_grasped` should matter little (the gripper state is in qpos).
- **Result** (`results/20260930_step2f_dp_lollipop_{goalstate,nogoalstate}_50k_s1/`, ~57 min each; success_once /
  success_at_end every 5k, 250 episodes):

  | iteration | (1) goal in state | (2) goal from pixels only |
  |---|---|---|
  | 5k | 0.04 / 0.02 | 0.00 / 0.00 |
  | 10k | 0.22 / 0.16 | 0.01 / 0.00 |
  | 15k | 0.58 / 0.42 | 0.00 / 0.00 |
  | 20k | 0.73 / 0.56 | 0.02 / 0.01 |
  | 25k | 0.79 / 0.64 | 0.02 / 0.01 |
  | 30k | 0.80 / 0.65 | 0.01 / 0.00 |
  | 35k | 0.80 / 0.63 | 0.01 / 0.00 |
  | 40k | 0.78 / 0.60 | 0.02 / 0.00 |
  | 45k | 0.82 / 0.64 | 0.02 / 0.02 |
  | **50k (final; last-3 mean)** | **0.772 (0.789) / 0.616 (0.620)** | **0.016 (0.019) / 0.004 (0.008)** |

  Run (2), final-eval video (one episode): the policy reaches and grasps the cube (`is_grasped` = 1 from ~step 40),
  then holds it near the table and never carries it towards the lollipop.
- **Takeaway:**
  - (1) With 50k iterations the lollipop run matches the hidden-sphere result (0.79 vs 0.78 last-3 mean): the lollipop
    costs nothing once the goal is in the state. (At 30k this run already had 0.80 vs 0.72 for the 30k run: the LR
    schedule spans the whole run, plus one-seed noise.)
  - (2) **Removing `goal_pos` + `is_grasped` collapses DP to ~0.02**, far beyond the hypothesised 0.1–0.25: the
    policy grasps but does not carry, i.e. it does not read the goal from the lollipop (with 100 demos, 50k iterations).
  - Stopped here as agreed (> 0.2 drop). Next diagnostics (not run): remove `goal_pos` and `is_grasped` separately;
    train on all 498 demos.

---

## 2026-09-30 — Step 2e: is the visible goal sphere the cause? (hidden-sphere run + occlusion count)

- **Protocol from now on:** final checkpoint (30k), mean of the last 3 checkpoints (20k/25k/30k) in brackets,
  250 eval episodes per checkpoint (`--num_eval_episodes 250`), no best checkpoint.
- **Setup:** (1) `StitchPickCubeHiddenGoal-v1` (our env, `goal_marker="hidden"`: pixel-identical to PickCube-v1),
  our demos re-exported without the sphere, unmodified DP recipe, seed 1. (2) Occlusion count, no training: for our
  first 100 demos, render each state with and without the sphere (segmentation) and count frames where sphere pixels
  cover cube or gripper (hand + fingers) pixels, overall and within ±5 steps of the gripper closing.
- **Hypotheses:** (1) ≈ 0.77, as on their env (the data is the same). (2) The sphere covers cube/gripper pixels in a
  noticeable fraction of frames near the grasp (≥ 10%), much more than overall.
- **Result 1, hidden sphere** (`results/20260930_step2e_dp_ourenv_hiddengoal_s1/`, 250 episodes, ~35 min):
  **0.776 (last 3: 0.779)** success_once, 0.632 (0.616) success_at_end. Same as their env (0.77).
- **Result 2, occlusion** (`results/20260930_step2e_goal_occlusion_{sphere,lollipop}/`, 100 demos, ~7.7k frames per
  camera; "near grasp" = ±5 steps around the gripper closing):

  | camera | frames where the marker covers cube/gripper: sphere | lollipop | cube pixels covered: sphere | lollipop |
  |---|---|---|---|---|
  | cam0 | 37% (near grasp 27%) | 32% (29%) | **10.2%** (5.0%) | **2.6%** (1.9%) |
  | cam1 | 18% (8%) | 15% (14%) | 4.5% (1.8%) | 2.5% (1.9%) |
  | cam2 | 36% (22%) | 35% (38%) | 6.7% (2.9%) | 2.2% (2.1%) |

  (lollipop = 8 mm pole for this count; the pole was then thickened to 12 mm.)
- **Takeaway:**
  - **The visible goal sphere is the cause of the gap:** hidden, our env matches their env (0.78 vs 0.77).
  - The sphere covers cube/gripper pixels often (37% of cam0 frames), most during the carry, not near the grasp as
    hypothesised. The lollipop covers ~4× fewer cube pixels, though its thin pole still touches cube/gripper pixels in
    a similar fraction of frames.
  - Replacement marker, approved: **lollipop** (`goal_marker="lollipop"`, env `StitchPickCubeLollipop-v1`): magenta
    1.25 cm sphere on a 12 mm vertical pole to the table, non-colliding (`results/20260930_step2e_goal_marker_lollipop/grid.png`).
- **Known limitations (relevant once `goal_pos` leaves the state):**
  - at some goal positions the hand hides the marker completely from cam0 (e.g. [-0.1, 0.1, 0.17]);
  - low goals (z ≈ 0.03) are only a few pixels next to the cube, and magenta sits close to the red cube in look0.
- **Next run, hypothesis:** lollipop visible in training and eval, `goal_pos` still in the state, seed 1, same
  protocol: close to the hidden-sphere result (≥ 0.70).
- **Result 3, lollipop** (`results/20260930_step2e_dp_ourenv_lollipop_s1/`, 250 episodes, ~34 min):
  **0.716 (last 3: 0.675)** success_once, 0.568 (0.523) success_at_end. Curve: 0.30 at 10k, 0.58 at 15k, 0.64 at 20k,
  0.67 at 25k, 0.72 at 30k — still rising at 30k, while the hidden-sphere run had plateaued (~0.78 from 20k).

  | goal marker (our env, our demos, seed 1) | success_once: final (last 3) | success_at_end: final (last 3) |
  |---|---|---|
  | hidden (= PickCube-v1) | 0.776 (0.779) | 0.632 (0.616) |
  | lollipop | 0.716 (0.675) | 0.568 (0.523) |
  | sphere (Step 2c, 100 episodes, seeds 1 / 2) | 0.51 / 0.56 | 0.39 / 0.46 |

- **Takeaway (lollipop):** most of the gap closes (sphere ~0.53 → lollipop 0.72 final), within ~0.06 of hidden at
  the final checkpoint but ~0.10 below on the last-3 mean: slower learning, one seed. The hypothesis (≥ 0.70) holds
  for the final checkpoint only.

---

## 2026-09-30 — Step 2d: do our demos differ from ManiSkill's? (training-free)

- **Setup:** `scripts/compare_demo_sets.py`, first 100 shared seeds, both sets in `pd_ee_delta_pos`. Their demos:
  ManiSkill's official motion-planning solution (commit 652ad93, planned in `pd_joint_pos`, converted with
  `replay_trajectory --use-first-env-state -c pd_ee_delta_pos -o rgb -b physx_cpu`). Ours: the same solution copied
  in `collect_demos.py`, converted with the same function (`from_pd_joint_pos_to_ee`).
- **Hypothesis:** our demos have more or longer rest steps after conversion.
- **Result** (`results/20260930_step2d_compare_demo_sets/`):

  | | length | gripper close step | rest steps (before / after close) | mean \|position action\| |
  |---|---|---|---|---|
  | theirs | 77.2 | 41.7 | 16.7% (10.6% / 24.6%) | 0.058 |
  | ours | 77.2 | 41.7 | 16.7% (10.6% / 24.6%) | 0.058 |

  Same seed: 100/100 same length, max action difference 0.009 (median 1e-5), state difference median 1e-4
  (one `is_grasped` flag flips a step earlier/later), frames differ only by the goal sphere (0.6 other pixels per frame).
- **Takeaway:**
  - **Hypothesis rejected: the demo sets are the same demos.** The Step 2c gap is not the demos; the only systematic
    difference in the training data is the goal sphere in our frames (visible in ~83% of them).
  - Steps 2–3 of the plan (re-render their demos in our env, fix our generation) are moot: re-rendering their demos
    in our env reproduces our demos. Open question: why a visible goal sphere costs ~0.2 when `goal_pos` is also in the state.

---

## 2026-09-30 — Step 2c: where does the DP gap come from (0.77 their env → 0.51 our env)?

- **Setup:** (1) frame/state check without training: frames from the baseline's eval env on our env id
  (`reconfiguration_freq=1`) vs our training frames for the same seeds, and vs PickCube-v1 for the same seeds
  (camera pose, intrinsics, resolution, textures, lighting); state vector fields and dims. (2) 2×2 swap with the
  unmodified DP recipe, seed 1, 30k iterations: their demos on our env, our demos on their env. (3) Seed 2 on our
  env with our demos. Goal sphere stays visible in our env; `goal_pos` / `is_grasped` stay in the state.
- **Hypotheses:**
  - Frames: identical to PickCube-v1 except the green goal sphere; identical between eval and training for the same
    seed; state vector identical (29-d: qpos 9, qvel 9, is_grasped 1, tcp_pose 7, goal_pos 3).
  - The gap is mostly DP seed variance plus the visible goal sphere, not the demos: our demos come from the same
    planner with the same length distribution. Expect seed 2 on our env within ±0.15 of 0.51, "their demos on our
    env" ≈ our env's number, "our demos on their env" ≈ their env's number.
  - Caveat: in both swap cells the goal sphere is in the training frames but not the eval frames (or the reverse),
    so a swap cell that drops points to the visible sphere, not to the demos.
- **Result 1, frames and state** (`results/20260930_step2c_check_dp_frames/`): as hypothesised. Eval frames =
  training frames (0 px differ, both envs); our env vs PickCube-v1 for the same seed differ only in the goal sphere
  (0–47 px); intrinsics/extrinsics equal; state identical (29-d, same fields and values). Both demo sets use seeds
  0, 1, 2, … so they share initial states.
- **Result 2, env × demos × seed** (unmodified DP recipe, 30k iterations, 100 eval episodes; success_once /
  success_at_end at 30k, best success_once over the 5k checkpoints in brackets):

  | env (eval) | demos (train) | goal sphere train → eval | seed | final | best |
  |---|---|---|---|---|---|
  | theirs | theirs | hidden → hidden | 1 | **0.77 / 0.64** | 0.77 |
  | theirs | ours | visible → hidden | 1 | 0.57 / 0.44 | 0.70 |
  | ours | theirs | hidden → visible | 1 | 0.65 / 0.59 | 0.65 |
  | ours | ours | visible → visible | 1 | 0.51 / 0.39 | 0.59 |
  | ours | ours | visible → visible | 2 | 0.56 / 0.46 | 0.61 |

  Checkpoint-to-checkpoint swings within one run are up to ~0.15 (e.g. 0.70 → 0.57), larger than the ~0.05
  binomial noise of 100 episodes. Runs are one at a time (~12 GB RAM, ~11 GB GPU each; ~30 min).
- **Takeaway:**
  - Seed variance on our env is small (0.51 vs 0.56), so the gap to 0.77 is real.
  - ~~The demos matter more than the env~~ — **corrected by Step 2d:** the two demo sets have identical actions and
    states, so "demos" here really means "goal sphere in the training frames". Read the table by that column:
    trained without the sphere 0.77 (eval hidden) / 0.65 (eval visible); trained with it 0.57 / 0.51–0.56.

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

## 2026-09-29 — Step 4 (offline, Mario latents): GW and action-pair + fused-GW aligners

- **Hypotheses:**
  - **H1:** GW alone (label-free) < action_pairs. Pure geometry cannot recover the map. Expected failure mode: collapsed SCIL clusters of similar size get swapped by a label-blind matcher, giving plausible geometry but the wrong actions. This shows up as low plan label accuracy (fraction of transported mass landing on same-action frames).
  - **H2:** action_pairs_fgw ≈ action_pairs with all pairs (≤100 per action), and better than action_pairs with few pairs (5 per action).
- **Setup:** offline only, on exported latents (`mario/export_latents.py` → `results/20260929_step4_latents/`). Nature CNN bc / scil / scil_taco3, seeds 0–2, encoder seed s + controller seed s+1. Settings: v1 version pairs (v0→v1, v1→v0, v1→v2, v2→v1; unpaired fits on disjoint episodes) and cross-level (1-1 ↔ 1-2). Agreement = stitched controller vs the native agent's action on held-out frames.
  - Aligners: identity; SAPS (versions only); gw (2–3 ε values, every ε reported, none selected on agreement); action_pairs at ≤100 and at 5 per action (5 draws, plus the same first 3 draws); action_pairs_fgw at ≤100 and at 5 per action (3 draws, α = 0.5).
  - GW/FGW run on 1000 frames per side, and the map is applied to all frames. GW subsamples uniformly (label-free). FGW with 5 per action: only the anchor frames keep labels; they are forced into the subsample, and pairs with unlabelled frames get a neutral mismatch of 0.5.
  - Reproduction check first: identity / SAPS / action_pairs on all 6 version permutations must match `20260929_anchors_offline*`.
- **Pilot** (seed 0, one draw, v0→v1 and 1-1 with the 1-2 controller): reproduction matches the reference to three decimals. At ε ∈ {0.005, 0.01, 0.05} the GW plans are nearly uniform (each source frame spreads its mass over 370–950 of 1000 targets), so GW failures there are an ε artefact. Full-run ε grid changed to **{0.0005, 0.001, 0.005}** for GW and FGW, chosen by plan sharpness (label-free), not by agreement. Surprise: at small ε, SCIL GW reaches action_pairs level across versions (.747 vs .734, 79% of mass on same-action frames) but fails across levels (.34 vs .83).
- **Added hypothesis H3 (written after the pilot, before the test):** GW works across versions because both clouds share one state distribution (same level, same human runs), and fails across levels because they don't. Test within one level: v0→v1, fitting on source frames from the first half of 1-1 (x-position below the median) and target frames from the second half (disjoint episodes as before); evaluate on all held-out frames. Prediction: GW drops to the cross-level result (plan label accuracy near chance), while action_pairs stays close to its normal value. bc and scil; gw (3 ε), action_pairs, fgw.
- **What was tested:** offline action agreement only, on held-out frames of exported Mario latents. No in-game play.
- **Result** (agreement, mean ± std over 3 seeds, each seed averaged over its pairs; versions = 4 v1 pairs, levels = both directions, halves = v0→v1 fitted on first half → second half of 1-1; `results/20260929_step4_aligners_offline_full/`):

  | aligner | bc versions | scil versions | scil_taco3 versions | bc levels | scil levels | scil_taco3 levels | bc halves | scil halves | fit time |
  |---|---|---|---|---|---|---|---|---|---|
  | identity | .396 ± .014 | .355 ± .018 | .382 ± .074 | .411 ± .016 | .309 ± .018 | .388 ± .082 | .423 ± .029 | .229 ± .068 | 0 s |
  | SAPS (paired ceiling) | .705 ± .001 | .752 ± .015 | .758 ± .010 | — | — | — | — | — | 0.07 s |
  | action_pairs ≤100 (5 / 3 draws) | .630 / .635 | .743 / .744 | .748 / .746 | .682 / .682 | .844 / .838 | .827 / .825 | .593 / .599 | .731 / .730 | 0.03 s |
  | action_pairs 5 (5 / 3 draws) | .473 / .472 | .731 / .731 | .718 / .719 | .508 / .501 | .825 / .820 | .780 / .784 | .450 / .461 | .716 / .722 | 0.03 s |
  | gw ε .0005 | .413 ± .016 | .670 ± .006 | .593 ± .051 | .401 ± .042 | .631 ± .049 | .818 ± .007 | .376 ± .031 | .405 ± .092 | 19.5 s |
  | gw ε .001 | .402 ± .043 | .647 ± .013 | .571 ± .064 | .397 ± .015 | .596 ± .047 | .817 ± .010 | .359 ± .035 | .409 ± .096 | 9.9 s |
  | gw ε .005 | .411 ± .022 | .645 ± .029 | .506 ± .081 | .389 ± .024 | .575 ± .032 | .795 ± .022 | .368 ± .049 | .349 ± .105 | 0.9 s |
  | fgw ≤100, ε .0005 / .001 / .005 | .664 / .662 / .663 | .745 / .742 / .742 | .748 / .741 / .741 | .717 / .712 / .708 | .828 / .815 / .808 | .831 / .829 / .829 | .594 / .593 / .605 | .697 / .697 / .696 | 3.0 / 1.7 / 1.1 s |
  | fgw 5, ε .0005 / .001 / .005 | .533 / .534 / .536 | .739 / .739 / .739 | .739 / .738 / .738 | .548 / .549 / .552 | .803 / .803 / .800 | .824 / .822 / .822 | .473 / .475 / .480 | .700 / .700 / .704 | 2.6 / 1.8 / 1.3 s |

  FGW uses 3 draws; compare it with the 3-draw action_pairs numbers. FGW std over seeds is ≤ .019 everywhere.
  Plan label accuracy for GW at ε .0005 (chance ≈ .28–.29): bc .32 / .32 / .32 (versions / levels / halves); scil .74 / .65 / .41; scil_taco3 .61 / .78 / —. Each source frame spreads its mass over 40–96 targets at ε .0005 and 340–620 at ε .005.
  Per-seed GW agreement tracks plan label accuracy. Examples: scil v0→v1 gives .74 / .51 / .71 at label accuracy .79 / .58 / .84, and on halves .47 / .27 / .48 at .47 / .25 / .53. scil_taco3 v0→v1 gives .36 / .50 / .60, yet across levels it gives .78–.84 in every seed.
  Full run: ~2 h of fitting on 6 CPU cores, dominated by GW at ε ≤ .001. At ε ≤ .001, FGW ≤100 does not fully converge (Sinkhorn marginal error up to 0.1–0.2); at ε .005 the error is 5e-3 and agreement is the same.
- **Takeaway:**
  - **H1 confirmed on average, and GW is unreliable.** Label-free GW is below action_pairs for every encoder and setting except scil_taco3 across levels (.818 vs .827). For BC it is near identity and its plans are at chance (label accuracy .32 vs .29): BC latents have no action geometry for GW to find. For collapsed encoders GW sometimes recovers the right cluster matching and sometimes not, depending on seed and pair. Agreement follows how much mass lands on same-action frames, which is the predicted failure mode (label-blind cluster swaps). We have no label-free signal yet that tells a good GW plan from a bad one.
  - **H2 holds only for BC.** With ≤100 pairs, FGW ≈ action_pairs for SCIL and SCIL+TACO (within ±.02; slightly worse on SCIL levels, −.010, and halves, −.033). With 5 pairs, FGW does **not** help SCIL: 5 pairs already give 98% of the ≤100 number. For BC, FGW adds +.03 (≤100) and +.03 to +.06 (5 pairs), still well below SAPS (.664 vs .705). The ε choice barely matters for FGW. **action_pairs stays the default aligner:** 100× faster, and never worse for collapsed encoders.
  - **H3 partly supported.** Within one level, fitting on disjoint halves drops SCIL GW from .670 (label accuracy .74) to .405 (.41), while action_pairs barely moves (.731 vs .743 on v1 pairs; .73 vs .73 on v0→v1 alone). But scil_taco3 GW works *across* levels (.818, every seed) and fails in some v0→v1 seeds, so "different levels = different state distributions" does not explain everything. **Claim to test, not a fact:** "pure geometric alignment requires matched state distributions; label alignment does not." Evidence for: the halves test. Evidence against: scil_taco3 across levels.
- **Next (not run):** check whether the GW objective (label-free) predicts which GW plans are good; if not, GW stays out of the method as more than a baseline. Test the halves split for scil_taco3 and in ManiSkill (Step 4 proper) before using this claim in the paper.

**Follow-ups (2026-09-30, hypotheses written before running):**
- **F1, halves for scil_taco3** (same setup as bc/scil, all aligners). If GW still works on mismatched halves, that favours a "shared dynamics structure" explanation over "matched state distributions". My expectation: it fails, like SCIL (.41). SCIL+TACO's cross-level success would then be about 1-1 and 1-2 sharing a whole-run structure that disjoint halves of one level do not.
- **F2, is the GW objective a label-free failure signal?** Over all GW fits (per draw, same seeds as the full run, so the same plans), Spearman ρ between the final GW objective (the square-loss GW term Σ (C1_ik − C2_jl)² P_ij P_kl, without entropy) and (a) plan label accuracy, (b) agreement. Pooled and within each (encoder, pair type, ε) group. Hypothesis: within a group, lower objective ↔ higher label accuracy (ρ ≲ −0.5), because a wrong cluster swap should distort the geometry more than the right matching; pooled ρ is confounded by encoder type.
- **F1 result** (scil_taco3, v0→v1 fitted on first half → second half of 1-1, mean ± std over 3 seeds; `results/20260930_step4_aligners_offline_followup/`):

  | scil_taco3 halves | identity | action_pairs ≤100 / 5 (3 draws) | gw ε .0005 / .001 / .005 | fgw ≤100 (ε .0005) | fgw 5 (ε .0005) |
  |---|---|---|---|---|---|
  | agreement | .437 ± .072 | .741 / .701 | **.673 ± .017** / .671 / .651 | .697 | .701 |
  | plan label accuracy (chance ≈ .28) | — | — | .67 / .66 / .54 | .81 | .69 |

  **Refutes my expectation.** GW on mismatched halves works for SCIL+TACO in every seed (.65–.69, 91% of action_pairs), while it fails for SCIL (.41) on the same split. For SCIL+TACO it even beats the *matched* v0→v1 case (.36 / .50 / .60 over seeds). So matched state distributions are neither necessary (SCIL+TACO halves and levels) nor sufficient (SCIL+TACO v0→v1) for GW. The candidate explanations are a geometry shaped by the temporal (TACO) term that is shared across parts of the game, or which local optimum the solver lands in. Neither is tested yet.
- **F2 result** (567 GW fits, one per draw; the plans are identical to the full run, whose agreement values match exactly): Spearman ρ between the final GW objective and plan label accuracy / agreement.

  | | versions | levels | halves |
  |---|---|---|---|
  | bc | −.08 / +.07 | +.13 / −.06 | +.01 / −.10 |
  | scil | −.21 / −.22 | −.29 / −.35 | +.25 / +.22 |
  | scil_taco3 | **−.67 / −.55** | −.36 / −.41 | +.15 / −.60 |

  (mean over the 3 ε values; n = 36 / 18 / 9 fits per ε). Pooled over everything: −.23 / −.05. Within an encoder, pooled over pair types: scil_taco3 +.29 with agreement, i.e. the wrong sign, because objective values are not comparable across pair types.

  **Hypothesis mostly rejected.** The GW objective predicts plan quality only for SCIL+TACO within a fixed pair type (ρ ≈ −.6 on versions). It is weak for SCIL (≈ −.25, and positive on halves), absent for BC, and useless across pair types. **No usable label-free failure signal for GW** in these runs; this is a limitation of GW as an aligner and rules the GW objective out as a stitchability component for now.
- **Updated claim to test:** the "pure geometric alignment requires matched state distributions; label alignment does not" reading is **not supported** as stated. The evidence for it (SCIL: halves .41 vs versions .67) is outweighed by SCIL+TACO (halves .67 and levels .82 work; matched v0→v1 .49 does not). What stands: **label alignment is robust to the state-distribution change** in every encoder (action_pairs moves by ≤ .03 on halves). When label-free geometry works is open (PROJECT.md, open question 10).

---

## 2026-09-29 — Step 0c: does a TACO temporal loss keep or break label-only alignability?

- **Hypotheses:**
  - **H1:** a TACO-only encoder (temporal InfoNCE between [latent of frame t + embedded actions t..t+K−1] and the latent of frame t+K; K ∈ {1, 3}; controller trained with BC) does not collapse, and aligns from labels about as badly as BC.
  - **H2:** SCIL + TACO (SupCon directly on the latent, TACO on its own small projection heads) keeps SCIL's collapse and label-only alignability, with equal or better native play.
- **Setup:** Nature CNN, 1-1 v0/v1/v2 and 1-2 v0, 3 seeds, 50 epochs, same recipe as BC/SCIL (whose agents are reused). With skip 4, "t+K" is K decisions later (4K NES frames). TACO: action embedding (8 → 32) per step, concatenated over K steps; query head MLP([z_t, actions]) → 128, key head MLP(z_{t+K}) → 128, cosine InfoNCE (τ = 0.1), in-batch negatives, weight 1. The TACO term uses its own batch of 1024 transitions per step (from within-episode pairs), while cross-entropy / SupCon keep batch 256 and the same number of steps as the existing agents. Pilot first (SCIL+TACO, K = 3, v0, seed 0) to check the losses don't conflict.
- **Pilot:** no conflict. All three losses decrease smoothly; SupCon ends at 4.65 (SCIL alone ≈ 4.5); NC1 0.74 vs 0.59 for SCIL (same seed).
- **Result** (Nature CNN, mean over 4 domains × 3 seeds; alignment = random same-action pairs; in-game % of native; `results/20260929_step0c_table/`):

  | | NC1 | eff. rank | var. in centroid span | native max x, 1-1 / 1-2 (flags) | offline agreement, versions / levels | in-game, v1 pairs | in-game, cross-level |
  |---|---|---|---|---|---|---|---|
  | BC | 4.97 | 43.8 | 23% | 1515 / 1075 (11%) | .630 / .682 | 703 (49%) | 488 (38%) |
  | SCIL | 0.47 | 6.7 | 71% | 1562 / 654 (8%) | .754 / .844 | 1366 (91%) | 999 (99%) |
  | TACO K=1 | 8.28 | 59.1 | 16% | 1532 / 1133 (10%) | .618 / .638 | 540 (37%) | 464 (35%) |
  | TACO K=3 | 7.61 | 59.9 | 16% | 1629 / 974 (9%) | .625 / .659 | 491 (30%) | 308 (25%) |
  | SCIL+TACO K=1 | 0.62 | 8.3 | 63% | 1651 / 773 (12%) | .748 / .822 | 1212 (77%) | 811 (73%) |
  | **SCIL+TACO K=3** | **0.65** | 8.7 | 61% | **1670 / 1207 (18%)** | **.750 / .827** | **1499 (97%)** | 966 (68%) |

  Cross-level % averages both directions; SCIL's 99% is inflated by its weak native 1-2 agent (654), so compare raw distance there (SCIL 999 vs SCIL+TACO K=3 966).
- **t-SNE** (1-1 training frames, `results/20260929_step0c_tsne/`): BC shows no action structure. SCIL shows tight, separated clusters per action. TACO alone shows long curved filaments that follow trajectories in time, with actions mixed along them (organised by time, not action). SCIL+TACO keeps SCIL's action clusters, with curved temporal strands inside the large clusters (R, R+B).
- **Takeaway:**
  - **H1 confirmed:** TACO alone does not collapse (NC1 ≈ 8, even less than BC) and aligns from labels as badly as or worse than BC (30–37% in-game on the v1 pairs).
  - **H2 confirmed for K = 3:** SCIL+TACO keeps SCIL's collapse and label-only alignability. Alignment *matches* SCIL (97% vs 91% in-game on the v1 pairs is within noise; same raw distance across levels), it does not beat it. K = 1 aligns a bit worse in-game (77% / 73%).
  - **The main gain is native play.** SCIL alone hurts 1-2 (654 vs BC 1075); SCIL+TACO recovers it (1207, 18% flags overall vs SCIL's 8%). So the collapse-versus-control tradeoff already appears in Mario, and the temporal term resolves it.
  - **Supports the PLAN Step 0c decision "SCIL+TACO keeps alignability"** → it goes into the Step 1b candidate list as the leading hybrid (K = 3).
- **Open control for the paper (not run):** the TACO term used its own batch of 1024 transitions per step *on top of* the 256-frame CE/SupCon batch, so TACO variants see more frames per step. A control with matched frames per step (or TACO on the same 256 frames) is needed before claiming the native-play gain comes from the temporal loss itself.
- **Next:** Step 1 / 1b (ManiSkill). The within-action information probe is where TACO should matter most.

**Follow-ups (hypotheses written before running):**
- **F1, seed pairing in notebook section 7.2:** SCIL+TACO K=3 has high no-map agreement (0.44 / 0.50 vs SCIL 0.19 / 0.48). Hypothesis: not a seed artefact (the notebook uses seed 0 for 1-1 and seed 1 for 1-2, so encoder and controller never share an initialisation); the no-map value is chance-level agreement driven by the action marginals (the other controller mostly outputs R, the native agent predicts R about half the time).
- **F2, within-action information probe:** for BC, SCIL and SCIL+TACO K=3, within each large action cluster (R, R+A, R+B, NOOP), a linear probe from the latent to (a) Mario's x-velocity, (b) the action 3 decisions later, (c) decisions until the next jump onset. Held-out frames; mean within-cluster R² (accuracy for b). Hypothesis: SCIL < BC ≈ SCIL+TACO.
- **F1 result:** confirmed, not a seed artefact. The notebook pairs 1-1 seed 0 with 1-2 seed 1, and a check over all (s, s+1) pairings gives the same picture. Without a map, the other level's controller collapses to one action (R for 69–100% of frames in most cases), so no-map agreement equals chance from the two agents' action frequencies (SCIL+TACO play 1-1: 0.44 vs 0.44 expected; all 12 cases at or below chance). SCIL's lower number (0.19) only reflects which action its unmapped controller gets stuck on.
- **F2 result** (held-out frames, within-cluster linear probe with cross-validated regularisation, mean over the 4 clusters and 3 seeds; `results/20260929_step0c_probe/`):

  | | velocity R² | action 3 decisions later: acc. (majority baseline) | decisions to next jump: R² |
  |---|---|---|---|
  | 1-1 BC | −0.18 | .664 (.621) | .097 |
  | 1-1 SCIL | −0.96 | .624 (.621) | .045 |
  | **1-1 SCIL+TACO K=3** | **−0.10** | **.654** (.621) | **.138** |
  | 1-2 BC / SCIL / SCIL+TACO | −2.5 / −18.3 / −6.8 | .561 / .466 / .497 (.558) | −.01 / −1.93 / −.69 |

  **Takeaway:** on 1-1 the ordering holds on all three targets, **SCIL < BC ≈ SCIL+TACO**: SCIL loses within-action information (future action at the majority baseline), and the temporal term recovers it. But the absolute signal is weak: velocity R² is negative for every method (linear probes from 9–10 human episodes do not generalise to held-out runs), and on 1-2 (small clusters, weak agents) the probe fails for all methods. Supportive, not conclusive; the real test of within-action information is the continuous-control probe in Step 1b.

**Mario is frozen after this entry; the GPU goes to ManiSkill.**

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
