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
