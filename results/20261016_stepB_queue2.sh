#!/bin/bash
# Step B, GPU queue after the PeS diagnosis (unit stepb-pes-diag; the faithful PeS chain was stopped): one job at a time (EXPERIMENTS.md, 2026-10-16 entries).
# A: GW closed loop on the 24 matrix cells at the eps chosen by the label-free rule (EPS below, from the fit).
# B: DART on Step R's combinations (same task, goal shift), seed pairs s -> s+1, N in {1, 5, 10}: source and target
#    full fine-tunes of the controller-side oracle (1000 iterations, batch 64), then the merge + closed loop.
# C: map-only fine-tuning at N = 1 (Step R's recipe; batch 64 since one demo has fewer frames than 256).
cd /home/ricc/projects/labelstitch-b
while systemctl --user is-active --quiet stepb-pes-diag; do sleep 300; done
R=$PWD
B=/home/ricc/projects/labelstitch-1b/results
RR=/home/ricc/projects/labelstitch-r/results
export PYTHONPATH=$R:$R/third_party/maniskill_diffusion_policy CUDA_VISIBLE_DEVICES=0
L=$B/20261001_step1b_labels/labels.pt
T=trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5
py() { uv run python -W ignore "$@"; }
run() { ls -d $R/results/2026*_$1 | tail -1; }
ck() { ls $1/runs/*/checkpoints/[0-9]*.pt | sort -V | tail -1; }
nx() { echo $(( $1 % 3 + 1 )); }
EPS=0.0005  # label-free rule: smallest eps with marginal error < .01 in all 24 cells (fit, 16 Oct)
L2=([1]=$B/20261003_step1b_dp_ref_look2_s1 [2]=$B/20261002_step1b_dp_ref_look2_s2 [3]=$B/20261003_step1b_dp_ref_look2_s3)
CAM0=([1]=/home/ricc/projects/labelstitch-step1/results/20261001_step2g_dp_core_nograsp_50k_s1 [2]=$B/20261002_step1b_dp_ref_cam0_s2 [3]=$B/20261002_step1b_dp_ref_cam0_s3)
GC=([1]=$RR/20261004_stepR_dp_ref_goal_cam0_s1 [2]=$B/20261003_step1b_dp_ref_goal_cam0_s2 [3]=$RR/20261004_stepR_dp_ref_goal_cam0_s3)
H0=/home/ricc/projects/labelstitch-step1/results/20260930_dp_ours_demos_default_panda_cam0_lollipop/$T
HGC=$B/20261003_dp_ours_demos_goal_panda_cam0_lollipop/$T
H2=$RR/20261002_dp_ours_demos_default_panda_cam0_look2_lollipop/$T
HG=$RR/20261003_dp_ours_demos_goal_panda_cam0_look2_lollipop/$T
V2=$RR/20261004_stepR_val_demos_look2/trajectory.h5
VG=$RR/20261004_stepR_val_demos_goal_look2/trajectory.h5
E0=stitch.envs:StitchPickCubeLollipopNoGrasp-v1
EGC=stitch.envs:StitchPickCubeLollipopNoGraspGoal-v1
E2=stitch.envs:StitchPickCubeLollipopNoGraspLook2-v1
EG=stitch.envs:StitchPickCubeLollipopNoGraspLook2Goal-v1

# A: GW closed loop
for d in look1 look2 look2light1 cam3; do for p in p1 p2 p3; do
  py scripts/stepB_gw_matrix.py eval $d $p --eps=$EPS > results/$(date +%Y%m%d)_stepB_gw_matrix_${d}_${p}_log.txt 2>&1; echo "gw $d $p exit $?"
done; done

# B: DART. combo  base(controller-side oracle)  source env  source demos  target env  target demos
for s in 1 2 3; do c=$(nx $s); for n in 1 5 10; do
  for spec in "same ${CAM0[$c]} $E0 $H0 $E2 $H2" "goal ${GC[$c]} $EGC $HGC $EG $HG"; do
    set -- $spec; K=$1; BASE=$2; SE=$3; SH=$4; TE=$5; TH=$6
    N=stepB_dart_ft_src_${K}_p${s}_n$n; D=$(date +%Y%m%d); mkdir -p results/${D}_$N
    NUM_DEMOS=$n TOTAL_ITERS=1000 EVAL_EPISODES=10 SUPCON_WEIGHT=0 EXTRA="--init_ckpt $(ck $BASE) --batch_size 64 --eval_freq 100000" \
      bash scripts/run_dp_supcon.sh $SE $SH $s $N $L > results/${D}_$N/log.txt 2>&1; echo "$N exit $?"
    N=stepB_dart_ft_tgt_${K}_p${s}_n$n; D=$(date +%Y%m%d); mkdir -p results/${D}_$N
    NUM_DEMOS=$n TOTAL_ITERS=1000 EVAL_EPISODES=250 SUPCON_WEIGHT=0 EXTRA="--init_ckpt $(ck $BASE) --batch_size 64 --eval_freq 1000" \
      bash scripts/run_dp_supcon.sh $TE $TH $s $N $L > results/${D}_$N/log.txt 2>&1; echo "$N exit $?"
    py scripts/stepB_dart.py ${K}_p${s}_n$n $BASE $(run stepB_dart_ft_src_${K}_p${s}_n$n) $(run stepB_dart_ft_tgt_${K}_p${s}_n$n) $TE \
      > results/$(date +%Y%m%d)_stepB_dart_${K}_p${s}_n${n}_log.txt 2>&1; echo "dart $K p$s n$n exit $?"
  done
done; done

# C: map-only fine-tuning at N = 1 (Step R phase C's recipe)
FT="--eval_freq 1000 --final_eval_episodes 250 --batch_size 64"
for s in 1 2 3; do c=$(nx $s); n=1
  for spec in "goal goal_look2 ${GC[$c]} goal_cam0 $EG $HG $VG" "same look2 ${CAM0[$c]} cam0 $E2 $H2 $V2"; do
    set -- $spec; K=$1; DOM=$2; CTRL=$3; D0=$4; ENVID=$5; H=$6; V=$7
    py scripts/step1b_stitch.py stepB_map_${K}_p${s}_n$n $DOM $CTRL ${L2[$s]} $L --domain0=$D0 --only-dir=10 --src-demos=$n \
      --maps=nn_affine_pca16 --offline-only --save-maps > results/$(date +%Y%m%d)_stepB_map_${K}_p${s}_n${n}_log.txt 2>&1; echo "map $K p$s n$n exit $?"
    MAP=$(run step1b_stitch_stepB_map_${K}_p${s}_n${n}_$DOM)/maps_${DOM}_to_${D0}.npz
    N=stepB_ft_map_${K}_p${s}_n$n; D=$(date +%Y%m%d); mkdir -p results/${D}_$N
    NUM_DEMOS=$n TOTAL_ITERS=5000 EVAL_EPISODES=100 SUPCON_WEIGHT=0 EXTRA="$FT --val_demo_path $V --init_encoder_ckpt $(ck ${L2[$s]}) \
      --init_controller_ckpt $(ck $CTRL) --init_map $MAP:nn_affine_pca16 --train_part map" \
      bash scripts/run_dp_supcon.sh $ENVID $H $s $N $L > results/${D}_$N/log.txt 2>&1; echo "$N exit $?"
  done
done
echo "queue2 done"
