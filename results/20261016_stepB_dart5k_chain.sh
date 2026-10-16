#!/bin/bash
# Step B: DART with map-only's fine-tuning budget (5000 iterations, batch 256), N in {5, 10}, same task and goal shift,
# 3 seed pairs s -> s+1 (EXPERIMENTS.md, 2026-10-16 DART-5k entry). As results/20261016_stepB_queue2.sh part B
# otherwise. One job at a time.
cd /home/ricc/projects/labelstitch-b
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
CAM0=([1]=/home/ricc/projects/labelstitch-step1/results/20261001_step2g_dp_core_nograsp_50k_s1 [2]=$B/20261002_step1b_dp_ref_cam0_s2 [3]=$B/20261002_step1b_dp_ref_cam0_s3)
GC=([1]=$RR/20261004_stepR_dp_ref_goal_cam0_s1 [2]=$B/20261003_step1b_dp_ref_goal_cam0_s2 [3]=$RR/20261004_stepR_dp_ref_goal_cam0_s3)
H0=/home/ricc/projects/labelstitch-step1/results/20260930_dp_ours_demos_default_panda_cam0_lollipop/$T
HGC=$B/20261003_dp_ours_demos_goal_panda_cam0_lollipop/$T
H2=$RR/20261002_dp_ours_demos_default_panda_cam0_look2_lollipop/$T
HG=$RR/20261003_dp_ours_demos_goal_panda_cam0_look2_lollipop/$T
E0=stitch.envs:StitchPickCubeLollipopNoGrasp-v1
EGC=stitch.envs:StitchPickCubeLollipopNoGraspGoal-v1
E2=stitch.envs:StitchPickCubeLollipopNoGraspLook2-v1
EG=stitch.envs:StitchPickCubeLollipopNoGraspLook2Goal-v1

for s in 1 2 3; do c=$(nx $s); for n in 5 10; do
  # combo  base(controller-side oracle)  source env  source demos  target env  target demos
  for spec in "same ${CAM0[$c]} $E0 $H0 $E2 $H2" "goal ${GC[$c]} $EGC $HGC $EG $HG"; do
    set -- $spec; K=$1; BASE=$2; SE=$3; SH=$4; TE=$5; TH=$6
    N=stepB_dart5k_ft_src_${K}_p${s}_n$n; D=$(date +%Y%m%d); mkdir -p results/${D}_$N
    NUM_DEMOS=$n TOTAL_ITERS=5000 EVAL_EPISODES=10 SUPCON_WEIGHT=0 EXTRA="--init_ckpt $(ck $BASE) --eval_freq 100000" \
      bash scripts/run_dp_supcon.sh $SE $SH $s $N $L > results/${D}_$N/log.txt 2>&1; echo "$N exit $?"
    N=stepB_dart5k_ft_tgt_${K}_p${s}_n$n; D=$(date +%Y%m%d); mkdir -p results/${D}_$N
    NUM_DEMOS=$n TOTAL_ITERS=5000 EVAL_EPISODES=250 SUPCON_WEIGHT=0 EXTRA="--init_ckpt $(ck $BASE) --eval_freq 5000" \
      bash scripts/run_dp_supcon.sh $TE $TH $s $N $L > results/${D}_$N/log.txt 2>&1; echo "$N exit $?"
    py scripts/stepB_dart.py 5k_${K}_p${s}_n$n $BASE $(run stepB_dart5k_ft_src_${K}_p${s}_n$n) $(run stepB_dart5k_ft_tgt_${K}_p${s}_n$n) $TE \
      > results/$(date +%Y%m%d)_stepB_dart5k_${K}_p${s}_n${n}_log.txt 2>&1; echo "dart5k $K p$s n$n exit $?"
  done
done; done
echo "dart5k done"
