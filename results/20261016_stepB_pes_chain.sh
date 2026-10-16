#!/bin/bash
# Step B: Perception Stitching (PeS) agents and zero-shot stitches (EXPERIMENTS.md, 2026-10-16). One job at a time.
# Trainings: PeS (K = 256 anchors) in cam0, look 2, cam0 goal variant, look 1, look 2 + light 1, cam3, seeds 1-3, 50k
# iterations, 100 demos (run_dp.sh settings, plain DP loss + 0.002 x disentanglement); three checks each (probe, blind,
# fake goal). Then K = 64 in cam0 and look 2, seeds 1-3 (native success only).
# Stitches (no map), as soon as both agents exist: matrix pair p = cam0 seed p <-> shift seed p % 3 + 1 (fwd and bwd);
# goal pair s = look 2 seed s -> goal cam0 seed s % 3 + 1. No gate: a failing agent's cells are flagged in the table.
cd /home/ricc/projects/labelstitch-b
R=$PWD
B=/home/ricc/projects/labelstitch-1b/results
export PYTHONPATH=$R:$R/third_party/maniskill_diffusion_policy CUDA_VISIBLE_DEVICES=0
L=$B/20261001_step1b_labels/labels.pt  # required by the trainer, SupCon is only logged (weight 0)
T=trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5
A=$R/results/20261016_stepB_pes_anchors
H0=/home/ricc/projects/labelstitch-step1/results/20260930_dp_ours_demos_default_panda_cam0_lollipop/$T  # cam0 default: the anchor states
py() { uv run python -W ignore "$@"; }
run() { ls -d $R/results/2026*_$1 | tail -1; }
nx() { echo $(( $1 % 3 + 1 )); }
declare -A ENVID=([cam0]=StitchPickCubeLollipopNoGrasp-v1 [look1]=StitchPickCubeLollipopNoGraspLook1-v1
  [look2]=StitchPickCubeLollipopNoGraspLook2-v1 [look2light1]=StitchPickCubeLollipopNoGraspLook2Light1-v1
  [cam3]=StitchPickCubeLollipopNoGraspCam3-v1 [goal_cam0]=StitchPickCubeLollipopNoGraspGoal-v1)
declare -A H5=([cam0]=$H0 [look1]=$B/20261001_dp_ours_demos_default_panda_cam0_look1_lollipop/$T
  [look2]=$B/20261002_dp_ours_demos_default_panda_cam0_look2_lollipop/$T
  [look2light1]=$B/20261002_dp_ours_demos_default_panda_cam0_look2_light1_lollipop/$T
  [cam3]=$B/20261002_dp_ours_demos_default_panda_cam3_lollipop/$T [goal_cam0]=$B/20261003_dp_ours_demos_goal_panda_cam0_lollipop/$T)

train() {  # domain seed K [checks]
  local DOM=$1 SEED=$2 K=$3 N=stepB_pes_k$3_$1_s$2 D=$(date +%Y%m%d)
  local AH=${H5[$1]}; [ $DOM = goal_cam0 ] && AH=$H0  # the goal variant renders default-task states identically
  mkdir -p results/${D}_$N
  SUPCON_WEIGHT=0 EXTRA="--encoder pes --pes_anchors $A/anchors_K$K.npz --pes_anchor_h5 $AH" \
    bash scripts/run_dp_supcon.sh stitch.envs:${ENVID[$DOM]} ${H5[$DOM]} $SEED $N $L > results/${D}_$N/log.txt 2>&1; echo "$N exit $?"
  [ "$4" = checks ] || return
  local RUN=$(run $N)
  py scripts/probe_cube_goal.py $RUN $DOM > $RUN/probe_log.txt 2>&1; echo "$N probe exit $?"
  py scripts/eval_dp_blind.py $RUN stitch.envs:${ENVID[$DOM]} ${H5[$DOM]} > $RUN/blind_log.txt 2>&1; echo "$N blind exit $?"
  py scripts/fake_goal_check.py $RUN $DOM > $RUN/fake_goal_log.txt 2>&1; echo "$N fake goal exit $?"
}
matrix() {  # shift K
  for p in 1 2 3; do
    py scripts/stepB_pes_stitch.py matrix $1 p$p $(run stepB_pes_k$2_cam0_s$p) $(run stepB_pes_k$2_$1_s$(nx $p)) --tag=_k$2 \
      > results/$(date +%Y%m%d)_stepB_pes_matrix_$1_p${p}_k$2_log.txt 2>&1; echo "matrix $1 p$p k$2 exit $?"
  done
}

for s in 1 2 3; do train cam0 $s 256 checks; done
for s in 1 2 3; do train look2 $s 256 checks; done
matrix look2 256
for s in 1 2 3; do train goal_cam0 $s 256 checks; done
for s in 1 2 3; do
  py scripts/stepB_pes_stitch.py goal p$s $(run stepB_pes_k256_look2_s$s) $(run stepB_pes_k256_goal_cam0_s$(nx $s)) --tag=_k256 \
    > results/$(date +%Y%m%d)_stepB_pes_goal_p${s}_k256_log.txt 2>&1; echo "goal p$s exit $?"
done
for d in look1 look2light1 cam3; do
  for s in 1 2 3; do train $d $s 256 checks; done
  matrix $d 256
done
for d in cam0 look2; do for s in 1 2 3; do train $d $s 64; done; done
matrix look2 64
echo "chain done"
