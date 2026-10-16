#!/bin/bash
# Step B: centred PeS (EXPERIMENTS.md, 2026-10-16 PeS entry, decision rule of the diagnosis). Runs only if the
# diagnosis agent stepB_diag_pes_centred_cam0_s1 reaches last-3 success_once >= .3; it starts after unit stepb-queue2.
# Same plan as results/20261016_stepB_pes_chain.sh with --encoder pes_centred and run names stepB_pesc_k<K>_<domain>_s<seed>;
# cam0 seed 1 (K = 256) is the diagnosis run itself (linked, then its three checks).
cd /home/ricc/projects/labelstitch-b
while systemctl --user is-active --quiet stepb-queue2; do sleep 300; done
DIAG=$(ls -d results/2026*_stepB_diag_pes_centred_cam0_s1 | tail -1)
PASS=$(uv run python -c "import re, numpy as np; s = [float(x) for x in re.findall(r'^success_once: ([0-9.]+)', open('$DIAG/log.txt').read().replace(chr(13), chr(10)), re.M)]; print(int(len(s) >= 11 and np.mean(s[-3:]) >= .3))")
echo "centred diagnosis passes the bar: $PASS"
[ "$PASS" = 1 ] || { echo "rule: PeS training dropped"; exit 0; }
ln -s $(realpath $DIAG) results/$(date +%Y%m%d)_stepB_pesc_k256_cam0_s1
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
  local DOM=$1 SEED=$2 K=$3 N=stepB_pesc_k$3_$1_s$2 D=$(date +%Y%m%d)
  local AH=${H5[$1]}; [ $DOM = goal_cam0 ] && AH=$H0  # the goal variant renders default-task states identically
  mkdir -p results/${D}_$N
  SUPCON_WEIGHT=0 EXTRA="--encoder pes_centred --pes_anchors $A/anchors_K$K.npz --pes_anchor_h5 $AH" \
    bash scripts/run_dp_supcon.sh stitch.envs:${ENVID[$DOM]} ${H5[$DOM]} $SEED $N $L > results/${D}_$N/log.txt 2>&1; echo "$N exit $?"
  [ "$4" = checks ] || return
  local RUN=$(run $N)
  py scripts/probe_cube_goal.py $RUN $DOM > $RUN/probe_log.txt 2>&1; echo "$N probe exit $?"
  py scripts/eval_dp_blind.py $RUN stitch.envs:${ENVID[$DOM]} ${H5[$DOM]} > $RUN/blind_log.txt 2>&1; echo "$N blind exit $?"
  py scripts/fake_goal_check.py $RUN $DOM > $RUN/fake_goal_log.txt 2>&1; echo "$N fake goal exit $?"
}
matrix() {  # shift K
  for p in 1 2 3; do
    py scripts/stepB_pes_stitch.py matrix $1 p$p $(run stepB_pesc_k$2_cam0_s$p) $(run stepB_pesc_k$2_$1_s$(nx $p)) --tag=_centred_k$2 \
      > results/$(date +%Y%m%d)_stepB_pes_matrix_$1_p${p}_centred_k$2_log.txt 2>&1; echo "matrix $1 p$p k$2 exit $?"
  done
}

RUN=$(run stepB_pesc_k256_cam0_s1)
py scripts/probe_cube_goal.py $RUN cam0 > $RUN/probe_log.txt 2>&1; echo "cam0 s1 probe exit $?"
py scripts/eval_dp_blind.py $RUN stitch.envs:${ENVID[cam0]} ${H5[cam0]} > $RUN/blind_log.txt 2>&1; echo "cam0 s1 blind exit $?"
py scripts/fake_goal_check.py $RUN cam0 > $RUN/fake_goal_log.txt 2>&1; echo "cam0 s1 fake goal exit $?"
for s in 2 3; do train cam0 $s 256 checks; done
for s in 1 2 3; do train look2 $s 256 checks; done
matrix look2 256
for s in 1 2 3; do train goal_cam0 $s 256 checks; done
for s in 1 2 3; do
  py scripts/stepB_pes_stitch.py goal p$s $(run stepB_pesc_k256_look2_s$s) $(run stepB_pesc_k256_goal_cam0_s$(nx $s)) --tag=_centred_k256 \
    > results/$(date +%Y%m%d)_stepB_pes_goal_p${s}_centred_k256_log.txt 2>&1; echo "goal p$s exit $?"
done
for d in look1 look2light1 cam3; do
  for s in 1 2 3; do train $d $s 256 checks; done
  matrix $d 256
done
for d in cam0 look2; do for s in 1 2 3; do train $d $s 64; done; done
matrix look2 64
echo "chain done"
