#!/bin/bash
# nn_rescale vs nn_affine vs nn_orth on the 3 embodiment pairs (own-data motion pairs, 250 episodes), one job at a time
cd /home/ricc/projects/labelstitch-nb
R=$PWD
export PYTHONPATH=$R:$R/third_party/maniskill_diffusion_policy CUDA_VISIBLE_DEVICES=0
B=/home/ricc/projects/labelstitch-1b/results; S=/home/ricc/projects/labelstitch-r/results
MAPS=--maps=nn_affine_motion_own,nn_orth_motion_own,nn_rescale_motion_own
run() { uv run python -W ignore scripts/step5b_motion_stepR.py emb $1 $2 $3 $MAPS --tag=_rescale > results/20261015_emb_rescale_p$1_log.txt 2>&1; echo "p$1 exit $?"; }
run 1 $B/20261003_step1b_dp_ref_look2_s1 $B/20261004_step1b_dp_ref_xarm_cam0_100k_s2
run 2 $B/20261002_step1b_dp_ref_look2_s2 $S/20261004_stepR_dp_ref_xarm_cam0_s3
run 3 $B/20261003_step1b_dp_ref_look2_s3 $S/20261004_stepR_dp_ref_xarm_cam0_s1
