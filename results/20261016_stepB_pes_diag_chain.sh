#!/bin/bash
# Step B, PeS diagnosis (EXPERIMENTS.md, 2026-10-16 PeS entry). Waits for the PeS chain's current training
# (look 2 seed 3) to exit, stops that chain, then trains on cam0 seed 1 (50k, run_dp.sh settings), one at a time:
# (a) l2norm: controller reads the L2-normalised PlainConv output only; (b) pes_centred: PeS with centring on the anchors.
cd /home/ricc/projects/labelstitch-b
until journalctl --user -u stepb-pes --no-pager | grep -q "stepB_pes_k256_look2_s3 exit"; do sleep 60; done
systemctl --user stop stepb-pes
R=$PWD
B=/home/ricc/projects/labelstitch-1b/results
export PYTHONPATH=$R:$R/third_party/maniskill_diffusion_policy CUDA_VISIBLE_DEVICES=0
L=$B/20261001_step1b_labels/labels.pt
H0=/home/ricc/projects/labelstitch-step1/results/20260930_dp_ours_demos_default_panda_cam0_lollipop/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5
A=$R/results/20261016_stepB_pes_anchors
for spec in "l2norm --encoder l2norm" "pes_centred --encoder pes_centred --pes_anchors $A/anchors_K256.npz --pes_anchor_h5 $H0"; do
  set -- $spec; V=$1; shift
  N=stepB_diag_${V}_cam0_s1; D=$(date +%Y%m%d); mkdir -p results/${D}_$N
  SUPCON_WEIGHT=0 EXTRA="$*" bash scripts/run_dp_supcon.sh stitch.envs:StitchPickCubeLollipopNoGrasp-v1 $H0 1 $N $L > results/${D}_$N/log.txt 2>&1; echo "$N exit $?"
done
echo "diag done"
