#!/bin/bash
# ManiSkill's Diffusion Policy RGB baseline, unmodified, on our env: StitchPickCube-v1 (cam0, default look/light/task,
# Panda) with our demos exported by scripts/export_dp_demos.py. Same command and settings as run_dp_reference.sh.
# --env-id uses gymnasium's "module:EnvId" form, so the forkserver evaluation workers import stitch.envs (which
# registers the env); the repo root is on PYTHONPATH for that.
# Usage: bash scripts/run_dp_ours.sh   (from the repo root)
set -e
ROOT=$(pwd)
OUT=$ROOT/results/$(date +%Y%m%d)_dp_ours_default_panda_cam0
DEMO=$ROOT/results/20260930_dp_ours_demos_default_panda_cam0/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5
mkdir -p $OUT
cd $OUT
PYTHONPATH=$ROOT:$ROOT/third_party/maniskill_diffusion_policy WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
uv run --project $ROOT python $ROOT/third_party/maniskill_diffusion_policy/train_rgbd.py --env-id stitch.envs:StitchPickCube-v1 \
  --demo-path $DEMO \
  --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos 100 --max_episode_steps 100 \
  --total_iters 30000 --obs-mode "rgb" \
  --exp-name diffusion_policy-StitchPickCube-v1-cam0-rgb-100_motionplanning_demos-1 \
  --demo_type=motionplanning --track --seed 1
