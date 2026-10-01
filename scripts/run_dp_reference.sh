#!/bin/bash
# Reproduce ManiSkill's Diffusion Policy RGB baseline on PickCube-v1, unmodified (third_party/maniskill_diffusion_policy,
# tag v3.0.1). Command = baselines.sh (RGB, 100 motion-planning demos) with --seed 1, the seed of the reference run
# (wandb stonet2000/ManiSkill, run 2j97yoou: success_once 0.81, success_at_end 0.67 at 30k iterations).
# Demos: python -m mani_skill.utils.download_demo PickCube-v1, then replay_for_il_baselines.sh's RGB line:
#   python -m mani_skill.trajectory.replay_trajectory --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
#     --use-first-env-state -c pd_ee_delta_pos -o rgb --save-traj --num-envs 10 -b physx_cpu
# wandb runs offline (no login on this machine); upload later with `wandb sync`.
# Usage: bash scripts/run_dp_reference.sh   (from the repo root)
set -e
ROOT=$(pwd)
OUT=$ROOT/results/$(date +%Y%m%d)_dp_reference_pickcube_rgb
mkdir -p $OUT
cd $OUT  # the baseline writes runs/<exp-name>/ and wandb/ relative to the working directory
PYTHONPATH=$ROOT/third_party/maniskill_diffusion_policy WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
uv run --project $ROOT python $ROOT/third_party/maniskill_diffusion_policy/train_rgbd.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos 100 --max_episode_steps 100 \
  --total_iters 30000 --obs-mode "rgb" \
  --exp-name diffusion_policy-PickCube-v1-rgb-100_motionplanning_demos-1 \
  --demo_type=motionplanning --track --seed 1
