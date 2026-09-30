#!/bin/bash
# ManiSkill's Diffusion Policy RGB baseline (third_party/maniskill_diffusion_policy, unmodified) with the settings of
# run_dp_reference.sh, for any env id / demo file / seed. From Step 2e on: 250 eval episodes per checkpoint
# (Step 2c ran with the default 100).
# Usage (from the repo root): bash scripts/run_dp.sh <env-id> <demo-h5> <seed> <out-name>
#   env-id: PickCube-v1 or stitch.envs:StitchPickCube-v1 (module:EnvId so eval workers import stitch.envs)
set -e
ENV_ID=$1; DEMO=$2; SEED=$3; NAME=$4
ROOT=$(pwd)
OUT=$ROOT/results/$(date +%Y%m%d)_$NAME
mkdir -p $OUT
cd $OUT
PYTHONPATH=$ROOT:$ROOT/third_party/maniskill_diffusion_policy WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
uv run --project $ROOT python $ROOT/third_party/maniskill_diffusion_policy/train_rgbd.py --env-id $ENV_ID \
  --demo-path $DEMO \
  --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos 100 --max_episode_steps 100 \
  --total_iters 30000 --obs-mode "rgb" \
  --exp-name $NAME \
  --num_eval_episodes 250 --demo_type=motionplanning --track --seed $SEED
