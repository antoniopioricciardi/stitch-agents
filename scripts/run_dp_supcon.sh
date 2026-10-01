#!/bin/bash
# Step 1b: run_dp.sh's settings with scripts/train_dp_supcon.py (DP + SupCon on z with action-chunk labels;
# SUPCON_WEIGHT=0 trains plain DP). Checkpoints: 40k / 45k / final (49999) for 50k iterations.
# Usage (from the repo root): bash scripts/run_dp_supcon.sh <env-id> <demo-h5> <seed> <out-name> <labels.pt>
#   env vars: SUPCON_WEIGHT (default 1.0), TOTAL_ITERS (50000), NUM_DEMOS (100), EVAL_EPISODES (250), LOG_FREQ (1000)
set -e
ENV_ID=$1; DEMO=$2; SEED=$3; NAME=$4; LABELS=$5
ROOT=$(pwd)
OUT=$ROOT/results/$(date +%Y%m%d)_$NAME
mkdir -p $OUT
cd $OUT
PYTHONPATH=$ROOT:$ROOT/third_party/maniskill_diffusion_policy WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
uv run --project $ROOT python $ROOT/scripts/train_dp_supcon.py --env-id $ENV_ID \
  --demo-path $DEMO \
  --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${NUM_DEMOS:-100} --max_episode_steps 100 \
  --total_iters ${TOTAL_ITERS:-50000} --obs-mode "rgb" \
  --exp-name $NAME \
  --num_eval_episodes ${EVAL_EPISODES:-250} --log_freq ${LOG_FREQ:-1000} --demo_type=motionplanning --track --seed $SEED \
  --labels $LABELS --supcon_weight ${SUPCON_WEIGHT:-1.0}
