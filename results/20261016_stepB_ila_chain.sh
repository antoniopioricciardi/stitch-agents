#!/bin/bash
# Step B: ILA on cam0 -> look 2, pairs 1-3 (EXPERIMENTS.md, 2026-10-16 ILA entry). Starts after unit stepb-dart5k.
cd /home/ricc/projects/labelstitch-b
while systemctl --user is-active --quiet stepb-dart5k; do sleep 300; done
export PYTHONPATH=$PWD:$PWD/third_party/maniskill_diffusion_policy CUDA_VISIBLE_DEVICES=0
for p in p1 p2 p3; do
  uv run python -W ignore scripts/stepB_ila.py $p > results/$(date +%Y%m%d)_stepB_ila_${p}_log.txt 2>&1; echo "ila $p exit $?"
done
echo "ila done"
