#!/bin/bash
# Usage: run_eval.sh <tag> <deploy_config_stem> <port> [start_seed] [end_seed]
# Runs grasp_chip evaluation for seeds start..end (default 10000..10049) against the policy server on <port>.
# The seed range is closed so an errored seed is never replaced by another seed.
TAG=$1; DEPLOY=$2; PORT=$3; S=${4:-10000}; E=${5:-10049}
source /root/autodl-tmp/run_env.sh
L=/root/autodl-tmp/logs/eval; mkdir -p $L
N=$((E - S + 1))
echo "[$(date -u +%F_%T)] start $TAG deploy=$DEPLOY port=$PORT seeds=$S..$E" >> $L/$TAG.log
python scripts/eval_policy.py grasp_chip grasp_chip openpi/abs_joint/$DEPLOY --openpi_port $PORT \
  --start_seed $S --max_seed $E --total_num $N >> $L/$TAG.log 2>&1
echo "[$(date -u +%F_%T)] EVAL_EXIT=$? $TAG" >> $L/$TAG.log
