#!/bin/bash
# diag_run.sh <tag> <deploy> <port> <seed>: one traced episode
source /root/autodl-tmp/run_env.sh
export TACGUIDE_TRACE=/root/autodl-tmp/trace_$1_$4
rm -rf $TACGUIDE_TRACE
python scripts/eval_policy.py grasp_chip grasp_chip openpi/abs_joint/$2 --openpi_port $3 --start_seed $4 --max_seed $4 --total_num 1 > /root/autodl-tmp/logs/eval/diag_$1_$4.log 2>&1
echo DIAG_EXIT=$? >> /root/autodl-tmp/logs/eval/diag_$1_$4.log
