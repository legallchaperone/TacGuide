#!/bin/bash
# replay_run.sh <seed>: run eval seed against the replay server on 8010 with tracing
source /root/autodl-tmp/run_env.sh
export TACGUIDE_TRACE=/root/autodl-tmp/trace_replay_$1
rm -rf $TACGUIDE_TRACE
python scripts/eval_policy.py grasp_chip grasp_chip openpi/abs_joint/diag_replay --openpi_port 8010 --start_seed $1 --max_seed $1 --total_num 1 > /root/autodl-tmp/logs/eval/replay_$1.log 2>&1
echo DIAG_EXIT=$? >> /root/autodl-tmp/logs/eval/replay_$1.log
