#!/bin/bash
# Run both one-episode link tests in parallel, keeping each run's first-observation dump separate.
source /root/autodl-tmp/run_env.sh
L=/root/autodl-tmp/logs
rm -rf debug/openpi_obs $L/dump_gcB $L/dump_gcA
python scripts/eval_policy.py grasp_chip grasp_chip openpi/abs_joint/deploy_grasp_chip_vision --openpi_port 8000 --start_seed 10000 --max_seed 10001 --total_num 1 > $L/14_eval_gcB_vision.log 2>&1 &
PB=$!
for i in $(seq 1 120); do [ -f debug/openpi_obs/000000_state.txt ] && break; sleep 2; done
sleep 2; mv debug/openpi_obs $L/dump_gcB
python scripts/eval_policy.py grasp_chip grasp_chip openpi/abs_joint/deploy_grasp_chip_vision_tactile --openpi_port 8001 --start_seed 10000 --max_seed 10001 --total_num 1 > $L/13_eval_gcA_tactile.log 2>&1 &
PA=$!
for i in $(seq 1 120); do [ -f debug/openpi_obs/000000_state.txt ] && break; sleep 2; done
sleep 2; cp -r debug/openpi_obs $L/dump_gcA
wait $PB; echo "EVAL_EXIT=$?" >> $L/14_eval_gcB_vision.log
wait $PA; echo "EVAL_EXIT=$?" >> $L/13_eval_gcA_tactile.log
echo LINK_TEST_DONE > $L/15_link_test_done
