#!/bin/bash
# expert_run.sh <seed>: scripted expert (collect mode) on one seed, output under data_expert/
source /root/autodl-tmp/run_env.sh
sed "s#^save_dir: ./data#save_dir: ./data_expert_$1#" task_config/grasp_chip.yml > task_config/grasp_chip_expert_$1.yml
python scripts/collect_data.py grasp_chip grasp_chip_expert_$1 --start_seed $1 --max_seed $1 --episode_num 1 --gpu 0 > /root/autodl-tmp/logs/eval/expert_$1.log 2>&1
echo DIAG_EXIT=$? >> /root/autodl-tmp/logs/eval/expert_$1.log
