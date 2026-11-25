#!/bin/bash
# Stop the b2_B loop after its current attempt, then run CFG(A,B) w=2 (port 8004), then w=1.5 (8003), then w=3 (8005).
L=/root/autodl-tmp/logs/eval/b2_B.log
n0=$(grep -c "result=" $L)
until [ "$(grep -c 'result=' $L)" -gt "$n0" ] || [ "$(grep -c ALL_DONE $L)" -ge 2 ]; do sleep 2; done
tmux kill-session -t cB 2>/dev/null
sleep 3
for p in $(ps -eo pid,args | grep "[s]cripts/eval_policy.py" | grep -- "--openpi_port 8000" | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 6
for p in $(ps -eo pid,args | grep "[s]cripts/eval_policy.py" | grep -- "--openpi_port 8000" | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
echo "[$(date -u +%F_%T)] STOPPED_BY_SWITCH b2_B" >> $L
source /root/autodl-tmp/run_env.sh
for spec in "2:8004" "1.5:8003" "3:8005"; do
  w=${spec%%:*}; port=${spec##*:}
  python /root/autodl-tmp/wait_port.py $port > /root/autodl-tmp/logs/eval/cfg_w$w.server
  bash /root/autodl-tmp/run_seeds.sh cfg_w$w deploy_grasp_chip_vision_tactile $port 0 1
  bash /root/autodl-tmp/run_seeds.sh cfg_w$w deploy_grasp_chip_vision_tactile $port 10000 10004
done
