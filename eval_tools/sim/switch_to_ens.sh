#!/bin/bash
# Stop a running run_seeds loop right after its current attempt ends, then start the ensemble run in its place.
# usage: switch_to_ens.sh <old tmux session> <old tag> <new tag> <deploy> <port>
OLD=$1; OLDTAG=$2; NEW=$3; DEPLOY=$4; PORT=$5
L=/root/autodl-tmp/logs/eval/$OLDTAG.log
n0=$(grep -c "result=" $L)
until [ "$(grep -c 'result=' $L)" -gt "$n0" ] || grep -q ALL_DONE $L; do sleep 2; done
tmux kill-session -t $OLD 2>/dev/null
sleep 3
for p in $(ps -eo pid,args | grep "[s]cripts/eval_policy.py" | grep -- "--openpi_port $PORT" | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 5
for p in $(ps -eo pid,args | grep "[s]cripts/eval_policy.py" | grep -- "--openpi_port $PORT" | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
echo "[$(date -u +%F_%T)] STOPPED_BY_SWITCH $OLDTAG" >> $L
bash /root/autodl-tmp/run_seeds.sh $NEW $DEPLOY $PORT 0 1
bash /root/autodl-tmp/run_seeds.sh $NEW $DEPLOY $PORT 10000 10004
