#!/bin/bash
# static_run.sh <name> [extra eval args...]: seed-0 scene, robot holds its initial pose; saves native-resolution
# camera frames at steps 0/20/40/60 and the 224x224 images actually sent (trace) under render_static/<name>/
name=$1; shift
source /root/autodl-tmp/run_env.sh
[ -n "$NGX_LIB" ] && export LD_LIBRARY_PATH=$NGX_LIB:$LD_LIBRARY_PATH
OUT=/root/autodl-tmp/render_static/$name; rm -rf $OUT; mkdir -p $OUT
for p in $(ps -eo pid,args | grep "[r]eplay_server.py" | awk '{print $1}'); do kill $p; done
HOLD=1 nohup python /root/autodl-tmp/replay_server.py /root/autodl-tmp/demo_ep0-2.npz 0 8010 > $OUT/replay_server.log 2>&1 &
sleep 5
export TACGUIDE_RAW=$OUT/raw TACGUIDE_RAW_PER_STEP=2 TACGUIDE_TRACE=$OUT/sent
python scripts/eval_policy.py grasp_chip grasp_chip openpi/abs_joint/diag_replay --openpi_port 8010 \
   --start_seed 0 --max_seed 0 --total_num 1 --max_policy_actions 61 "$@" > $OUT/run.log 2>&1
echo "args: $*" > $OUT/SETTINGS.txt
grep -h "NGX\|DLSS" $(ls -td eval_result/openpi/grasp_chip/diag_replay/*/ | head -1)attempts/0/launch.log | cut -c1-200 >> $OUT/SETTINGS.txt
ls $OUT/raw $OUT/sent >> $OUT/SETTINGS.txt
echo STATIC_DONE >> $OUT/SETTINGS.txt
