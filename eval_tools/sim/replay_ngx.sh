#!/bin/bash
# seed-0 demonstration replay with DLSS enabled; traces sent images each query and native frames at steps 0/20/40/60/100
source /root/autodl-tmp/run_env.sh
OUT=/root/autodl-tmp/render_static/default_with_ngx_replay; rm -rf $OUT; mkdir -p $OUT
for p in $(ps -eo pid,args | grep "[r]eplay_server.py" | awk "{print \$1}"); do kill $p; done
nohup python /root/autodl-tmp/replay_server.py /root/autodl-tmp/demo_ep0-2.npz 0 8010 > $OUT/replay_server.log 2>&1 &
sleep 5
export TACGUIDE_RAW=$OUT/raw TACGUIDE_RAW_PER_STEP=2 TACGUIDE_TRACE=$OUT/sent
python scripts/eval_policy.py grasp_chip grasp_chip openpi/abs_joint/diag_replay --openpi_port 8010 --start_seed 0 --max_seed 0 --total_num 1 > $OUT/run.log 2>&1
L=$(ls -td eval_result/openpi/grasp_chip/diag_replay/*/ | head -1)
tr "\r" "\n" < ${L}attempts/0/launch.log | grep -i "ngx\|dlss" | cut -c1-220 > $OUT/NGX_LOG_LINES.txt
cp ${L}attempts/0/result.json ${L}metadata.json $OUT/ 2>/dev/null; cp ${L}video/*.mp4 $OUT/ 2>/dev/null
echo STATIC_DONE > $OUT/DONE
