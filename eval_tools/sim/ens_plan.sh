#!/bin/bash
# ens_plan.sh <A|B>: ensemble on the previously failed eval seeds first, then training seeds, then the rest, then the h50 control
if [ "$1" = A ]; then T=ens_A; D=ens_grasp_chip_vision_tactile; P=8001; H=diag_tactile_h50; else T=ens_B; D=ens_grasp_chip_vision; P=8000; H=diag_vision_h50; fi
R=/root/autodl-tmp/run_seeds.sh
bash $R $T $D $P 10000 10000
bash $R $T $D $P 10004 10004
bash $R $T $D $P 10001 10001
echo "[$(date -u +%F_%T)] PHASE1_DONE" >> /root/autodl-tmp/logs/eval/$T.log
bash $R $T $D $P 0 1
bash $R $T $D $P 10002 10003
echo "[$(date -u +%F_%T)] PHASE2_DONE" >> /root/autodl-tmp/logs/eval/$T.log
bash $R h50_$1 $H $P 10000 10000
bash $R h50_$1 $H $P 10004 10004
bash $R h50_$1 $H $P 0 1
bash $R h50_$1 $H $P 10001 10003
echo "[$(date -u +%F_%T)] PHASE3_DONE" >> /root/autodl-tmp/logs/eval/$T.log
