#!/bin/bash
# after_ens.sh <ens tag> <new tag> <deploy> <port>: wait until the ensemble run has finished both seed ranges, then run the control
until [ "$(grep -c ALL_DONE /root/autodl-tmp/logs/eval/$1.log 2>/dev/null)" -ge 2 ]; do sleep 20; done
bash /root/autodl-tmp/run_seeds.sh $2 $3 $4 0 1
bash /root/autodl-tmp/run_seeds.sh $2 $3 $4 10000 10004
