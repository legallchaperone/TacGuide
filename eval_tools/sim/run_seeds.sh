#!/bin/bash
# run_seeds.sh <tag> <deploy_config_stem> <port> <first_seed> <last_seed>
# One invocation per seed. Each episode is checked at step 0 (wrist brightness, tactile marker count); a broken
# simulator start-up or a crashed/timed-out worker is retried with the SAME seed (up to 4 attempts), so every
# config is scored on the same seeds. Per-seed traces and START_CHECK.json go to trace/<tag>/<seed>/try<k>/.
TAG=$1; DEPLOY=$2; PORT=$3; S=$4; E=$5
source /root/autodl-tmp/run_env.sh
L=/root/autodl-tmp/logs/eval; mkdir -p $L
R=/root/autodl-tmp/results/$TAG; mkdir -p $R
export TACGUIDE_VALIDATE=1 TACGUIDE_TACTILE_REF=/root/autodl-tmp/tactile_rest_reference_left_224.npy
for seed in $(seq $S $E); do
  [ -f $R/$seed.json ] && continue
  for try in 1 2 3 4; do
    export TACGUIDE_TRACE=/root/autodl-tmp/trace/$TAG/$seed/try$try
    rm -rf $TACGUIDE_TRACE
    echo "[$(date -u +%F_%T)] start $TAG seed=$seed try=$try deploy=$DEPLOY port=$PORT" >> $L/$TAG.log
    python scripts/eval_policy.py grasp_chip grasp_chip openpi/abs_joint/$DEPLOY --openpi_port $PORT \
      --start_seed $seed --max_seed $seed --total_num 1 > $L/$TAG.last 2>&1
    OUT=$(grep -o '"output": "[^"]*"' $L/$TAG.last | tail -1 | cut -d'"' -f4)
    RES=$OUT/attempts/$seed/result.json
    status=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('result','error'))" $RES 2>/dev/null || echo error)
    echo "[$(date -u +%F_%T)] seed=$seed try=$try result=$status out=$OUT" >> $L/$TAG.log
    if [ "$status" = "success" ] || [ "$status" = "fail" ]; then
      python3 - "$RES" "$OUT" "$try" "$TACGUIDE_TRACE" > $R/$seed.json <<'PY'
import json, sys, os
r = json.load(open(sys.argv[1])); r["output"] = sys.argv[2]; r["tries"] = int(sys.argv[3])
c = os.path.join(sys.argv[4], "START_CHECK.json")
r["start_check"] = json.load(open(c)) if os.path.exists(c) else None
print(json.dumps(r))
PY
      break
    fi
    mkdir -p $R/rejected; cp $RES $R/rejected/${seed}_try$try.json 2>/dev/null
    cp $TACGUIDE_TRACE/START_CHECK.json $R/rejected/${seed}_try${try}_start_check.json 2>/dev/null
  done
done
echo "[$(date -u +%F_%T)] ALL_DONE $TAG" >> $L/$TAG.log
