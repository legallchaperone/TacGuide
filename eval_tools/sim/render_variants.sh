#!/bin/bash
# Render frame 0 of the seed-0 replay under different renderer settings; saves the 224x224 images the client would send.
source /root/autodl-tmp/run_env.sh
OUT=/root/autodl-tmp/render_variants; mkdir -p $OUT
L=/root/autodl-tmp/logs/eval
pgrep -f replay_server.py >/dev/null || (nohup python /root/autodl-tmp/replay_server.py /root/autodl-tmp/demo_ep0-2.npz 0 8010 > $L/replay_server2.log 2>&1 &)
sleep 5
run() {
  name=$1; shift
  [ -f $OUT/$name/000000_image_sent.png ] && return
  rm -rf debug/openpi_obs
  python scripts/eval_policy.py grasp_chip grasp_chip openpi/abs_joint/diag_replay --openpi_port 8010 \
     --start_seed 0 --max_seed 0 --total_num 1 --max_policy_actions 1 "$@" > $L/variant_$name.log 2>&1
  mkdir -p $OUT/$name; cp debug/openpi_obs/* $OUT/$name/ 2>/dev/null
  echo "$name :: $*" >> $OUT/SETTINGS.txt
  grep -c "Failed to create NGX" $(ls -td eval_result/openpi/grasp_chip/diag_replay/*/ | head -1)attempts/0/launch.log >> $OUT/SETTINGS.txt
  echo "[$(date -u +%T)] done $name files=$(ls $OUT/$name | wc -l)"
}
if [ "$1" = "base" ]; then
run v00_default
run v01_mode_performance --rendering_mode performance
run v02_mode_quality --rendering_mode quality
run v03_aa_none --kit_args="--/rtx/post/aa/op=0"
run v04_aa_taa --kit_args="--/rtx/post/aa/op=1"
run v05_aa_fxaa --kit_args="--/rtx/post/aa/op=2"
run v06_aa_dlss --kit_args="--/rtx/post/aa/op=3"
run v07_aa_dlaa --kit_args="--/rtx/post/aa/op=4"
run v08_no_dldenoiser --kit_args="--/rtx-transient/dldenoiser/enabled=false"
run v09_no_tonemap --kit_args="--/rtx/post/tonemap/op=0"
else
  "$@"
fi
echo VARIANTS_DONE2 >> $OUT/SETTINGS.txt
