#!/usr/bin/env bash
# Keep gc_A and gc_B training until step $TG_STEPS, on the training machine, inside tmux:
#
#   tmux new-session -d -s watch_ab "training/watch_ab.sh gc_A:7 gc_B:5"
#
# Every 5 minutes it writes $TG/logs/watch_ab.status (one line per run). If a run's tmux
# session is gone before its final checkpoint exists, it is resumed from the last checkpoint
# on the same card, but only when no other TacGuide process is on that card and at least
# 54 GB are free; otherwise the status line says WAITING. A run is resumed at most 5 times.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/training/tacguide_env.sh"
STEPS="${TG_STEPS:-10000}"
declare -A tries
model_dir() { case "$1" in gc_A) echo openvtla ;; gc_B) echo pi05 ;; esac; }
last_step() { tr '\r' '\n' < "$TG/logs/train_$1.log" 2>/dev/null | grep -o 'step=[0-9]*' | tail -1 | cut -d= -f2; }
while true; do
  out=""; all_done=1
  for spec in "$@"; do
    exp="${spec%%:*}"; gpu="${spec##*:}"; dir="$TG/outputs/training/$(model_dir "$exp")/gc/$exp"
    if [ -f "$dir/$STEPS/model.safetensors" ]; then state=DONE
    elif tmux has-session -t "train_$exp" 2>/dev/null; then state=RUNNING; all_done=0
    else
      all_done=0
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
      uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu")
      ours=0
      for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | cut -d, -f2); do
        tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q TacGuide && ours=1
      done
      n="${tries[$exp]:-0}"
      if [ "$n" -ge 5 ]; then state=GAVE_UP
      elif [ "$ours" = 1 ] || [ "$free" -lt 54000 ]; then state="WAITING(free=${free}MiB,ours=$ours)"
      else
        tries[$exp]=$((n + 1))
        flag=--resume; ls "$dir" 2>/dev/null | grep -q '^[0-9][0-9]*$' || flag=
        [ -z "$flag" ] && [ -d "$dir" ] && rmdir "$dir" 2>/dev/null
        tmux new-session -d -s "train_$exp" "cd $ROOT; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True training/tacguide.sh train $exp $gpu $flag 2>&1 | tee -a $TG/logs/train_$exp.log"
        state="RESUMED(#${tries[$exp]})"
      fi
    fi
    out+="$(date -u +%FT%TZ) $exp gpu=$gpu state=$state step=$(last_step "$exp") checkpoints=$(ls "$dir" 2>/dev/null | grep -c '^[0-9]')"$'\n'
  done
  printf '%s' "$out" > "$TG/logs/watch_ab.status"
  printf '%s' "$out" >> "$TG/logs/watch_ab.history"
  [ "$all_done" = 1 ] && break
  sleep 300
done
