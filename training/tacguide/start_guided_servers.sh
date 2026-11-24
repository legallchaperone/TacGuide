#!/usr/bin/env bash
# Start the four guided-inference servers of one evaluation stage, each in its own tmux session.
#
#   training/tacguide/start_guided_servers.sh cfg  <gpu for w=1,1.5> <gpu for w=2,3>   # stage 4: strong gc_A, weak gc_B
#   training/tacguide/start_guided_servers.sh zero <gpu for w=1,1.5> <gpu for w=2,3>   # stage 6: strong gc_C, weak gc_B
#   training/tacguide/start_guided_servers.sh stop cfg|zero
#
# Ports: 8002 (w = 1, the correctness check), 8003 (w = 1.5), 8004 (w = 2), 8005 (w = 3).
# Final (10000-step) checkpoints, noise seed 0. Two servers fit on one card (about 17 GB each).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/training/tacguide_env.sh"
mode="${1:?cfg, zero or stop}"
if [ "$mode" = stop ]; then
  for w in 1 1_5 2 3; do tmux kill-session -t "tg_${2:?cfg or zero}_w$w" 2>/dev/null || true; done
  exit 0
fi
case "$mode" in
  cfg) strong=gc_A ;;
  zero) strong=gc_C ;;
  *) echo "unknown mode '$mode'" >&2; exit 2 ;;
esac
gpu_a="${2:?gpu index}"; gpu_b="${3:?gpu index}"
mkdir -p "$TG/logs/distill"
start() {  # w gpu port
  local session="tg_${mode}_w${1/./_}"
  tmux new-session -d -s "$session" "cd '$ROOT' && source training/tacguide_env.sh && CUDA_VISIBLE_DEVICES=$2 \
    '$TG/venv/bin/python' training/tacguide/serve_guided.py --strong $strong --weak gc_B --w $1 --step 10000 --port $3 \
    2>&1 | tee '$TG/logs/distill/serve_${mode}_w$1.log'"
  echo "started $session: $strong vs gc_B, w = $1, gpu $2, port $3"
}
start 1 "$gpu_a" 8002
start 1.5 "$gpu_a" 8003
start 2 "$gpu_b" 8004
start 3 "$gpu_b" 8005
