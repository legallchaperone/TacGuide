#!/usr/bin/env bash
# Start TacGuide policy servers and the job scheduler, one at a time, each isolated.
#
#   start_guided_servers.sh single <experiment> <gpu> <port>      # one model, e.g. single gc_C 4 8006
#   start_guided_servers.sh cfg  <gpu> [w ...]                    # CFG(A,B): strong gc_A, weak gc_B
#   start_guided_servers.sh zero <gpu> [w ...]                    # zero-training: strong gc_C, weak gc_B
#   start_guided_servers.sh queue                                 # the student job scheduler
#   start_guided_servers.sh stop <name>                           # e.g. stop cfg_w2, stop gc_C, stop queue
#   start_guided_servers.sh list
#
# Guided servers use fixed ports: w = 1 -> 8002, 1.5 -> 8003, 2 -> 8004, 3 -> 8005 (default w: 2 1.5 3).
# All models are the 10000-step checkpoints, noise seed 0.
#
# Why "one at a time, each in its own tmux server": on 2026-11-25 the memory daemon (systemd-oomd)
# twice killed every TacGuide process while a server was loading its model. First all jobs shared one
# tmux server; then they were in `systemd-run --user` scopes, which sit under user@<uid>.service and
# its 50% memory-pressure kill rule. A plain login session with a tmux server of its own is outside
# that rule, so that is what this script uses. Run it from a fresh ssh connection per process when
# you can, check `cat /proc/pressure/memory` first, and never load two models at once.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/training/tacguide_env.sh"
mkdir -p "$TG/logs/distill" "$TG/queue"
STEP=10000

pressure_ok() {  # refuse to add load while the machine is already stalling on memory
  local avg10
  avg10="$(awk '/^full/ {sub("avg10=", "", $2); print $2}' /proc/pressure/memory 2>/dev/null || echo 0)"
  if awk -v v="$avg10" 'BEGIN { exit !(v > 2.0) }'; then
    echo "memory pressure is ${avg10}% (full, 10 s); not starting anything now" >&2; return 1
  fi
}

isolated() {  # name command...: run in a tmux server of its own
  local name="$1"; shift
  if tmux -L "tg_$name" has-session 2>/dev/null; then echo "tg_$name is already running"; return 1; fi
  pressure_ok
  tmux -L "tg_$name" new-session -d -s "$name" "$*"
}

serve() {  # name gpu port serve_guided-arguments...
  local name="$1" gpu="$2" port="$3"; shift 3
  isolated "$name" "cd '$ROOT' && source training/tacguide_env.sh && CUDA_VISIBLE_DEVICES=$gpu \
    '$TG/venv/bin/python' training/tacguide/serve_guided.py $* --step $STEP --port $port 2>&1 | tee '$TG/logs/distill/serve_$name.log'"
  echo -n "started $name on gpu $gpu, port $port; waiting for it to load"
  for _ in $(seq 1 120); do
    if [ "$(curl -s -m 3 "localhost:$port/healthz" || true)" = OK ]; then echo " ... ready"; return 0; fi
    if ! tmux -L "tg_$name" has-session 2>/dev/null; then echo " ... FAILED, see $TG/logs/distill/serve_$name.log"; return 1; fi
    echo -n "."; sleep 5
  done
  echo " ... not ready after 10 minutes"; return 1
}

port_for() { case "$1" in 1) echo 8002 ;; 1.5) echo 8003 ;; 2) echo 8004 ;; 3) echo 8005 ;; *) echo "no port for w=$1" >&2; exit 2 ;; esac; }

mode="${1:?single, cfg, zero, queue, stop or list}"; shift
case "$mode" in
  single)
    exp="${1:?experiment}"; gpu="${2:?gpu}"; port="${3:?port}"
    serve "$exp" "$gpu" "$port" --strong "$exp"
    ;;
  cfg|zero)
    strong=gc_A; [ "$mode" = zero ] && strong=gc_C
    gpu="${1:?gpu}"; shift
    [ $# -gt 0 ] || set -- 2 1.5 3
    for w in "$@"; do serve "${mode}_w${w/./_}" "$gpu" "$(port_for "$w")" --strong "$strong" --weak gc_B --w "$w"; done
    ;;
  queue)
    isolated queue "cd '$ROOT' && source training/tacguide_env.sh && python3 training/tacguide/gpu_queue.py run \
      --queue training/tacguide/queue.gc.json --interval 180 2>&1 | tee -a '$TG/queue/scheduler.log'"
    echo "started the scheduler (tmux -L tg_queue attach)"
    ;;
  stop)
    tmux -L "tg_${1:?name}" kill-server && echo "stopped $1"
    ;;
  list)
    for socket in "${TMUX_TMPDIR:-/tmp}/tmux-$(id -u)"/tg_*; do
      [ -S "$socket" ] && tmux -L "$(basename "$socket")" ls 2>/dev/null | sed "s/^/$(basename "$socket"): /" || true
    done
    ;;
  *) echo "unknown mode '$mode'" >&2; exit 2 ;;
esac
