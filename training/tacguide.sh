#!/usr/bin/env bash
# TacGuide experiment commands for the training machine. Run from anywhere:
#
#   training/tacguide.sh prep  <dir with the collected .hdf5 episodes>   # link data, convert, stats
#   training/tacguide.sh smoke gc_A|gc_B <gpu>                           # 10 steps, batch 64, no wandb
#   training/tacguide.sh train gc_A|gc_B <gpu> [--resume]                # full run; TG_WANDB=1 logs to wandb project TacGuide
#   training/tacguide.sh serve gc_A|gc_B <gpu> [port] [step]             # policy server for evaluation
#
# <task>_A = OpenVTLA (vision + touch), <task>_B = π0.5 (vision only). The task is TG_TASK (default gc);
# for Gear Assembly use TG_TASK=ga with ga_A / ga_B.
# Steps and checkpoint interval can be changed with TG_STEPS and TG_SAVE_INTERVAL.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/training/tacguide_env.sh"
cd "$ROOT"
TASK="${TG_TASK:-gc}"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
STEPS="${TG_STEPS:-10000}"
SAVE_INTERVAL="${TG_SAVE_INTERVAL:-1000}"

model_for() {
  case "$1" in
    *_A*) echo openvtla ;;
    *_B*) echo pi05 ;;
    *) echo "unknown experiment '$1' (expected <task>_A or <task>_B, e.g. gc_A, ga_B)" >&2; exit 2 ;;
  esac
}

cmd="${1:?usage: tacguide.sh prep|smoke|train|serve ...}"; shift
case "$cmd" in
  prep)
    src="$(realpath "${1:?directory with .hdf5 episodes}")"
    ls "$src"/*.hdf5 >/dev/null
    mkdir -p data/training/hdf5
    ln -sfn "$src" "data/training/hdf5/$TASK"
    "$PY" training/run.py convert --task "$TASK"
    CUDA_VISIBLE_DEVICES= "$PY" training/run.py stats --task "$TASK" --model pi05
    CUDA_VISIBLE_DEVICES= "$PY" training/run.py stats --task "$TASK" --model openvtla
    "$PY" training/checks/check_conversion.py --task "$TASK" --episodes 3 --out "data/training/checks/$TASK"
    ;;
  smoke)
    exp="${1:?gc_A or gc_B}"; gpu="${2:?gpu index}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" training/run.py train --task "$TASK" --model "$(model_for "$exp")" \
      --exp-name "${exp}_smoke" --steps "${TG_SMOKE_STEPS:-10}" --save-interval "${TG_SMOKE_STEPS:-10}" \
      --log-interval 1 --overwrite
    ;;
  train)
    exp="${1:?gc_A or gc_B}"; gpu="${2:?gpu index}"; shift 2
    # wandb is not reachable from every machine: TG_WANDB_PROXY (e.g. a local SOCKS forward) is used by this process only.
    if [ -n "${TG_WANDB:-}" ] && [ -n "${TG_WANDB_PROXY:-}" ]; then export HTTPS_PROXY="$TG_WANDB_PROXY" https_proxy="$TG_WANDB_PROXY"; fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" training/run.py train --task "$TASK" --model "$(model_for "$exp")" \
      --exp-name "$exp" --steps "$STEPS" --save-interval "$SAVE_INTERVAL" ${TG_WANDB:+--wandb} "$@"
    ;;
  serve)
    exp="${1:?gc_A or gc_B}"; gpu="${2:?gpu index}"; port="${3:-8000}"; step="${4:-}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" training/serve.py --task "$TASK" --model "$(model_for "$exp")" \
      --exp-name "$exp" --port "$port" ${step:+--step "$step"}
    ;;
  *) echo "unknown command '$cmd'" >&2; exit 2 ;;
esac
