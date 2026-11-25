#!/usr/bin/env bash
# Run one long job in its own tmux server, so that a dying tmux server takes down only that job,
# not every job of the account.
#
#   training/isolated.sh <name> <command...>      # start; attach with: tmux -L tg-<name> attach
#   training/isolated.sh --alive <name>           # exit 0 if the job's tmux server is still up
#   training/isolated.sh --stop <name>            # send Ctrl-C to the job
#
# Start one job at a time and wait until it is running before starting the next.
set -euo pipefail
case "${1:-}" in
  --alive) tmux -L "tg-$2" has-session -t "$2" 2>/dev/null ;;
  --stop)  tmux -L "tg-$2" send-keys -t "$2" C-c ;;
  "") echo "usage: isolated.sh <name> <command...>" >&2; exit 2 ;;
  *)
    name="$1"; shift
    if tmux -L "tg-$name" has-session -t "$name" 2>/dev/null; then echo "$name is already running" >&2; exit 1; fi
    # Deliberately NOT systemd-run --user: units under user@UID.service are watched by systemd-oomd
    # (memory-pressure limit 50% for 20 s) and were killed within minutes while loading checkpoints
    # from the slow shared disk. A plain tmux server stays in the login session scope instead.
    setsid tmux -L "tg-$name" new-session -d -s "$name" "$*"
    ;;
esac
