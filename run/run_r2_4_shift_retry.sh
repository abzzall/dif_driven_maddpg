#!/usr/bin/env bash
# Re-run of the R2.4 shift study after the 2026-09-08 17:00 failure.
# Cause: run/eval_shift.py did not pass mlp_hidden, so B_wide checkpoints (hidden 512)
# failed to load against a default 256-wide model. Fixed and smoke-tested on all three
# arms before this re-queue. No results were produced by the failed attempt.
# Waits for the follow-up-2 chain so the two never share the GPU.
set -uo pipefail
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJ" || exit 1
WAIT_PID="${WAIT_PID:-}"
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
while pgrep -f 'run/train_seeded.py|run/eval_shift.py|run/eval_frozen.py' >/dev/null; do sleep 60; done
exec bash run/run_r2_4_shift.sh
