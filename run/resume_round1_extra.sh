#!/usr/bin/env bash
# Resume the round-1 additional experiments after any interruption.
#
#   bash run/resume_round1_extra.sh          # resume in the foreground
#   setsid nohup bash run/resume_round1_extra.sh >/dev/null 2>&1 < /dev/null &   # detached
#
# Safe to run at any time. Completed work is never redone:
#   Phase 1 skips any evaluation cell whose CSV already exists.
#   Phase 2 skips any training run with verified finished metadata, and resumes
#   an interrupted run from its saved training_state.pkl.
# Refuses to start a second copy while one is already running.
set -uo pipefail

PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT="$PROJ/review_round1_runs/round1_extra"
LOCK="$ROOT/orchestrator.lock"
mkdir -p "$ROOT"
cd "$PROJ" || exit 1

if [ -f "$LOCK" ]; then
  prior=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$prior" ] && kill -0 "$prior" 2>/dev/null; then
    echo "Already running as pid $prior. Nothing to do."
    echo "Follow it with:  tail -f $ROOT/stream.log"
    exit 0
  fi
  echo "Clearing stale lock from pid ${prior:-unknown}."
  rm -f "$LOCK"
fi

echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "[$(date '+%F %T')] RESUME WRAPPER START (pid $$)"
bash run/run_round1_extra_experiments.sh
status=$?
echo "[$(date '+%F %T')] RESUME WRAPPER EXIT status=$status"
exit $status
