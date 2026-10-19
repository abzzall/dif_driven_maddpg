#!/usr/bin/env bash
# Second round-1 follow-up, queued 2026-09-08 while the R2.4 shift study was running.
# WAITS for that study to finish; never runs concurrently with it.
#
#   Step 1  extend component controls D and E from five to ten seeds (10 training runs)
#   Step 2  held-out evaluation of the new D and E checkpoints
#   Step 3  landmark-distribution shift, the last untested axis of R2.4
#
# Resumable: every step skips finished work.
set -uo pipefail

PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT="$PROJ/review_round1_runs/round1_extra"
ABL="$PROJ/review_round1_runs/r2_3_ablation_seeds"
SHIFT="$PROJ/review_round1_runs/r2_4_shift"
LOCK="$ROOT/followup2.lock"
WAIT_PID="${WAIT_PID:-}"
mkdir -p "$ROOT" "$ABL"
cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"

if [ -f "$LOCK" ]; then
  prior=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$prior" ] && kill -0 "$prior" 2>/dev/null; then
    echo "Follow-up 2 already running as pid $prior."; exit 0
  fi
  rm -f "$LOCK"
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
exec >> "$ROOT/followup2.log" 2>&1
echo "[$(date '+%F %T')] FOLLOW-UP 2 QUEUED (pid $$), waiting for ${WAIT_PID:-none}"

if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
while pgrep -f 'run/train_seeded.py|run/eval_shift.py|run/eval_frozen.py|run/eval_zeroshot.py' >/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] GPU free; starting"

# --- gate: the shift study must have completed all 8 cells ------------------
cells=$(find "$SHIFT/cells" -name '*.csv' 2>/dev/null | wc -l)
if [ "$cells" -lt 8 ]; then
  echo "[$(date '+%F %T')] WARNING: shift study has $cells/8 cells; continuing anyway"
  echo "  (re-run run/run_r2_4_shift.sh separately to finish it)"
fi

# --- step 1: D and E seeds 6-10 --------------------------------------------
echo "[$(date '+%F %T')] ===== STEP 1: D and E seeds 6-10 (10 training runs) ====="
if ROOT="$ABL" SEEDS="6 7 8 9 10" bash run/run_ablation_stream.sh; then
  echo "[$(date '+%F %T')] STEP 1 OK"
else
  echo "[$(date '+%F %T')] STEP 1 FAILED -- continuing to step 3"
fi

# --- step 2: held-out evaluation of the new D and E runs -------------------
echo "[$(date '+%F %T')] ===== STEP 2: held-out evaluation of new D and E runs ====="
OUT="$ABL/ablation_heldout_eval_seeds6_10.csv"
if [ -f "$OUT" ]; then
  echo "SKIP (exists)"
else
  mapfile -t NEWRUNS < <(find "$ABL/runs" -maxdepth 1 -type d \( -name 'D_deepsets_n5_seed*' -o -name 'E_nomask_n5_seed*' \) | sort)
  if [ "${#NEWRUNS[@]}" -gt 0 ]; then
    "${VENV:-$PROJ/.venv}/bin/python" run/eval_frozen.py "${NEWRUNS[@]}" --episodes 100 --seed0 900000 \
        --out "$OUT" && echo "STEP 2 OK -> $OUT" || echo "STEP 2 FAILED"
  else
    echo "STEP 2 SKIPPED: no new run directories found"
  fi
fi

# --- step 3: landmark-distribution shift -----------------------------------
echo "[$(date '+%F %T')] ===== STEP 3: landmark-distribution shift ====="
CELL="$SHIFT/cells/landmarks_clustered.csv"
if [ -f "$CELL" ]; then
  echo "SKIP (exists)"
else
  LOCAL="$PROJ/review_round1_runs/c_checkpoints_local"
  NEW="$PROJ/review_round1_runs/r2_3_additional_seeds/runs"
  RUNS=()
  for arm in C B B_wide; do
    for s in 1 2 3 4 5;  do RUNS+=("$LOCAL/${arm}_n5_seed$s"); done
    for s in 6 7 8 9 10; do RUNS+=("$NEW/${arm}_n5_seed$s");   done
  done
  "${VENV:-$PROJ/.venv}/bin/python" run/eval_shift.py "${RUNS[@]}" --condition landmarks_clustered \
      --landmark-dist clustered --episodes 100 --out "$CELL" \
    && echo "STEP 3 OK" || echo "STEP 3 FAILED"
fi

echo "[$(date '+%F %T')] FOLLOW-UP 2 COMPLETE"
