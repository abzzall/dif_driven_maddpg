#!/usr/bin/env bash
# Round-1 follow-up work, queued 2026-09-08 while the first orchestrator was still
# running. It WAITS for that orchestrator to exit, then does the work that needs a
# free GPU. Nothing here touches or restarts the running job.
#
#   Step 1  restore the full R2.2 protocol by adding the lr=1e-3 arm (15 runs)
#   Step 2  scale-control cell for R2.5: n=5 at env_size 25.298
#   Step 3  held-out evaluation of every lr-sensitivity checkpoint
#
# Resumable: every step skips work that is already finished.
set -uo pipefail

PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT="$PROJ/review_round1_runs/round1_extra"
LRROOT="$PROJ/review_round1_runs/r2_2_lr_sensitivity"
R25="$PROJ/review_round1_runs/r2_5_cardinality_density"
LOCK="$ROOT/followup.lock"
WAIT_PID="${WAIT_PID:-}"

mkdir -p "$ROOT"
cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"

if [ -f "$LOCK" ]; then
  prior=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$prior" ] && kill -0 "$prior" 2>/dev/null; then
    echo "Follow-up already running as pid $prior."; exit 0
  fi
  rm -f "$LOCK"
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

exec >> "$ROOT/followup.log" 2>&1
echo "[$(date '+%F %T')] FOLLOW-UP QUEUED (pid $$), waiting for orchestrator ${WAIT_PID:-none}"

# --- wait for the running orchestrator, without touching it -------------------
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[$(date '+%F %T')] orchestrator $WAIT_PID exited"
fi
# belt and braces: never start while a trainer or evaluator is live
while pgrep -f 'run/train_seeded.py|run/eval_zeroshot.py|run/eval_frozen.py' >/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] GPU free; starting follow-up"

# --- gate: only proceed if the lr=1e-4 arm actually completed -----------------
done14=$(find "$LRROOT/runs" -maxdepth 1 -name '*_lr1em4_seed*' -type d 2>/dev/null | wc -l)
if [ "$done14" -lt 15 ]; then
  echo "[$(date '+%F %T')] ABORT: only $done14/15 lr=1e-4 runs present; phase 2 did not finish."
  echo "  Fix the phase-2 problem, then re-run this script."
  exit 1
fi
echo "[$(date '+%F %T')] gate passed: 15/15 lr=1e-4 runs present"

# --- step 1: add the lr=1e-3 arm ---------------------------------------------
echo "[$(date '+%F %T')] ===== STEP 1: lr=1e-3 arm (restores the full predeclared protocol) ====="
if LRS="1e-3" bash run/run_r2_2_lr_sensitivity.sh; then
  echo "[$(date '+%F %T')] STEP 1 OK"
else
  echo "[$(date '+%F %T')] STEP 1 FAILED -- continuing to steps 2 and 3 anyway"
fi

# --- step 2: R2.5 scale control ----------------------------------------------
echo "[$(date '+%F %T')] ===== STEP 2: scale control, n=5 at env_size 25.298 ====="
CTRL="$R25/cells/scalecontrol_n5.csv"
if [ -f "$CTRL" ]; then
  echo "SKIP (exists)"
else
  RUNS=()
  for s in 1 2 3 4 5; do RUNS+=("$PROJ/review_round1_runs/c_checkpoints_local/C_n5_seed$s"); done
  for s in 6 7 8 9 10; do RUNS+=("$PROJ/review_round1_runs/r2_3_additional_seeds/runs/C_n5_seed$s"); done
  "${VENV:-$PROJ/.venv}/bin/python" run/eval_zeroshot.py "${RUNS[@]}" --eval-n 5 --episodes 100 \
      --env-size 25.298 --out "$CTRL" && echo "STEP 2 OK" || echo "STEP 2 FAILED"
fi

# --- step 3: held-out evaluation of every lr-sensitivity checkpoint -----------
echo "[$(date '+%F %T')] ===== STEP 3: held-out evaluation of lr-sensitivity checkpoints ====="
OUT="$LRROOT/lr_sensitivity_heldout_eval.csv"
mapfile -t LRRUNS < <(find "$LRROOT/runs" -maxdepth 1 -type d -name '*_lr*_seed*' | sort)
echo "evaluating ${#LRRUNS[@]} checkpoints on seeds 900000-900099"
if [ "${#LRRUNS[@]}" -gt 0 ]; then
  "${VENV:-$PROJ/.venv}/bin/python" run/eval_frozen.py "${LRRUNS[@]}" --episodes 100 --seed0 900000 \
      --out "$OUT" && echo "STEP 3 OK -> $OUT" || echo "STEP 3 FAILED"
fi

echo "[$(date '+%F %T')] FOLLOW-UP COMPLETE"
