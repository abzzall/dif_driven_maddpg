#!/usr/bin/env bash
# Round-1 additional experiments, launched 2026-09-08.
#   Phase 1  R2.5  wider team sizes with independent density control (evaluation only)
#   Phase 2  R2.2  learning-rate sensitivity, reduced scope: lr=1e-4 only, 15 training runs
#
# Phases run SEQUENTIALLY and never concurrently. The training stopping rule has a
# three-hour wall-clock cap, so GPU contention from a second job could change which
# runs hit that cap and would confound the experiment. The single 6 GB GPU is also
# shared. Do not parallelise these phases.
set -uo pipefail

PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT="${ROOT:-$PROJ/review_round1_runs/round1_extra}"
mkdir -p "$ROOT"
cd "$PROJ" || exit 1
exec > >(tee -a "$ROOT/stream.log") 2>&1

echo "[$(date '+%F %T')] ORCHESTRATOR START (pid $$)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "[$(date '+%F %T')] ===== PHASE 1: R2.5 cardinality/density ====="
if bash run/run_r2_5_cardinality_density.sh; then
  echo "[$(date '+%F %T')] PHASE 1 OK"
else
  echo "[$(date '+%F %T')] PHASE 1 FAILED -- stopping before phase 2"
  exit 1
fi

echo "[$(date '+%F %T')] ===== PHASE 2: R2.2 learning-rate sensitivity (lr=1e-4) ====="
if LRS="1e-4" bash run/run_r2_2_lr_sensitivity.sh; then
  echo "[$(date '+%F %T')] PHASE 2 TRAINING OK"
else
  echo "[$(date '+%F %T')] PHASE 2 FAILED"
  exit 1
fi

echo "[$(date '+%F %T')] ORCHESTRATOR COMPLETE -- checkpoints still need held-out evaluation"
