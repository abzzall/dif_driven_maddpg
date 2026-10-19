#!/usr/bin/env bash
set -euo pipefail
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJ"
source "${VENV:-$PROJ/.venv}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
AUDIT_ROOT=review_round1_runs/r2_6_collision_audit
mkdir -p "$AUDIT_ROOT"
exec > >(tee -a "$AUDIT_ROOT/stream.log") 2>&1
date -Iseconds
python run/eval_collision_audit.py --verify --out "$AUDIT_ROOT"
python run/eval_collision_audit.py --out "$AUDIT_ROOT"
date -Iseconds
