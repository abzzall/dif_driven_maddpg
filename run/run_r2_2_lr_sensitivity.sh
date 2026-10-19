#!/usr/bin/env bash
# Reviewer 2.2: predeclared equal-budget learning-rate sensitivity study.
# Protocol: .ai/PROTOCOL_R2_2_LR_SENSITIVITY.md
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)/review_round1_runs/r2_2_lr_sensitivity}"
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"

cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
# shellcheck disable=SC1091
source "${VENV:-$PROJ/.venv}/bin/activate" || { echo "FATAL: cannot activate virtualenv (set VENV)"; exit 1; }
mkdir -p "$ROOT/logs" "$ROOT/runs"
exec > >(tee -a "$ROOT/logs/stream.log") 2>&1

if ! python -c 'import torch, sys; print(f"CUDA preflight: available={torch.cuda.is_available()} count={torch.cuda.device_count()}"); sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)'; then
  echo "FATAL: CUDA preflight failed"
  exit 1
fi

run_one() {
  local arm="$1" lr="$2" seed="$3"
  local lr_tag="${lr//./p}"
  lr_tag="${lr_tag//-/m}"
  local name="${arm}_lr${lr_tag}_seed${seed}"
  local out="$ROOT/runs/$name"
  local log="$ROOT/logs/$name.log"
  local -a arch_flags=()

  case "$arm" in
    C) arch_flags+=(--obs graph) ;;
    B) arch_flags+=(--obs flat) ;;
    B_wide) arch_flags+=(--obs flat --mlp_hidden 512) ;;
    *) echo "FATAL: unknown arm $arm"; return 2 ;;
  esac

  if [ -f "$out/meta.json" ] && python -c 'import json, math, sys; m=json.load(open(sys.argv[1])); assert m.get("launcher_status") == "finished"; assert m.get("seed") == int(sys.argv[2]); assert math.isclose(float(m.get("learning_rate")), float(sys.argv[3]), rel_tol=0, abs_tol=1e-12)' "$out/meta.json" "$seed" "$lr"; then
    echo "[$(date '+%F %T')] SKIP  $name (finished metadata verified)"
    return 0
  fi

  # Interruption handling. A directory holding training_state.pkl is a cleanly
  # interrupted run: train_seeded.py chdirs into it and resumes from the saved
  # episode index, and a stale .run_all.lock is reclaimed by pid liveness check.
  # Any other non-empty directory is unexplained and still stops the stream.
  local action="START"
  if [ -d "$out" ] && [ -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    if [ -f "$out/training_state.pkl" ]; then
      action="RESUME"
      echo "$(date '+%F %T') resume $name" >> "$ROOT/logs/resumes.log"
    else
      echo "[$(date '+%F %T')] FATAL $name has a non-empty directory with no training_state.pkl"
      return 1
    fi
  fi

  mkdir -p "$out"
  echo "[$(date '+%F %T')] $action $name (arm=$arm lr=$lr seed=$seed)"
  python run/train_seeded.py --algorithm mappo --n 5 --mode full --seed "$seed" \
      --v_ang_max pi2 --reward new "${arch_flags[@]}" --learning_rate "$lr" \
      --no_step_csv --plot_every 50 --convergence --conv_max_hours 3 \
      --conv_max_episodes 5000 --require_gpu --out_dir "$out" >> "$log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "[$(date '+%F %T')] FATAL $name failed (exit $status); stream stopped"
    return 1
  fi

  if ! python -c 'import json, math, sys; m=json.load(open(sys.argv[1])); assert m.get("launcher_status") == "finished"; assert m.get("seed") == int(sys.argv[2]); assert math.isclose(float(m.get("learning_rate")), float(sys.argv[3]), rel_tol=0, abs_tol=1e-12)' "$out/meta.json" "$seed" "$lr"; then
    echo "[$(date '+%F %T')] FATAL $name metadata validation failed; stream stopped"
    return 1
  fi
  echo "[$(date '+%F %T')] END   $name (metadata verified)"
}

# LRS defaults to the full predeclared two-rate protocol; override to run a reduced
# scope (e.g. LRS="1e-4"). Any reduction must be recorded in the protocol file.
for lr in ${LRS:-1e-4 1e-3}; do
  for arm in C B B_wide; do
    for seed in 1 2 3 4 5; do
      run_one "$arm" "$lr" "$seed" || exit 1
    done
  done
done

echo "[$(date '+%F %T')] R2.2 LR SENSITIVITY TRAINING COMPLETE"
