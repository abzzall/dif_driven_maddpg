#!/usr/bin/env bash
# Reviewer 2.3: predeclared extension from five to ten primary seeds.
# Protocol: .ai/PROTOCOL_R2_3_ADDITIONAL_SEEDS.md
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)/review_round1_runs/r2_3_additional_seeds}"
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
  local arm="$1" seed="$2"
  local name="${arm}_n5_seed${seed}"
  local out="$ROOT/runs/$name"
  local log="$ROOT/logs/$name.log"
  local expected_obs expected_width
  local -a arch_flags=()

  case "$arm" in
    C) expected_obs=graph; expected_width=None; arch_flags+=(--obs graph) ;;
    B) expected_obs=flat; expected_width=None; arch_flags+=(--obs flat) ;;
    B_wide) expected_obs=flat; expected_width=512; arch_flags+=(--obs flat --mlp_hidden 512) ;;
    *) echo "FATAL: unknown arm $arm"; return 2 ;;
  esac

  if [ -f "$out/meta.json" ] && python -c 'import json, math, sys; m=json.load(open(sys.argv[1])); width=None if sys.argv[4] == "None" else int(sys.argv[4]); assert m.get("launcher_status") == "finished"; assert m.get("seed") == int(sys.argv[2]); assert m.get("observation") == sys.argv[3]; assert m.get("mlp_hidden") == width; assert math.isclose(float(m.get("learning_rate", 3e-4)), 3e-4, rel_tol=0, abs_tol=1e-12)' "$out/meta.json" "$seed" "$expected_obs" "$expected_width"; then
    echo "[$(date '+%F %T')] SKIP  $name (finished metadata verified)"
    return 0
  fi

  if [ -d "$out" ] && [ -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "[$(date '+%F %T')] FATAL $name has an incomplete or mismatched existing directory"
    return 1
  fi

  mkdir -p "$out"
  echo "[$(date '+%F %T')] START $name (arm=$arm seed=$seed lr=3e-4)"
  python run/train_seeded.py --algorithm mappo --n 5 --mode full --seed "$seed" \
      --v_ang_max pi2 --reward new "${arch_flags[@]}" \
      --no_step_csv --plot_every 50 --convergence --conv_max_hours 3 \
      --conv_max_episodes 5000 --require_gpu --out_dir "$out" >> "$log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "[$(date '+%F %T')] FATAL $name failed (exit $status); stream stopped"
    return 1
  fi

  if ! python -c 'import json, math, sys; m=json.load(open(sys.argv[1])); width=None if sys.argv[4] == "None" else int(sys.argv[4]); assert m.get("launcher_status") == "finished"; assert m.get("seed") == int(sys.argv[2]); assert m.get("observation") == sys.argv[3]; assert m.get("mlp_hidden") == width; assert math.isclose(float(m.get("learning_rate", 3e-4)), 3e-4, rel_tol=0, abs_tol=1e-12)' "$out/meta.json" "$seed" "$expected_obs" "$expected_width"; then
    echo "[$(date '+%F %T')] FATAL $name metadata validation failed; stream stopped"
    return 1
  fi
  echo "[$(date '+%F %T')] END   $name (metadata verified)"
}

for seed in 6 7 8 9 10; do
  for arm in C B B_wide; do
    run_one "$arm" "$seed" || exit 1
  done
done

echo "[$(date '+%F %T')] R2.3 ADDITIONAL-SEED TRAINING COMPLETE"
