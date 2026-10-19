#!/usr/bin/env bash
# Sequential reviewer-facing ablations: Deep Sets, then attention without masking.
# One trainer runs at a time. A failed or malformed run stops the stream.
set -uo pipefail

ROOT="${ROOT:-${PPO_STUDY_ROOT:-$HOME/dif_driven_ppo_study}}"
# SEEDS defaults to the original five; override to extend the cohort, e.g. SEEDS="6 7 8 9 10".
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"

cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
# shellcheck disable=SC1091
source "${VENV:-$PROJ/.venv}/bin/activate" || { echo "FATAL: venv"; exit 1; }
mkdir -p "$ROOT/logs" "$ROOT/runs"

exec > >(tee -a "$ROOT/logs/ablation_stream.log") 2>&1

if ! python -c 'import torch, sys; print(f"CUDA preflight: available={torch.cuda.is_available()} count={torch.cuda.device_count()}"); sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)'; then
  echo "FATAL: CUDA preflight failed"
  exit 1
fi

run_one() {
  local arm="$1" seed="$2" encoder="$3" mask="$4"
  local name="${arm}_n5_seed${seed}"
  local out="$ROOT/runs/$name"
  local log="$ROOT/logs/$name.log"
  local -a encoder_flags=()

  if [ "$encoder" = "deepsets" ]; then
    encoder_flags+=(--entity_encoder deepsets)
  fi
  if [ "$mask" = "false" ]; then
    encoder_flags+=(--no_entity_mask)
  fi

  if [ -f "$out/meta.json" ] && python -c 'import json, sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m.get("launcher_status") == "finished" and m.get("entity_encoder") == sys.argv[2] and m.get("use_entity_mask") == (sys.argv[3] == "true") else 1)' "$out/meta.json" "$encoder" "$mask"; then
    echo "[$(date '+%F %T')] SKIP  $name (finished and configuration verified)"
    return 0
  fi

  mkdir -p "$out"
  echo "[$(date '+%F %T')] START $name (encoder=$encoder mask=$mask)"
  python run/train_seeded.py --algorithm mappo --n 5 --mode full --seed "$seed" \
      --v_ang_max pi2 --reward new --obs graph "${encoder_flags[@]}" \
      --no_step_csv --plot_every 50 --convergence --conv_max_hours 3 \
      --conv_max_episodes 5000 --require_gpu --out_dir "$out" >> "$log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "[$(date '+%F %T')] FATAL $name failed (exit $status); stream stopped"
    return 1
  fi

  if ! python -c 'import json, sys; m=json.load(open(sys.argv[1])); assert m.get("launcher_status") == "finished", m; assert m.get("entity_encoder") == sys.argv[2], m; assert m.get("use_entity_mask") == (sys.argv[3] == "true"), m' "$out/meta.json" "$encoder" "$mask"; then
    echo "[$(date '+%F %T')] FATAL $name metadata validation failed; stream stopped"
    return 1
  fi
  echo "[$(date '+%F %T')] END   $name (configuration verified)"
}

for seed in ${SEEDS:-1 2 3 4 5}; do
  run_one D_deepsets "$seed" deepsets true || exit 1
done
for seed in ${SEEDS:-1 2 3 4 5}; do
  run_one E_nomask "$seed" attention false || exit 1
done

echo "[$(date '+%F %T')] ABLATION STREAM COMPLETE"
