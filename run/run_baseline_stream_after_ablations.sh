#!/usr/bin/env bash
# Sequential off-policy controls after ablation training + held-out evaluation.
set -uo pipefail

ROOT="${ROOT:-${PPO_STUDY_ROOT:-$HOME/dif_driven_ppo_study}}"
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
ABLATION_EVAL="$ROOT/ablation_heldout_eval.csv"
BASELINE_CPU="${BASELINE_CPU:-11}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
# shellcheck disable=SC1091
source "${VENV:-$PROJ/.venv}/bin/activate" || { echo "FATAL: venv"; exit 1; }
mkdir -p "$ROOT/logs" "$ROOT/runs"
exec > >(tee -a "$ROOT/logs/baseline_stream.log") 2>&1

ablation_eval_complete() {
  python -c 'import pandas as pd, sys; p=sys.argv[1]; d=pd.read_csv(p); expected={f"D_deepsets_n5_seed{i}" for i in range(1,6)}|{f"E_nomask_n5_seed{i}" for i in range(1,6)}; raise SystemExit(0 if expected <= set(d["run"].astype(str)) else 1)' "$ABLATION_EVAL" 2>/dev/null
}

echo "[$(date '+%F %T')] Waiting for completed ablation held-out evaluation"
while ! ablation_eval_complete; do
  if ! pgrep -f '[r]un/run_ablation_stream.sh|[r]un/run_ablation_eval_after_training.sh' >/dev/null; then
    echo "[$(date '+%F %T')] FATAL: prerequisite streams exited without complete evaluation"
    exit 1
  fi
  sleep 30
done

if ! python -c 'import torch,sys; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "FATAL: CUDA unavailable"
  exit 1
fi

run_one() {
  local label="$1" algorithm="$2" seed="$3"
  local name="${label}_n5_seed${seed}"
  local out="$ROOT/runs/$name"
  local log="$ROOT/logs/$name.log"
  echo "[$(date '+%F %T')] START $name (algorithm=$algorithm, new reward, flat obs)"
  if ! systemd-run --user --scope --quiet \
      -p MemoryMax=2G -p MemorySwapMax=0 \
      taskset -c "$BASELINE_CPU" \
      python run/train_offpolicy_baseline.py --algorithm "$algorithm" --seed "$seed" \
      --n 5 --conv_max_hours 3 --conv_max_episodes 5000 --require_gpu \
      --out_dir "$out" >> "$log" 2>&1; then
    echo "[$(date '+%F %T')] FATAL $name failed; baseline stream stopped"
    return 1
  fi
  if ! python -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m.get("launcher_status")=="finished"; assert m.get("algorithm")==sys.argv[2]; assert m.get("reward_design")=="new"; assert m.get("observation")=="flat"' "$out/meta.json" "$algorithm"; then
    echo "[$(date '+%F %T')] FATAL $name metadata validation failed"
    return 1
  fi
  echo "[$(date '+%F %T')] END   $name (metadata verified)"
}

for seed in 1 2 3 4 5; do
  run_one F_maddpg_scratch maddpg_scratch "$seed" || exit 1
done
for seed in 1 2 3 4 5; do
  run_one G_iddpg_scratch iddpg_scratch "$seed" || exit 1
done

baseline_runs=()
for seed in 1 2 3 4 5; do
  baseline_runs+=("$ROOT/runs/F_maddpg_scratch_n5_seed${seed}")
done
for seed in 1 2 3 4 5; do
  baseline_runs+=("$ROOT/runs/G_iddpg_scratch_n5_seed${seed}")
done

echo "[$(date '+%F %T')] BASELINE TRAINING COMPLETE; starting held-out evaluation"
systemd-run --user --scope --quiet \
  -p MemoryMax=2G -p MemorySwapMax=0 \
  taskset -c "$BASELINE_CPU" \
  python run/eval_offpolicy_baselines.py "${baseline_runs[@]}" \
  --episodes 100 --seed0 900000 --max_steps 500 \
  --out "$ROOT/baseline_heldout_eval.csv" || exit 1
echo "[$(date '+%F %T')] BASELINE STREAM AND EVALUATION COMPLETE"
