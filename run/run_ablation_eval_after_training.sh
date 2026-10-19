#!/usr/bin/env bash
# Wait for the single ablation-training stream, validate all ten runs, then
# evaluate their final checkpoints on the original deterministic held-out bank.
set -uo pipefail

ROOT="${ROOT:-${PPO_STUDY_ROOT:-$HOME/dif_driven_ppo_study}}"
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="$ROOT/ablation_heldout_eval.csv"

cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
# shellcheck disable=SC1091
source "${VENV:-$PROJ/.venv}/bin/activate" || { echo "FATAL: venv"; exit 1; }
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/ablation_eval_after_training.log") 2>&1

runs=()
for seed in 1 2 3 4 5; do
  runs+=("$ROOT/runs/D_deepsets_n5_seed${seed}")
done
for seed in 1 2 3 4 5; do
  runs+=("$ROOT/runs/E_nomask_n5_seed${seed}")
done

all_valid() {
  python -c '
import json, pathlib, sys
for raw in sys.argv[1:]:
    run = pathlib.Path(raw)
    try:
        meta = json.loads((run / "meta.json").read_text())
    except Exception:
        raise SystemExit(1)
    if meta.get("launcher_status") != "finished" or not meta.get("finished"):
        raise SystemExit(1)
    if run.name.startswith("D_deepsets_"):
        expected = ("deepsets", True)
    elif run.name.startswith("E_nomask_"):
        expected = ("attention", False)
    else:
        raise SystemExit(1)
    if (meta.get("entity_encoder"), meta.get("use_entity_mask")) != expected:
        raise SystemExit(1)
' "${runs[@]}"
}

echo "[$(date '+%F %T')] Waiting for 10 validated ablation runs"
while ! all_valid; do
  if ! pgrep -f '[r]un/run_ablation_stream.sh' >/dev/null; then
    echo "[$(date '+%F %T')] FATAL: training stream exited before all runs validated"
    exit 1
  fi
  sleep 30
done

echo "[$(date '+%F %T')] All runs validated; starting deterministic held-out evaluation"
python run/eval_frozen.py "${runs[@]}" \
  --episodes 100 --seed0 900000 --max_steps 500 --out "$OUT"
status=$?
if [ "$status" -ne 0 ]; then
  echo "[$(date '+%F %T')] FATAL: held-out evaluation failed (exit $status)"
  exit "$status"
fi

echo "[$(date '+%F %T')] ABLATION HELD-OUT EVALUATION COMPLETE: $OUT"
