#!/usr/bin/env bash
# Locked R2.3 follow-up: validate all 15 new runs and evaluate final checkpoints
# deterministically on the submitted held-out scenario bank.
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT="$PROJ/review_round1_runs/r2_3_additional_seeds"
OUT="$ROOT/additional_seeds_heldout_eval.csv"
LOG="$ROOT/logs/heldout_eval.log"

cd "$PROJ"
export PYTHONPATH="$PROJ"
source "${VENV:-$PROJ/.venv}/bin/activate"

runs=()
for seed in 6 7 8 9 10; do
  runs+=("$ROOT/runs/C_n5_seed${seed}")
  runs+=("$ROOT/runs/B_n5_seed${seed}")
  runs+=("$ROOT/runs/B_wide_n5_seed${seed}")
done

python -c '
import json, pathlib, sys
for raw in sys.argv[1:]:
    run = pathlib.Path(raw)
    meta = json.loads((run / "meta.json").read_text())
    arm = run.name.rsplit("_n", 1)[0]
    expected = {
        "C": ("graph", "attention", True, None),
        "B": ("flat", None, None, None),
        "B_wide": ("flat", None, None, 512),
    }[arm]
    actual = (meta.get("observation"), meta.get("entity_encoder"),
              meta.get("use_entity_mask"), meta.get("mlp_hidden"))
    assert meta.get("finished") is True and meta.get("launcher_status") == "finished", run
    assert meta.get("stop_reason") == "converged", run
    assert meta.get("learning_rate") == 3e-4, run
    assert meta.get("seed") == int(run.name.rsplit("seed", 1)[1]), run
    assert actual == expected, (run, actual, expected)
    actor = "gnn_actor.pth" if arm == "C" else "mappo_actor.pth"
    critic = "gnn_critic.pth" if arm == "C" else "mappo_critic.pth"
    assert (run / actor).is_file() and (run / critic).is_file(), run
print(f"validated {len(sys.argv)-1} final-checkpoint runs")
' "${runs[@]}"

if [[ -e "$OUT" ]]; then
  echo "FATAL: output already exists: $OUT" >&2
  exit 1
fi

python run/eval_frozen.py "${runs[@]}" \
  --episodes 100 --seed0 900000 --max_steps 500 --out "$OUT" \
  >"$LOG" 2>&1

rows=$(($(wc -l < "$OUT") - 1))
if [[ "$rows" -ne 15 ]]; then
  echo "FATAL: expected 15 evaluation rows, found $rows" >&2
  exit 1
fi

echo "R2.3 HELD-OUT EVALUATION COMPLETE: $OUT"
