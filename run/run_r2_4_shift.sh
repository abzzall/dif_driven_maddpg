#!/usr/bin/env bash
# Reviewer 2.4: distribution-shift robustness (sensing noise, actuation uncertainty).
# Protocol: .ai/PROTOCOL_R2_4_DISTRIBUTION_SHIFT.md
# Evaluation only. No retraining. Workspace-scale shift is covered separately by
# the R2.5 study, which already evaluates five unseen workspace sizes.
set -uo pipefail

PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT="${ROOT:-$PROJ/review_round1_runs/r2_4_shift}"
EPISODES="${EPISODES:-100}"
cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
mkdir -p "$ROOT/cells" "$ROOT/logs"
exec >> "$ROOT/stream.log" 2>&1

LOCAL="$PROJ/review_round1_runs/c_checkpoints_local"
NEW="$PROJ/review_round1_runs/r2_3_additional_seeds/runs"
RUNS=()
for arm in C B B_wide; do
  for s in 1 2 3 4 5;      do RUNS+=("$LOCAL/${arm}_n5_seed$s"); done
  for s in 6 7 8 9 10;     do RUNS+=("$NEW/${arm}_n5_seed$s");   done
done
for r in "${RUNS[@]}"; do
  [ -f "$r/meta.json" ] || { echo "FATAL missing $r"; exit 1; }
done
echo "[$(date '+%F %T')] R2.4 shift study: ${#RUNS[@]} policies, $EPISODES episodes/cell"

# condition  sensor_sigma  actuation_sigma
CONDS=(
  "baseline 0.0 0.0"
  "sensor005 0.005 0.0"
  "sensor010 0.01 0.0"
  "sensor020 0.02 0.0"
  "sensor050 0.05 0.0"
  "act005 0.0 0.05"
  "act010 0.0 0.10"
  "act020 0.0 0.20"
)
for spec in "${CONDS[@]}"; do
  set -- $spec
  name=$1 sn=$2 an=$3
  out="$ROOT/cells/${name}.csv"
  if [ -f "$out" ]; then echo "[$(date '+%F %T')] SKIP  $name"; continue; fi
  echo "[$(date '+%F %T')] START $name (sensor=$sn actuation=$an)"
  "${VENV:-$PROJ/.venv}/bin/python" run/eval_shift.py "${RUNS[@]}" \
      --condition "$name" --sensor-noise "$sn" --actuation-noise "$an" \
      --episodes "$EPISODES" --out "$out" >> "$ROOT/logs/${name}.log" 2>&1 \
    || { echo "[$(date '+%F %T')] FATAL $name"; exit 1; }
  echo "[$(date '+%F %T')] END   $name"
done

"${VENV:-$PROJ/.venv}/bin/python" - "$ROOT" <<'PY'
import sys, glob, pandas as pd
root=sys.argv[1]
d=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{root}/cells/*.csv"))])
d.to_csv(f"{root}/shift_eval.csv", index=False)
print(d.pivot_table(index='condition',columns='configuration',values='success_pct',aggfunc='mean').round(1).to_string())
PY
echo "[$(date '+%F %T')] R2.4 SHIFT STUDY COMPLETE"
