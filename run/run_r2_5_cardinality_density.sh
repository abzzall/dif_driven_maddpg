#!/usr/bin/env bash
# Reviewer 2.5: wider team-size range with independent density control.
# Protocol: .ai/PROTOCOL_R2_5_CARDINALITY_DENSITY.md
# Evaluation only. No retraining, no checkpoint selection, no policy/reward change.
set -uo pipefail

PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT="${ROOT:-$PROJ/review_round1_runs/r2_5_cardinality_density}"
SIZES="${SIZES:-3 4 5 6 7 8}"
EPISODES="${EPISODES:-100}"

cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
mkdir -p "$ROOT/cells" "$ROOT/logs"

RUNS=()
for s in 1 2 3 4 5; do RUNS+=("$PROJ/review_round1_runs/c_checkpoints_local/C_n5_seed$s"); done
for s in 6 7 8 9 10; do RUNS+=("$PROJ/review_round1_runs/r2_3_additional_seeds/runs/C_n5_seed$s"); done

for r in "${RUNS[@]}"; do
  [ -f "$r/gnn_actor.pth" ] || { echo "FATAL: missing checkpoint $r"; exit 1; }
done

# Condition A: fixed arena (env_size 20) -- density rises with team size.
# Condition B: fixed density (env_size = 20*sqrt(n/5)) -- area per robot held at 80.
for cond in fixed_arena fixed_density; do
  for n in $SIZES; do
    out="$ROOT/cells/${cond}_n${n}.csv"
    if [ -f "$out" ]; then echo "[$(date '+%F %T')] SKIP  $cond n=$n"; continue; fi
    if [ "$cond" = "fixed_arena" ]; then
      envsize_flag=()
    else
      es=$(python3 -c "import math;print(20.0*math.sqrt($n/5.0))")
      envsize_flag=(--env-size "$es")
    fi
    echo "[$(date '+%F %T')] START $cond n=$n ${envsize_flag[*]:-default arena}"
    "${VENV:-$PROJ/.venv}/bin/python" run/eval_zeroshot.py "${RUNS[@]}" \
        --eval-n "$n" --episodes "$EPISODES" "${envsize_flag[@]}" \
        --out "$out" >> "$ROOT/logs/${cond}_n${n}.log" 2>&1 \
      || { echo "[$(date '+%F %T')] FATAL $cond n=$n failed"; exit 1; }
    echo "[$(date '+%F %T')] END   $cond n=$n"
  done
done

"${VENV:-$PROJ/.venv}/bin/python" - "$ROOT" <<'PY'
import sys, glob, os, pandas as pd
root=sys.argv[1]; rows=[]
for f in sorted(glob.glob(f"{root}/cells/*.csv")):
    b=os.path.basename(f)[:-4]; cond=b.rsplit("_n",1)[0]
    d=pd.read_csv(f); d.insert(0,"condition",cond); rows.append(d)
out=pd.concat(rows).sort_values(["condition","eval_n","seed"])
out.to_csv(f"{root}/cardinality_density_eval.csv", index=False)
print(out.groupby(["condition","eval_n"]).success.agg(["mean","std","min"]).round(1).to_string())
PY
echo "[$(date '+%F %T')] R2.5 CARDINALITY/DENSITY COMPLETE"
