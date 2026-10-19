#!/usr/bin/env bash
# Density study: are the reported team-size limits a density artifact?
#
# Prior results on this task class fail at BOTH ends -- n=3 (omitted from the
# published work) and n=8/9 -- while n=4..6 works. With the arena fixed at 20,
# area per agent is 133 at n=3 and 44 at n=9 against 80 at n=5, so "team size"
# and "crowding" are confounded. This separates them:
#
#   fixed  : arena stays 20, so density varies with n  (the published setting)
#   scaled : arena = sqrt(n/5)*20, so area per agent is 80 at every n
#
# Prediction: the fixed column is U-shaped (failures at both extremes) and the
# scaled column is flat. A flat scaled column means the limits were an
# experimental artifact, not a property of the method.
#
# n=5 is the reference and is NOT re-run here: arm C at n=5 already exists with
# the identical configuration (fixed and scaled coincide at n=5, env_size 20).
set -u

ROOT="${ROOT:-${PPO_STUDY_ROOT:-$HOME/dif_driven_ppo_study}}"
SEEDS="${SEEDS:-1 2 3 4 5}"
NS="${NS:-3 9}"
MODES="${MODES:-fixed scaled}"
PARALLEL="${PARALLEL:-4}"
MAX_HOURS="${MAX_HOURS:-3}"
MAX_EPISODES="${MAX_EPISODES:-5000}"
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"

cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
# shellcheck disable=SC1091
source "${VENV:-$PROJ/.venv}/bin/activate" || { echo "FATAL: cannot activate virtualenv (set VENV)"; exit 1; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "FATAL: CUDA unavailable"; exit 1; }

mkdir -p "$ROOT/logs"

launch() {
  local n="$1" mode="$2" seed="$3"
  local name="C_n${n}_${mode}_seed${seed}"
  local out="$ROOT/runs/$name"
  local extra=""
  [ "$mode" = "scaled" ] && extra="--scale_env"

  if [ -f "$out/meta.json" ] && grep -q '"finished": true' "$out/meta.json" 2>/dev/null; then
    echo "SKIP  $name (already finished)"; return
  fi
  mkdir -p "$out"
  echo "START $name (n=$n $mode)"
  # shellcheck disable=SC2086
  python run/train_seeded.py --algorithm mappo --n "$n" --mode full --seed "$seed" \
      --v_ang_max pi2 --reward new --obs graph --no_step_csv --plot_every 50 \
      --convergence --conv_max_hours "$MAX_HOURS" --conv_max_episodes "$MAX_EPISODES" \
      $extra --out_dir "$out" > "$ROOT/logs/$name.log" 2>&1
  echo "DONE  $name (exit $?)"
}

pids=()
for n in $NS; do
  for mode in $MODES; do
    for s in $SEEDS; do
      while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
      launch "$n" "$mode" "$s" &
      pids+=("$!")
    done
  done
done
for p in "${pids[@]}"; do wait "$p"; done
echo "DENSITY STUDY COMPLETE"
