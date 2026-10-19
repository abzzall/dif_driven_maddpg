#!/usr/bin/env bash
# Replicate the n=5 result at n=4 and n=6 -- the "does it hold" step that has to
# pass before any extension (density, obstacles, other environments) is worth
# running. Arena stays at the published env_size=20 so these are directly
# comparable to the existing n=5 arms and to the prior published settings.
#
# Arms: C (the proposed method) and A (the standard-MAPPO reference). B and
# B_wide are attribution controls for the n=5 analysis and are not needed to
# establish that the result replicates.
set -u

ROOT="${ROOT:-${PPO_STUDY_ROOT:-$HOME/dif_driven_ppo_study}}"
SEEDS="${SEEDS:-1 2 3 4 5}"
NS="${NS:-4 6}"
ARMS="${ARMS:-C A}"
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

arm_flags() {
  case "$1" in
    A) echo "--reward legacy --no_done_as_obstacles" ;;
    B) echo "--reward new --obs flat" ;;
    C) echo "--reward new --obs graph" ;;
    *) echo "UNKNOWN" ;;
  esac
}

launch() {
  local arm="$1" n="$2" seed="$3"
  local name="${arm}_n${n}_seed${seed}"
  local out="$ROOT/runs/$name"
  local flags; flags="$(arm_flags "$arm")"
  if [ -f "$out/meta.json" ] && grep -q '"finished": true' "$out/meta.json" 2>/dev/null; then
    echo "SKIP  $name (already finished)"; return
  fi
  mkdir -p "$out"
  echo "START $name"
  # shellcheck disable=SC2086
  python run/train_seeded.py --algorithm mappo --n "$n" --mode full --seed "$seed" \
      --v_ang_max pi2 --no_step_csv --plot_every 50 \
      --convergence --conv_max_hours "$MAX_HOURS" --conv_max_episodes "$MAX_EPISODES" \
      $flags --out_dir "$out" > "$ROOT/logs/$name.log" 2>&1
  echo "DONE  $name (exit $?)"
}

# Wait for BOTH any running trainer AND the seeds-4/5 chain to be gone.
# Polling only for idle trainers would race that chain: both would observe an
# idle machine at the same instant and launch together, putting 8 trainers on a
# 15 GB box at ~2 GB each.
echo "[$(date '+%F %T')] waiting for in-flight runs and the seeds-4/5 chain..."
while ps -C python -o args= 2>/dev/null | grep -q "train_seeded.py" \
   || pgrep -f "run_ppo_study_seeds45.sh" >/dev/null \
   || pgrep -f "run_ppo_study.sh" >/dev/null; do
  sleep 60
done
echo "[$(date '+%F %T')] launching n=4,6 replication"

pids=()
for n in $NS; do
  for arm in $ARMS; do
    for s in $SEEDS; do
      while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
      launch "$arm" "$n" "$s" &
      pids+=("$!")
    done
  done
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[$(date '+%F %T')] n=4,6 REPLICATION COMPLETE"
