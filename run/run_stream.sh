#!/usr/bin/env bash
# Dead-simple sequential runner: one run at a time, in order, no job control.
#
# The parallel launcher's `jobs -rp` / `wait -n` throttle proved unreliable in
# detached non-interactive shells (runs exiting 1 with no traceback). Running N
# copies of THIS script gives N-way parallelism with no shared state and no
# job-control involvement.
#
# Usage:  bash run/run_stream.sh "A:4 A:5 B:4"
set -u

ROOT="${ROOT:-${PPO_STUDY_ROOT:-$HOME/dif_driven_ppo_study}}"
N="${N:-5}"
MAX_HOURS="${MAX_HOURS:-3}"
MAX_EPISODES="${MAX_EPISODES:-5000}"
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"

cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
# shellcheck disable=SC1091
source "${VENV:-$PROJ/.venv}/bin/activate" || { echo "FATAL: venv"; exit 1; }
mkdir -p "$ROOT/logs"

flags_for() {
  case "$1" in
    A)      echo "--reward legacy --no_done_as_obstacles" ;;
    B)      echo "--reward new --obs flat" ;;
    C)      echo "--reward new --obs graph" ;;
    B_wide) echo "--reward new --obs flat --mlp_hidden 512" ;;
    *)      echo "UNKNOWN" ;;
  esac
}

for spec in $1; do
  arm="${spec%%:*}"; seed="${spec##*:}"
  name="${arm}_n${N}_seed${seed}"
  out="$ROOT/runs/$name"
  f="$(flags_for "$arm")"
  [ "$f" = "UNKNOWN" ] && { echo "skip $name: unknown arm"; continue; }
  if [ -f "$out/meta.json" ] && grep -q '"finished": true' "$out/meta.json" 2>/dev/null; then
    echo "[$(date '+%T')] SKIP $name (finished)"; continue
  fi
  rm -f "$out/.run_all.lock"
  mkdir -p "$out"
  echo "[$(date '+%T')] START $name"
  # shellcheck disable=SC2086
  python run/train_seeded.py --algorithm mappo --n "$N" --mode full --seed "$seed" \
      --v_ang_max pi2 --no_step_csv --plot_every 50 \
      --convergence --conv_max_hours "$MAX_HOURS" --conv_max_episodes "$MAX_EPISODES" \
      $f --out_dir "$out" >> "$ROOT/logs/$name.log" 2>&1
  echo "[$(date '+%T')] END   $name (exit $?)"
done
echo "[$(date '+%T')] STREAM COMPLETE"
