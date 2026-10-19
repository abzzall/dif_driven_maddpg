#!/usr/bin/env bash
# On-policy coverage study -- all arms from ONE frozen codebase.
#
# Arms differ only by flags, never by code version, so the ablation is valid:
#   A  legacy reward, flat observations   (standard MAPPO reference)
#   B  new reward,    flat observations   (isolates the reward redesign)
#   C  new reward,    graph observations  (isolates the graph encoder)
#   A_cap  as A but stopped at a fixed episode count (budget-protocol contrast)
#
# A->B is the reward's contribution; B->C is the encoder's.
# Everything else -- seed, team size, network width, PPO hyperparameters -- is
# held fixed. Runs are resume-safe: relaunching continues from the checkpoint.
#
# Tonight: SEEDS=1, N=5. Extend with SEEDS="1 2 3" / N=4,6 on later nights;
# already-finished runs are skipped automatically.
set -u

ROOT="${ROOT:-${PPO_STUDY_ROOT:-$HOME/dif_driven_ppo_study}}"
SEEDS="${SEEDS:-1}"
N="${N:-5}"
PARALLEL="${PARALLEL:-4}"
ARMS="${ARMS:-A B C A_cap}"
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"

cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"
# shellcheck disable=SC1091
source "${VENV:-$PROJ/.venv}/bin/activate" || { echo "FATAL: cannot activate virtualenv (set VENV)"; exit 1; }
command -v python >/dev/null || { echo "FATAL: python missing after activate"; exit 1; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "FATAL: CUDA unavailable"; exit 1; }

mkdir -p "$ROOT/logs"

# Compute caps. The wall-clock cap is meant to be the binding one: the episode
# cap sits high enough that a run stops because it plateaued or ran out of
# night, not because it hit an arbitrary episode count -- which is the artifact
# this protocol exists to avoid.
# Single-instance lock: two launchers at once oversubscribe RAM and the OOM
# killer truncates checkpoints mid-write, which is how runs became unresumable.
LOCKDIR="$ROOT/.launcher.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "FATAL: another launcher holds $LOCKDIR. Refusing to start."; exit 3
fi
trap 'rm -rf "$LOCKDIR"' EXIT INT TERM

MAX_HOURS="${MAX_HOURS:-10}"
MAX_EPISODES="${MAX_EPISODES:-5000}"
CONV="--convergence --conv_max_hours $MAX_HOURS --conv_max_episodes $MAX_EPISODES"

arm_flags() {
  case "$1" in
    A)     echo "--reward legacy --no_done_as_obstacles $CONV" ;;
    B)     echo "--reward new --obs flat  $CONV" ;;
    C)     echo "--reward new --obs graph $CONV" ;;
    B_wide) echo "--reward new --obs flat --mlp_hidden 512 $CONV" ;;
    A_cap) echo "--reward legacy --no_done_as_obstacles --episodes 1000" ;;
    *)     echo "UNKNOWN" ;;
  esac
}

launch() {
  local arm="$1" seed="$2"
  local name="${arm}_n${N}_seed${seed}"
  local out="$ROOT/runs/$name"
  local flags; flags="$(arm_flags "$arm")"
  [ "$flags" = "UNKNOWN" ] && { echo "SKIP  $name (unknown arm)"; return; }

  if [ -f "$out/meta.json" ] && grep -q '"finished": true' "$out/meta.json" 2>/dev/null; then
    echo "SKIP  $name (already finished)"; return
  fi
  mkdir -p "$out"
  echo "START $name  [$flags]"
  # shellcheck disable=SC2086
  python run/train_seeded.py --algorithm mappo --n "$N" --mode full \
      --seed "$seed" --v_ang_max pi2 --no_step_csv --plot_every 50 \
      $flags --out_dir "$out" > "$ROOT/logs/$name.log" 2>&1
  echo "DONE  $name (exit $?)"
}

pids=()
for s in $SEEDS; do
  for arm in $ARMS; do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
    launch "$arm" "$s" &
    pids+=("$!")
  done
done

# Wait on this launcher's own children only.
for p in "${pids[@]}"; do wait "$p"; done
echo "ALL RUNS COMPLETE"
