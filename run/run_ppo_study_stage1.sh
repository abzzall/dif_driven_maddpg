#!/usr/bin/env bash
# Stage 1 of the on-policy study: does the budget protocol change the result?
#
# Two arms, identical in every respect except how the run is stopped:
#   conv : stop on a plateau of the rolling full-team success rate
#   cap  : stop at 1000 episodes (the protocol used previously)
#
# Everything else -- environment, reward, network, seed -- is held fixed, so
# any difference is attributable to the budget alone.
set -u

ROOT="${ROOT:-${PPO_STUDY_ROOT:-$HOME/dif_driven_ppo_study}}"
SEEDS="${SEEDS:-1 2 3}"
N="${N:-5}"
PARALLEL="${PARALLEL:-3}"
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"

cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ"

# The venv must be activated inside the script: nohup/detached shells do not
# inherit an activation done in the calling terminal.
# shellcheck disable=SC1091
source "${VENV:-$PROJ/.venv}/bin/activate" || { echo "FATAL: cannot activate virtualenv (set VENV)"; exit 1; }
command -v python >/dev/null || { echo "FATAL: python not on PATH after activate"; exit 1; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "FATAL: CUDA not available"; exit 1; }

mkdir -p "$ROOT/logs"

launch() {
  local name="$1"; shift
  local out="$ROOT/runs/$name"
  if [ -f "$out/meta.json" ] && grep -q '"finished": true' "$out/meta.json" 2>/dev/null; then
    echo "SKIP  $name (already finished)"; return
  fi
  mkdir -p "$out"
  echo "START $name -> $out"
  python run/train_seeded.py --algorithm mappo --n "$N" --mode full \
      --v_ang_max pi2 --no_step_csv --plot_every 50 \
      --out_dir "$out" "$@" > "$ROOT/logs/$name.log" 2>&1
  echo "DONE  $name (exit $?)"
}

jobs_running() { jobs -rp | wc -l; }
throttle() { while [ "$(jobs_running)" -ge "$PARALLEL" ]; do wait -n; done; }

for s in $SEEDS; do
  throttle; launch "mappo_n${N}_conv_seed${s}" --seed "$s" --convergence &
  throttle; launch "mappo_n${N}_cap_seed${s}"  --seed "$s" --episodes 1000 &
done

# Wait only on the launcher's own children.
wait
echo "ALL RUNS COMPLETE"
