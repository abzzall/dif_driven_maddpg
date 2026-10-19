#!/usr/bin/env bash
# Queue seeds 4 and 5 for every arm, starting once the in-flight capacity
# control (B_wide seeds 1-3) has finished.
#
# Running both batches at once would put 7 trainers on a 15 GB machine at
# ~2 GB each, so this waits rather than competing for memory.
set -u

PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG=${PPO_STUDY_ROOT:-$HOME/dif_driven_ppo_study}/logs/ppo_study_seeds45.log

cd "$PROJ" || exit 1

echo "[$(date '+%F %T')] waiting for in-flight runs to finish..."
while ps -C python -o args= 2>/dev/null | grep -q "train_seeded.py"; do
  sleep 60
done
echo "[$(date '+%F %T')] slots free; launching seeds 4-5 for A B C B_wide"

ARMS="A B C B_wide" SEEDS="4 5" PARALLEL=4 MAX_HOURS=3 MAX_EPISODES=5000 \
  bash run/run_ppo_study.sh
echo "[$(date '+%F %T')] seeds 4-5 batch complete"
