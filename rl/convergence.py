"""Convergence-based stopping for training runs.

Motivation
----------
Budgeting a run by a fixed episode count conflates "the method converged" with
"the episode cap was reached". That is not neutral across algorithm families:
an off-policy learner performs one gradient update per environment step, while
an on-policy learner performs one update per rollout window, so an equal-episode
budget hands the two families very different amounts of optimization. Stopping
on a *plateau of the task metric* makes the budget a measured outcome of the run
rather than a design constant, and the compute actually consumed is recorded so
it can be reported alongside the result.

Criterion
---------
Track the rolling full-team success rate over the most recent ``window``
episodes. A run is converged when the best rolling SR observed so far has not
improved by at least ``min_delta`` (percentage points) for ``patience``
consecutive episodes, provided at least ``min_episodes`` have elapsed. Two hard
caps bound the compute: ``max_episodes`` and ``max_hours``.

The metric is the success rate rather than the mean return because return is
expressed in reward units and is therefore incomparable across runs that use
different reward decompositions; the success rate is invariant to that choice
and means the same thing on every arm of a comparison.

The monitor is algorithm-agnostic: feed it one boolean per finished episode.
State is serializable so a run resumes without losing its plateau history.
"""

from __future__ import annotations

from typing import List, Optional

# Defaults are deliberately conservative: the pilot on the prior MAPPO runs
# showed 200-episode blocks still gaining +7.3pp on average at episode 1000
# (and up to +28.5pp), so patience must span more than one reporting window.
DEFAULT_WINDOW = 200
DEFAULT_MIN_DELTA = 2.0        # percentage points
DEFAULT_PATIENCE = 400         # episodes without a new best rolling SR
DEFAULT_MIN_EPISODES = 600
DEFAULT_MAX_EPISODES = 2500
DEFAULT_MAX_HOURS = 6.0


class ConvergenceMonitor:
    """Plateau detector over the rolling full-team success rate."""

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        min_delta: float = DEFAULT_MIN_DELTA,
        patience: int = DEFAULT_PATIENCE,
        min_episodes: int = DEFAULT_MIN_EPISODES,
        max_episodes: int = DEFAULT_MAX_EPISODES,
        max_hours: float = DEFAULT_MAX_HOURS,
    ) -> None:
        self.window = int(window)
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.min_episodes = int(min_episodes)
        self.max_episodes = int(max_episodes)
        self.max_hours = float(max_hours)

        self.success_history: List[int] = []
        self.best_sr: float = -1.0
        self.episodes_since_best: int = 0
        self.stop_reason: Optional[str] = None

    # ------------------------------------------------------------------
    @property
    def episodes(self) -> int:
        return len(self.success_history)

    def rolling_sr(self) -> float:
        """Full-team success rate (%) over the last ``window`` episodes."""
        if not self.success_history:
            return 0.0
        tail = self.success_history[-self.window:]
        return 100.0 * sum(tail) / len(tail)

    # ------------------------------------------------------------------
    def update(self, full_team_success: bool, elapsed_hours: float = 0.0) -> bool:
        """Record one episode outcome; return True when the run should stop.

        ``full_team_success`` is True when every agent reached a landmark.
        """
        self.success_history.append(1 if full_team_success else 0)

        # The rolling mean is only meaningful once a full window exists.
        if self.episodes >= self.window:
            sr = self.rolling_sr()
            if sr > self.best_sr + self.min_delta:
                self.best_sr = sr
                self.episodes_since_best = 0
            else:
                # The reference stays pinned at the last confirmed improvement.
                # It must NOT track the running maximum: the rolling mean can
                # move by at most 1/window per episode, so a creeping reference
                # could never be cleared by min_delta and the patience counter
                # would run away while the run was still genuinely improving.
                self.episodes_since_best += 1

        if self.max_hours > 0 and elapsed_hours >= self.max_hours:
            self.stop_reason = "max_hours"
        elif self.episodes >= self.max_episodes:
            self.stop_reason = "max_episodes"
        elif (
            self.episodes >= self.min_episodes
            and self.episodes_since_best >= self.patience
        ):
            self.stop_reason = "converged"

        return self.stop_reason is not None

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Fields recorded into meta.json for the compute-disclosure table."""
        return {
            "stop_reason": self.stop_reason,
            "episodes_used": self.episodes,
            "final_rolling_sr": round(self.rolling_sr(), 3),
            "best_rolling_sr": round(max(self.best_sr, 0.0), 3),
            "episodes_since_best": self.episodes_since_best,
            "convergence_window": self.window,
            "convergence_min_delta": self.min_delta,
            "convergence_patience": self.patience,
            "convergence_min_episodes": self.min_episodes,
            "convergence_max_episodes": self.max_episodes,
            "convergence_max_hours": self.max_hours,
        }

    def state_dict(self) -> dict:
        return {
            "success_history": self.success_history,
            "best_sr": self.best_sr,
            "episodes_since_best": self.episodes_since_best,
            "stop_reason": self.stop_reason,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.success_history = list(sd.get("success_history", []))
        self.best_sr = float(sd.get("best_sr", -1.0))
        self.episodes_since_best = int(sd.get("episodes_since_best", 0))
        self.stop_reason = sd.get("stop_reason")
