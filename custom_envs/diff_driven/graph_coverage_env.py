"""Graph-structured coverage environment for the on-policy (PPO) study.

Why a new class
---------------
The observation and reward interfaces of ``DiffDriveParallelEnvDone`` were
designed around an off-policy learner and are kept fixed for comparability with
earlier results. This study varies exactly those interfaces, so it gets its own
class rather than another set of flags on the old one. Dynamics, the Hungarian
assignment and the episode bookkeeping are inherited unchanged.

Two departures from the inherited design.

**Observations are typed entity sets rather than a flat vector.** The legacy
parent uses a fixed-cardinality, canonically distance-sorted representation;
the new flat control uses fixed index-order concatenation. Here the scene is
emitted as typed agent and landmark tokens with masks. The actor is
permutation-equivariant with respect to agent re-indexing, and its parameters
do not depend on team size. These are architectural properties, not by
themselves explanations for the measured performance difference.

**The reward is redesigned around the metric being reported.** Three defects in
the inherited 9-component reward motivated this:

1. *The reward penalised proximity to done agents.* Here the learned proximity
   term ranges over active agents only. Completed agents nevertheless remain
   physical bodies in the simulator's geometric non-overlap projection.
2. *No team-completion term.* The environment emits a completion component to
   every agent when the last landmark is covered. Because MAPPO truncates each
   agent's trajectory at its own termination, only agents still active on that
   step include this component in their learning return.
3. *The success signal was swamped.* A one-time +10 for reaching a goal sat
   against a time penalty accumulating to -500 over an episode, plus comparable
   heading and velocity terms. Dense shaping dominated the sparse term that
   actually defines the task. Weights below put the terminal events on top.

Reward components (per agent, fixed order)::

    0 progress    -- reduction in distance to the assigned landmark
    1 goal        -- +1 on the step this agent covers a landmark
    2 team        -- +1 to every agent on the step the team completes
    3 collision   -- proximity penalty, ACTIVE agents only
    4 heading     -- cos(angle to assigned landmark) - 1
    5 time        -- -1 per step while active

As with the parent, ``current_rewards`` is ``[N, C]`` and the algorithm reduces
it by a dot product with a scale vector, so components can be reweighted or
ablated without touching the environment.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from custom_envs.diff_driven.gym_env.centered_paralelenv.env import (
    DiffDriveParallelEnvDone,
)

# Component order is fixed; see module docstring.
COMPONENTS = ("progress", "goal", "team", "collision", "heading", "time")
NUM_COMPONENTS = len(COMPONENTS)

# Terminal events should outweigh the dense shaping, but the shaping still has
# to be strong enough to get an agent to a landmark in the first place -- the
# sparse goal/team terms cannot teach anything until they fire at least once.
#
# The first setting (progress 1.0, heading 0.1, time 0.05) was too weak: both
# arms using it sat at 0/5 agents done after 250 episodes while the legacy
# reward reached 3/5 by episode ~130. This formulation also has no absolute
# distance-to-goal term (the legacy reward carried one at weight 1.0 alongside
# its progress delta), so the guidance rests entirely on progress and heading
# and they need weights to match.
DEFAULT_SCALES = (5.0, 20.0, 50.0, 5.0, 1.0, 0.1)


class GraphCoverageEnv(DiffDriveParallelEnvDone):
    """Differential-drive coverage with graph observations and a task-aligned reward."""

    #: Per-node feature widths, for the encoder to size its input projections.
    AGENT_FEATURES = 7      # x, y, sin, cos, v_lin, v_ang, done
    LANDMARK_FEATURES = 3   # x, y, covered

    def __init__(self, *args, flat_observations: bool = False, **kwargs):
        # Exclude finished agents from the learned proximity penalty. They
        # remain collision bodies in the inherited geometric projection.
        kwargs["done_as_obstacles"] = False
        super().__init__(*args, **kwargs)
        self.num_components = NUM_COMPONENTS
        self._prev_assigned_dist: Optional[torch.Tensor] = None
        self.flat_observations = bool(flat_observations)
        # Flat layout: own features + other agents + all landmarks.
        self.obs_dim = self.AGENT_FEATURES * self._num_agents + \
            self.LANDMARK_FEATURES * self.num_landmarks

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def graph_obs(self) -> Dict[str, torch.Tensor]:
        """Scene as typed node sets with masks.

        Returns a dict of tensors:
          ``agent_nodes``    [N, AGENT_FEATURES]
          ``landmark_nodes`` [L, LANDMARK_FEATURES]
          ``agent_mask``     [N] bool, True where the agent is still active
          ``landmark_mask``  [L] bool, True where the landmark is uncovered

        Positions are scaled by half the arena so features sit near [-1, 1];
        headings are given as (sin, cos) to avoid the wrap discontinuity at
        +/-pi that a raw angle would introduce.
        """
        device = self.device
        half = self.env_size / 2.0

        agent_nodes = torch.stack(
            [
                self.agent_pos[:, 0] / half,
                self.agent_pos[:, 1] / half,
                torch.sin(self.agent_dir),
                torch.cos(self.agent_dir),
                self.agent_vel_lin / self.v_lin_max,
                self.agent_vel_ang / self.v_ang_max,
                self.dones.float(),
            ],
            dim=1,
        ).float()

        landmark_nodes = torch.stack(
            [
                self.landmarks[:, 0] / half,
                self.landmarks[:, 1] / half,
                self.covered.float(),
            ],
            dim=1,
        ).float()

        return {
            "agent_nodes": agent_nodes.to(device),
            "landmark_nodes": landmark_nodes.to(device),
            "agent_mask": (~self.dones).to(device),
            "landmark_mask": (~self.covered).to(device),
        }

    def flat_obs(self) -> torch.Tensor:
        """The same information as ``graph_obs``, flattened in fixed index order.

        This is the fixed-index control arm. It uses the same underlying node
        features and reward, but the complete configurations also differ in
        actor, critic and how completed entities are masked. The comparison
        therefore does not isolate attention alone.

        Layout per agent i: own features, then every other agent in index
        order, then all landmarks. Shape [N, 7*N + 3*L].
        """
        g = self.graph_obs()
        a, lm = g["agent_nodes"], g["landmark_nodes"]
        N = self._num_agents
        rows = []
        for i in range(N):
            others = torch.cat([a[:i], a[i + 1:]], dim=0).reshape(-1)
            rows.append(torch.cat([a[i], others, lm.reshape(-1)], dim=0))
        return torch.stack(rows, dim=0).float()

    def get_all_obs_tensor(self):  # type: ignore[override]
        """Observation in the configured form (dict for graph, tensor for flat)."""
        return self.flat_obs() if self.flat_observations else self.graph_obs()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _reset_episode(self, seed: Optional[int] = None) -> None:
        super()._reset_episode(seed)
        self._prev_assigned_dist = None
        # The parent allocates a 9-wide score accumulator; this formulation has
        # a different component count.
        self.current_score = torch.zeros(
            self._num_agents, NUM_COMPONENTS, device=self.device
        )

    def _compute_rewards_tensor(self) -> torch.Tensor:
        device = self.device
        N = self._num_agents
        components = torch.zeros((N, NUM_COMPONENTS), device=device)

        active = ~self.dones
        (
            assigned_dists,
            covered_agent_indices,
            covered_landmark_indices,
            assigned_landmark_indices,
        ) = self.get_hungarian_distances()

        # --- 0: progress toward the assigned landmark ---------------------
        # Distance reduction since the previous step, normalised by the
        # furthest an agent could have travelled, so the term is O(1).
        if self._prev_assigned_dist is not None:
            delta = self._prev_assigned_dist - assigned_dists
            # A reassignment changes which landmark the distance refers to, so
            # the delta no longer measures progress; suppress those steps
            # rather than feed the learner a spurious jump.
            plausible = delta.abs() <= self.v_lin_max
            components[active & plausible, 0] = (
                delta[active & plausible] / self.v_lin_max
            )
        self._prev_assigned_dist = assigned_dists.clone()

        # --- 1: this agent covered a landmark this step -------------------
        if covered_agent_indices.numel() > 0:
            components[covered_agent_indices, 1] = 1.0
        # Latch the landmark as covered. Without this the flag never persists,
        # the team can never complete, and the goal term re-fires every step
        # instead of once.
        self.covered[covered_landmark_indices] = True

        # --- 2: the team finished this step -------------------------------
        # Emitted to every agent. Earlier-terminated agents no longer have
        # active trajectory samples, so their learning returns do not include
        # this late component.
        if bool(self.covered.all()):
            components[:, 2] = 1.0

        # --- 3: proximity between ACTIVE agents ---------------------------
        # Finished agents are excluded: they sit on solved goals, and
        # penalising proximity to them would push the remaining agents out of
        # the region they still have to reach.
        if int(active.sum()) > 1:
            act_pos = self.agent_pos[active]
            d = torch.cdist(act_pos, act_pos)
            eye = torch.eye(d.shape[0], dtype=torch.bool, device=device)
            d = d.masked_fill(eye, float("inf"))
            close = d < self.safe_dist
            pen = torch.zeros_like(d)
            pen[close] = -torch.exp(-d[close] * 7.0 / self.safe_dist)
            components[active, 3] = pen.sum(dim=1)

        # --- 4: heading alignment with the assigned landmark --------------
        if assigned_landmark_indices.numel() > 0:
            target = self.landmarks[assigned_landmark_indices]
            to_lm = target - self.agent_pos
            to_lm = to_lm / to_lm.norm(dim=1, keepdim=True).clamp(min=1e-8)
            facing = torch.stack(
                [torch.cos(self.agent_dir), torch.sin(self.agent_dir)], dim=1
            )
            cos_sim = (facing * to_lm).sum(dim=1)
            heading = cos_sim - 1.0
            heading[~active] = 0.0
            components[:, 4] = heading

        # --- 5: time ------------------------------------------------------
        components[active, 5] = -1.0

        self.current_rewards = components
        self.old_hungarian = assigned_dists
        # Terminate last, so every component above was computed against the
        # active set as it stood when the actions were taken.
        self.terminate_agents(covered_agent_indices)
        return components

    # ------------------------------------------------------------------
    def step_tensor(self, actions_tensor: torch.Tensor):
        self._make_step(actions_tensor)
        return (
            self.state_tensor(),
            self.get_all_obs_tensor(),
            self.current_rewards,
            self.get_dones_tensor(),
        )

    def reset_tensor(self, seed: Optional[int] = None):
        self._reset_episode(seed)
        return self.state_tensor(), self.get_all_obs_tensor()
