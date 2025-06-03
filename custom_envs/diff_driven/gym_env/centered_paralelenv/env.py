from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from config import (
    env_name, num_agents, obs_dim, action_dim, obs_low, obs_high,
    act_low, act_high, render_mode,
    env_size, num_obstacles, v_lin_max, v_ang_max, dv_lin_max,
    dv_ang_max, agent_radius, safe_dist, sens_range, max_steps,
    obstacle_size_min, obstacle_size_max, collision_penalty_scale
)


class DiffDriveParallelEnv(ParallelEnv):
    metadata = {"render_modes": [render_mode], "name": env_name}

    def __init__(self):
        super().__init__()
        self._num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.possible_agents = self.agents[:]

        # World settings from config
        self.env_size = env_size
        self.num_landmarks = self._num_agents
        self.num_obstacles = num_obstacles
        self.v_lin_max = v_lin_max
        self.v_ang_max = v_ang_max
        self.dv_lin_max = dv_lin_max
        self.dv_ang_max = dv_ang_max
        self.agent_radius = agent_radius
        self.safe_dist = safe_dist
        self.sens_range = sens_range

        self.max_steps = max_steps
        self.timestep = 0

        # Spaces
        self.observation_spaces = {
            agent: spaces.Box(low=obs_low, high=obs_high, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=act_low, high=act_high, shape=(action_dim,), dtype=np.float32)
            for agent in self.agents
        }

        # State containers
        self.agent_states = {}
        self.landmarks = []
        self.obstacles = []

        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.agents = self.possible_agents[:]
        self._init_agents()
        self._init_landmarks()
        self._init_obstacles()
        observations = self._get_all_observations()
        return observations

    def step(self, actions):
        self.timestep += 1
        self._apply_actions(actions)
        self._update_positions()
        self._handle_collisions()

        observations = self._get_all_observations()
        rewards = self._compute_rewards()
        terminations = {agent: self.timestep >= self.max_steps for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos
    def render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.clear()
        self.ax.set_xlim(0, self.env_size)
        self.ax.set_ylim(0, self.env_size)
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Step {self.timestep}")

        # Draw obstacles
        for ob in self.obstacles:
            circle = plt.Circle(ob["pos"], ob["radius"], color='gray', alpha=0.5)
            self.ax.add_patch(circle)

        # Draw landmarks
        for lm in self.landmarks:
            self.ax.plot(lm[0], lm[1], 'rx', markersize=8)

        # Draw agents
        for aid, state in self.agent_states.items():
            pos = state["pos"]
            angle = np.deg2rad(state["angle"])
            circle = plt.Circle(pos, self.agent_radius, color='blue', alpha=0.6)
            self.ax.add_patch(circle)
            dx = self.agent_radius * np.cos(angle)
            dy = self.agent_radius * np.sin(angle)
            self.ax.arrow(pos[0], pos[1], dx, dy, head_width=0.2, head_length=0.2, fc='blue', ec='blue')

        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    def state(self):
        # Geometric center of landmarks
        lm_center = np.mean(self.landmarks, axis=0)

        # Weighted circular mean angle for 0x axis
        vectors = [lm - lm_center for lm in self.landmarks]
        distances = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2([v[1] for v in vectors], [v[0] for v in vectors])
        sin_sum = np.sum(np.sin(angles) * distances)
        cos_sum = np.sum(np.cos(angles) * distances)
        mean_angle = np.arctan2(sin_sum, cos_sum)
        rot_matrix = np.array([
            [np.cos(mean_angle), np.sin(mean_angle)],
            [-np.sin(mean_angle), np.cos(mean_angle)]
        ])

        # Landmarks in rotated frame
        rel_landmarks = [rot_matrix @ (lm - lm_center) for lm in self.landmarks]

        # Obstacles: position and size
        rel_obstacles = [
            np.concatenate([rot_matrix @ (ob["pos"] - lm_center), [ob["radius"]]])
            for ob in self.obstacles
        ]

        # Agents: position, linear speed, angle
        rel_agents = [
            np.concatenate([
                rot_matrix @ (st["pos"] - lm_center),
                [st["v_lin"], np.deg2rad(st["angle"])]
            ])
            for st in self.agent_states.values()
        ]

        full_state = np.concatenate([
            np.array(rel_landmarks).flatten(),
            np.array(rel_obstacles).flatten(),
            np.array(rel_agents).flatten()
        ])
        return full_state.astype(np.float32)

    def _init_agents(self):
        self.agent_states = {}
        placed_positions = []
        for agent in self.agents:
            while True:
                pos = np.random.uniform(agent_radius, env_size - agent_radius, size=2)
                angle = np.random.uniform(0, 360)
                collision = False
                for other_pos in placed_positions:
                    if np.linalg.norm(pos - other_pos) < 2 * agent_radius:
                        collision = True
                        break
                if not collision and all(
                    np.linalg.norm(pos - obs["pos"]) > agent_radius + obs["radius"] for obs in self.obstacles
                ):
                    break
            placed_positions.append(pos)
            self.agent_states[agent] = {
                "pos": pos,
                "angle": angle,
                "v_lin": 0.0,
                "v_ang": 0.0
            }

    def _init_landmarks(self):
        self.landmarks = [
            np.random.uniform(0, env_size, size=2)
            for _ in range(self.num_landmarks)
        ]

    def _init_obstacles(self):
        self.obstacles = []
        for _ in range(self.num_obstacles):
            pos = np.random.uniform(0, env_size, size=2)
            radius = np.random.uniform(obstacle_size_min, obstacle_size_max)
            self.obstacles.append({"pos": pos, "radius": radius})

    def _apply_actions(self, actions):
        for agent, action in actions.items():
            state = self.agent_states[agent]
            dv_lin = np.clip(action[0], -self.dv_lin_max, self.dv_lin_max)
            dv_ang = np.clip(action[1], -self.dv_ang_max, self.dv_ang_max)

            state["v_lin"] = np.clip(state["v_lin"] + dv_lin, 0, self.v_lin_max)
            state["v_ang"] = np.clip(state["v_ang"] + dv_ang, -self.v_ang_max, self.v_ang_max)

    def _update_positions(self):
        for agent in self.agents:
            state = self.agent_states[agent]
            theta_rad = np.deg2rad(state["angle"])
            dx = state["v_lin"] * np.cos(theta_rad)
            dy = state["v_lin"] * np.sin(theta_rad)

            new_pos = state["pos"] + np.array([dx, dy])
            new_pos = np.clip(new_pos, self.agent_radius, self.env_size - self.agent_radius)
            state["pos"] = new_pos
            state["angle"] = (state["angle"] + state["v_ang"]) % 360

    def _handle_collisions(self):
        for i, agent_i in enumerate(self.agents):
            ai = self.agent_states[agent_i]
            for j, agent_j in enumerate(self.agents):
                if i >= j:
                    continue
                aj = self.agent_states[agent_j]
                vec = aj["pos"] - ai["pos"]
                dist = np.linalg.norm(vec)
                if dist < 2 * self.agent_radius:
                    overlap = 2 * self.agent_radius - dist
                    if dist > 0:
                        direction = vec / dist
                        ai["pos"] -= direction * (overlap / 2)
                        aj["pos"] += direction * (overlap / 2)
                    ai["v_lin"] = 0
                    aj["v_lin"] = 0

            for ob in self.obstacles:
                vec = ob["pos"] - ai["pos"]
                dist = np.linalg.norm(vec)
                if dist < self.agent_radius + ob["radius"]:
                    overlap = self.agent_radius + ob["radius"] - dist
                    if dist > 0:
                        direction = -vec / dist
                        ai["pos"] += direction * overlap
                    ai["v_lin"] = 0

    def _get_all_observations(self):
        observations = {}
        for agent_id in self.agents:
            agent = self.agent_states[agent_id]
            pos = agent["pos"]
            angle = np.deg2rad(agent["angle"])

            rot_matrix = np.array([
                [np.cos(angle), np.sin(angle)],
                [-np.sin(angle), np.cos(angle)]
            ])

            # Other agents (excluding self)
            rel_agents = [
                rot_matrix @ (self.agent_states[other]["pos"] - pos)
                for other in self.agents if other != agent_id
            ]

            # Landmarks
            rel_landmarks = [
                rot_matrix @ (lm - pos)
                for lm in self.landmarks
            ]

            # Obstacle distances in sensing range
            obs_dists = []
            for obs in self.obstacles:
                center_dist = np.linalg.norm(pos - obs["pos"])
                edge_dist = center_dist - obs["radius"]
                obs_dists.append(edge_dist if edge_dist < self.sens_range else 0.0)

            flat_obs = np.concatenate([
                np.array(rel_agents).flatten(),
                np.array(rel_landmarks).flatten(),
                np.array(obs_dists)
            ])
            observations[agent_id] = flat_obs.astype(np.float32)
        return observations
    def _compute_rewards(self):
        # Distance cost matrix (agents x landmarks)
        agent_positions = [self.agent_states[a]["pos"] for a in self.agents]
        cost_matrix = np.linalg.norm(
            np.expand_dims(agent_positions, 1) - np.expand_dims(self.landmarks, 0), axis=2
        )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_distance = cost_matrix[row_ind, col_ind].sum()
        base_reward = -total_distance / self._num_agents

        # Local collision penalties
        penalties = {a: 0.0 for a in self.agents}
        for i, aid in enumerate(self.agents):
            ai = self.agent_states[aid]
            for j, bid in enumerate(self.agents):
                if i >= j:
                    continue
                aj = self.agent_states[bid]
                d = np.linalg.norm(ai["pos"] - aj["pos"])
                if d < self.safe_dist:
                    p = 10 * np.exp(-d)
                    penalties[aid] -= p
                    penalties[bid] -= p
            for ob in self.obstacles:
                d = np.linalg.norm(ai["pos"] - ob["pos"]) - ob["radius"]
                if d < self.safe_dist:
                    penalties[aid] -= collision_penalty_scale * np.exp(-d)


        # Total reward
        rewards = {a: base_reward + penalties[a] for a in self.agents}
        return rewards
