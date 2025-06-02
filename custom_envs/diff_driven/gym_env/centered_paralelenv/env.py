from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from config import (
    env_name, num_agents, obs_dim, action_dim, obs_low, obs_high,
    act_low, act_high, render_mode,
    env_size, num_obstacles, v_lin_max, v_ang_max, dv_lin_max,
    dv_ang_max, agent_radius, safe_dist, sens_range, max_steps,
    obstacle_size_min, obstacle_size_max
)


class DiffDriveParallelEnv(ParallelEnv):
    metadata = {"render_modes": [render_mode], "name": env_name}

    def __init__(self):
        super().__init__()
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.possible_agents = self.agents[:]

        # World settings from config
        self.env_size = env_size
        self.num_landmarks = num_agents
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
        pass

    def close(self):
        pass

    def state(self):
        pass

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
        pass  # Update linear/angular velocity based on dV inputs and clamp

    def _update_positions(self):
        pass  # Move agents based on current velocities and orientations

    def _handle_collisions(self):
        pass  # Implement agent-agent and agent-obstacle boundary collisions

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
        pass  # Use Hungarian algorithm and local collision penalties
