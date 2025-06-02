from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from config import (
    env_name, num_agents, obs_dim, action_dim, obs_low, obs_high,
    act_low, act_high, render_mode,
    env_size, num_obstacles, v_lin_max, v_ang_max, dv_lin_max,
    dv_ang_max, agent_radius, safe_dist, sens_range, max_steps
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
        pass  # Set random initial position and orientation for each agent, zero velocity

    def _init_landmarks(self):
        pass  # Random placement of landmarks with distance constraints

    def _init_obstacles(self):
        pass  # Random placement and random sizes

    def _apply_actions(self, actions):
        pass  # Update linear/angular velocity based on dV inputs and clamp

    def _update_positions(self):
        pass  # Move agents based on current velocities and orientations

    def _handle_collisions(self):
        pass  # Implement agent-agent and agent-obstacle boundary collisions

    def _get_all_observations(self):
        pass  # Compute agent-local observations (agents, landmarks, obstacle distances)

    def _compute_rewards(self):
        pass  # Use Hungarian algorithm and local collision penalties
