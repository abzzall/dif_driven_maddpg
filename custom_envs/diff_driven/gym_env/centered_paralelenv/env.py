# env/my_parallel_env.py

from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np

# Import individual config variables
from config import (
    env_name,
    num_agents,
    obs_dim,
    action_dim,
    obs_low,
    obs_high,
    act_low,
    act_high,
    render_mode,
)


class MyParallelEnv(ParallelEnv):
    metadata = {"render_modes": [render_mode], "name": env_name}

    def __init__(self):
        super().__init__()
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.possible_agents = self.agents[:]

        obs_shape = (obs_dim,)
        act_shape = (action_dim,)

        self.observation_spaces = {
            agent: spaces.Box(low=obs_low, high=obs_high, shape=obs_shape, dtype=np.float32)
            for agent in self.agents
        }

        self.action_spaces = {
            agent: spaces.Box(low=act_low, high=act_high, shape=act_shape, dtype=np.float32)
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def state(self):
        pass
