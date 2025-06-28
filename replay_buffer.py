import torch
from typing import Union, Tuple
from config import (
    num_agents, replay_buffer_size, device, batch_size
)

class ReplayBuffer:
    def __init__(
            self,
            obs_dim: int,  # Dimension of each agent's observation
            action_dim: int,  # Dimension of each agent's action
            state_dim: int,  # Dimension of global state
            num_agents: int = num_agents,  # Total number of agents
            replay_buffer_size: int = replay_buffer_size,  # Max buffer capacity
            device: Union[str, torch.device] = device  # Device where tensors are stored
    ):
        """
        Initializes a replay buffer for multi-agent off-policy RL.

        Args:
            obs_dim (int): Dimensionality of each agent's local observation.
            action_dim (int): Dimensionality of each agent's action.
            state_dim (int): Dimensionality of the global state.
            num_agents (int): Number of agents in the environment.
            replay_buffer_size (int): Maximum number of transitions to store.
            device (str or torch.device): Device for storing sampled batches (e.g., 'cuda', 'cpu').

        Buffers (shapes):
            - obs_buf:         [max_size, num_agents, obs_dim]
            - act_buf:         [max_size, num_agents, action_dim]
            - reward_buf:      [max_size, num_agents]
            - next_obs_buf:    [max_size, num_agents, obs_dim]
            - done_buf:        [max_size, num_agents]
            - state_buf:       [max_size, state_dim]
            - next_state_buf:  [max_size, state_dim]
        """
        self.obs_buf = torch.zeros((replay_buffer_size, num_agents, obs_dim), dtype=torch.float32)
        self.act_buf = torch.zeros((replay_buffer_size, num_agents, action_dim), dtype=torch.float32)
        self.reward_buf = torch.zeros((replay_buffer_size, num_agents), dtype=torch.float32)
        self.next_obs_buf = torch.zeros((replay_buffer_size, num_agents, obs_dim), dtype=torch.float32)
        self.done_buf = torch.zeros((replay_buffer_size, num_agents), dtype=torch.float32)
        self.state_buf = torch.zeros((replay_buffer_size, state_dim), dtype=torch.float32)
        self.next_state_buf = torch.zeros((replay_buffer_size, state_dim), dtype=torch.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = replay_buffer_size
        self.device = torch.device(device)

    def add(
            self,
            state: torch.Tensor,  # shape: [state_dim]
            observations: torch.Tensor,  # shape: [num_agents, obs_dim]
            actions: torch.Tensor,  # shape: [num_agents, action_dim]
            rewards: torch.Tensor,  # shape: [num_agents]
            next_state: torch.Tensor,  # shape: [state_dim]
            next_observations: torch.Tensor,  # shape: [num_agents, obs_dim]
            dones: torch.Tensor  # shape: [num_agents], dtype: bool or float
    ) -> None:
        self.obs_buf[self.ptr] = observations
        self.act_buf[self.ptr] = actions
        self.reward_buf[self.ptr] = rewards
        self.next_obs_buf[self.ptr] = next_observations
        self.done_buf[self.ptr] = dones.float()
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int = batch_size) -> Tuple[
        torch.Tensor,  # obs_buf: [B, N, obs_dim]
        torch.Tensor,  # next_obs_buf: [B, N, obs_dim]
        torch.Tensor,  # state_buf: [B, state_dim]
        torch.Tensor,  # next_state_buf: [B, state_dim]
        torch.Tensor,  # act_buf: [B, N, action_dim]
        torch.Tensor,  # reward_buf: [B, N]
        torch.Tensor  # done_buf: [B, N]
    ]:
        idx = torch.randint(0, self.size, (batch_size,))
        return (
            self.obs_buf[idx].to(self.device),               # observation
            self.next_obs_buf[idx].to(self.device),     # next observation (o')
            self.state_buf[idx].to(self.device),           # state
            self.next_state_buf[idx].to(self.device),        # next state (s')
            self.act_buf[idx].to(self.device),            # action
            self.reward_buf[idx].to(self.device),            #reward
            self.done_buf[idx].to(self.device),             # done
        )


    def save(self, filepath):
        torch.save({
            'obs': self.obs_buf[:self.size],
            'action': self.act_buf[:self.size],
            'reward': self.reward_buf[:self.size],
            'next_obs': self.next_obs_buf[:self.size],
            'done': self.done_buf[:self.size],
            'state': self.state_buf[:self.size],
            'next_state': self.next_state_buf[:self.size],
            'size': self.size,
            'ptr': self.ptr
        }, filepath)

    def load(self, filepath):
        data = torch.load(filepath, map_location=self.device)
        self.size = data['size']
        self.ptr = data['ptr']
        self.obs_buf[:self.size] = data['obs']
        self.act_buf[:self.size] = data['action']
        self.reward_buf[:self.size] = data['reward']
        self.next_obs_buf[:self.size] = data['next_obs']
        self.done_buf[:self.size] = data['done']
        self.state_buf[:self.size] = data['state']
        self.next_state_buf[:self.size] = data['next_state']

    def filled(self):
        return self.size >= self.max_size

    def __len__(self):
        return self.size
