import torch
from config import (
    num_agents, replay_buffer_size, device, batch_size
)

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, state_dim, num_agents=num_agents, replay_buffer_size=replay_buffer_size, device=device):
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

    def add(self, state, observations, actions, rewards, next_state, next_observations, dones):
        self.obs_buf[self.ptr] = observations
        self.act_buf[self.ptr] = actions
        self.reward_buf[self.ptr] = rewards
        self.next_obs_buf[self.ptr] = next_observations
        self.done_buf[self.ptr] = dones
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=batch_size):
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
