import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import Optional
from config import *
from typing import Union
import numpy as np


class SimpleActor(nn.Module):
    """
    Shared actor for interchangeable agents. Outputs deterministic or noisy actions.
    """

    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            hidden_dim: Optional[int] = None,
            max_action: float = 1,  # ← from config.py
            std_scale: float = std_scale,  # ← from config.py
            use_noise: bool = use_noise,  # ← from config.py
            lr: float = actor_lr,  # ← from config.py
            device: str = device,  # ← from config.py
            chckpnt_file: str = 'simple_actor.pth'
    ):

        super().__init__()
        self.device = torch.device(device)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.use_noise = use_noise

        self.std = std_scale * max_action
        self.noise_limit = 3.0 * self.std

        if hidden_dim is None:
            hidden_dim = max(128, observation_dim)

        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        self._init_weights()
        self.chckpnt_file = chckpnt_file

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, mean: bool = False) -> torch.Tensor:
        """
        Forward pass of the actor.

        Args:
            obs (torch.Tensor): Observation of shape [B, obs_dim]
            mean (bool): If True, return deterministic output (no noise)

        Returns:
            actions (torch.Tensor): Action of shape [B, act_dim], clipped
        """
        obs = obs.to(self.device)
        mu = self.net(obs) * self.max_action

        if self.use_noise and not mean:
            noise = self.std * torch.randn_like(mu)
            noise = noise.clamp(-self.noise_limit, self.noise_limit)
            return (mu + noise).clamp(-self.max_action, self.max_action)
        else:
            return mu.clamp(-self.max_action, self.max_action)
    def save_checkpoint(self, filepath=None):
        """
        Saves model and optimizer state to a checkpoint file.
        """
        if filepath is None:
            filepath = self.chckpnt_file
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath=None):
        """
        Loads model and optimizer state from a checkpoint file.
        """
        if filepath is None:
            filepath = self.chckpnt_file

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Checkpoint file '{filepath}' not found.")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.to(self.device)
        print(f"Checkpoint loaded from {filepath}")

    def choose_action(self, observation, use_noise=True, eval_mode=True) -> torch.Tensor:
        """
        Selects action(s) for either a single agent or multiple agents.

        Use cases:
            - Training: per-agent or shared actor for stepping the environment
            - Application: single-agent inference (real-time, deployment)

        Args:
            observation (list, np.ndarray, or torch.Tensor):
                - shape [obs_dim]: for single agent
                - shape [num_agents, obs_dim]: for shared actor

            use_noise (bool): Add noise for exploration (True during training)
            eval_mode (bool): Temporarily switch to eval mode for safe inference

        Returns:
            np.ndarray:
                - shape [action_dim] for single agent
                - shape [num_agents, action_dim] for shared actor
        """

        if not torch.is_tensor(observation):
            observation = torch.tensor(observation, dtype=torch.float32)

        if observation.ndim == 1:
            # Single agent input: [obs_dim]
            observation = observation.unsqueeze(0)  # → [1, obs_dim]
            is_single = True
        elif observation.ndim == 2:
            # Shared actor input: [num_agents, obs_dim]
            is_single = False
        else:
            raise ValueError(f"Invalid observation shape: {observation.shape}. Expected 1D or 2D tensor.")

        observation = observation.to(self.device)

        prev_mode = self.training
        if eval_mode:
            self.eval()

        with torch.no_grad():
            action = self.forward(observation, mean=not use_noise)

        if eval_mode and prev_mode:
            self.train()
        return action[0] if is_single else action
