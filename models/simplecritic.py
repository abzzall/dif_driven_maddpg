import torch
import torch.nn as nn
import torch.optim as optim
import os
from config import *
from typing import Optional, Type



class SharedCritic(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            hidden_dim: Optional[int] = None,
            activation: Type[nn.Module] = nn.ReLU,
            device: str = device,
            lr: float = critic_lr,
            chckpnt_file: str = critic_ckpt
    ):

        super().__init__()
        self.device = torch.device(device)
        input_dim = state_dim + n_agents * action_dim

        if hidden_dim is None:
            hidden_dim = max(128, input_dim)

        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, n_agents)  # per-agent Q-values
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

    def forward(
            self,
            state: torch.Tensor,  # shape: [B, state_dim], device: any
            joint_action: torch.Tensor  # shape: [B, N * act_dim], device: any
    ) -> torch.Tensor:
        """
        Forward pass of the shared critic.

        Args:
            state (torch.Tensor): Global state tensor of shape [B, state_dim].
            joint_action (torch.Tensor): Joint actions of all agents, concatenated, shape [B, N * act_dim].

        Returns:
            torch.Tensor: Q-values for each agent, shape [B, N], on self.device.
        """

        # Ensure tensors are on the correct device
        state = state.to(self.device)  # [B, state_dim], device: self.device
        joint_action = joint_action.to(self.device)  # [B, N * act_dim], device: self.device

        # Concatenate state and joint actions
        x = torch.cat([state, joint_action], dim=-1)  # [B, state_dim + N * act_dim]

        # Pass through the Q-network
        q_values = self.q_net(x)  # [B, N]

        return q_values  # shape: [B, N], device: self.device

    def save_checkpoint(self, filepath: str | None = None) -> None:
        """
        Saves model and optimizer state to a checkpoint file.

        Args:
            filepath (str | None): Optional custom file path. If None, uses default self.chckpnt_file.
        """
        if filepath is None:
            filepath = self.chckpnt_file
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str = None, raise_on_no_file: bool = False) -> None:
        """
        Loads model and optimizer state from a checkpoint file.
        If the file is not found, behavior depends on raise_on_no_file.

        Args:
            filepath (str): Path to the saved checkpoint. If None, uses self.chckpnt_file.
            raise_on_no_file (bool): If True, raises FileNotFoundError when missing.
                                     If False, skips loading and keeps model uninitialized or freshly initialized.
        """
        if filepath is None:
            filepath = self.chckpnt_file

        if not os.path.isfile(filepath):
            if raise_on_no_file:
                raise FileNotFoundError(f"Checkpoint file '{filepath}' not found.")
            else:
                print(f"Checkpoint file '{filepath}' not found. Skipping load.")
                self.to(self.device)
                return

        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.to(self.device)
        print(f"Checkpoint loaded from {filepath}")
