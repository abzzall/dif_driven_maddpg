import torch
import torch.nn as nn
import torch.optim as optim
import os



class SharedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents, hidden_dim=None,
                 activation=nn.ReLU, device='cpu', lr=1e-3, chckpnt_file='shared_critic.pth'):
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

    def forward(self, state, joint_action):
        # Inputs:
        #   state:        [B, state_dim]
        #   joint_action: [B, N * act_dim]  (concatenated actions)
        state = state.to(self.device)
        joint_action = joint_action.to(self.device)

        x = torch.cat([state, joint_action], dim=-1)  # [B, state_dim + N * act_dim]
        q_values = self.q_net(x)                      # [B, N]
        return q_values
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

    def load_checkpoint(self, filepath):
        """
        Loads model and optimizer state from a checkpoint file.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Checkpoint file '{filepath}' not found.")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.to(self.device)
        print(f"Checkpoint loaded from {filepath}")