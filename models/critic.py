import torch
import torch.nn as nn

class SimpleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=None, activation=nn.ReLU, device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        input_dim = state_dim + action_dim
        if hidden_dim is None:
            hidden_dim = max(128, input_dim)

        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 1)
        )

        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        x = torch.cat([state, action], dim=-1)  # (B, state+action)
        q_value = self.q_net(x)  # (B, 1)
        return q_value
