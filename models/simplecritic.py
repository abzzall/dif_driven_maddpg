import torch
import torch.nn as nn

class SharedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents, hidden_dim=None, activation=nn.ReLU, device='cpu'):
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

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        self.to(self.device)
        self._init_weights()

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
