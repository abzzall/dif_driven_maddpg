import torch
import torch.nn as nn
import torch.optim as optim

class SimpleActor(nn.Module):
    """
    Shared actor for interchangeable agents. Outputs deterministic or noisy actions.
    """
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim=None,
        max_action=1.0,
        std_scale=0.3,
        use_noise=True,
        lr=1e-3,
        device='cpu'
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

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, obs, mean=False):
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
