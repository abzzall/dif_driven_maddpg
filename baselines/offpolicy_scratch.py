"""Isolated off-policy baselines for the MAPPO interface study.

No existing training/model module imports this file.  It implements two
continuous-control controls from scratch:

* MADDPG: shared decentralized actor, centralized joint-observation/action Q.
* IDDPG: shared decentralized actor and shared per-agent local Q.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CentralCritic(nn.Module):
    def __init__(self, n_agents: int, obs_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        joint_dim = n_agents * (obs_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(joint_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs.flatten(1), action.flatten(1)], dim=-1)
        return self.net(x).squeeze(-1)


class LocalCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.obs = torch.empty(capacity, n_agents, obs_dim)
        self.next_obs = torch.empty_like(self.obs)
        self.action = torch.empty(capacity, n_agents, action_dim)
        self.reward = torch.empty(capacity, n_agents)
        self.done = torch.empty(capacity, n_agents, dtype=torch.bool)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.ptr
        self.obs[i].copy_(obs.detach().cpu())
        self.action[i].copy_(action.detach().cpu())
        self.reward[i].copy_(reward.detach().cpu())
        self.next_obs[i].copy_(next_obs.detach().cpu())
        self.done[i].copy_(done.detach().cpu())
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = torch.randint(self.size, (batch_size,))
        return {
            "obs": self.obs[idx].to(device),
            "action": self.action[idx].to(device),
            "reward": self.reward[idx].to(device),
            "next_obs": self.next_obs[idx].to(device),
            "done": self.done[idx].to(device),
        }


@dataclass
class Losses:
    actor: float
    critic: float


class OffPolicyBaseline:
    def __init__(
        self,
        kind: str,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        gamma: float = 0.95,
        tau: float = 0.01,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
    ) -> None:
        if kind not in {"maddpg", "iddpg"}:
            raise ValueError(kind)
        self.kind = kind
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.actor = SharedActor(obs_dim, action_dim).to(device)
        self.actor_target = SharedActor(obs_dim, action_dim).to(device)
        critic_cls = CentralCritic if kind == "maddpg" else LocalCritic
        critic_args = (n_agents, obs_dim, action_dim) if kind == "maddpg" else (obs_dim, action_dim)
        self.critic = critic_cls(*critic_args).to(device)
        self.critic_target = critic_cls(*critic_args).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        action = self.actor(obs.to(self.device))
        if noise_std:
            action = action + noise_std * torch.randn_like(action)
        return action.clamp(-1.0, 1.0)

    def update(self, batch: dict[str, torch.Tensor]) -> Losses:
        obs, action = batch["obs"], batch["action"]
        reward, next_obs, done = batch["reward"], batch["next_obs"], batch["done"]

        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            next_action = next_action * (~done).unsqueeze(-1)
            if self.kind == "maddpg":
                next_q = self.critic_target(next_obs, next_action)
                target = reward.mean(dim=1) + self.gamma * (~done.all(dim=1)).float() * next_q
            else:
                next_q = self.critic_target(next_obs, next_action)
                target = reward + self.gamma * (~done).float() * next_q

        predicted = self.critic(obs, action)
        critic_loss = F.mse_loss(predicted, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        policy_action = self.actor(obs)
        actor_loss = -self.critic(obs, policy_action).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        with torch.no_grad():
            for online, target_net in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
                for p, tp in zip(online.parameters(), target_net.parameters()):
                    tp.lerp_(p, self.tau)
        return Losses(float(actor_loss.detach()), float(critic_loss.detach()))

    def checkpoint(self) -> dict:
        return {
            "kind": self.kind,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_opt.state_dict(),
            "critic_optimizer": self.critic_opt.state_dict(),
        }

    def load_actor(self, checkpoint: dict) -> None:
        self.actor.load_state_dict(checkpoint["actor"])

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.actor.parameters()) + sum(
            p.numel() for p in self.critic.parameters()
        )
