"""Permutation-invariant graph networks for the on-policy coverage study.

The scene is a set of typed nodes -- agents and landmarks -- not a vector. The
previous flat encoding sorted agents by distance to the origin and concatenated
them in index order, which is discontinuous (two agents swapping rank moves the
input by a jump) and not permutation-invariant (one world state has many
encodings). Both cost an on-policy learner disproportionately, because it has
far fewer updates over which to average the inconsistency away.

These modules encode the scene with masked attention over node sets, so:

* the output is invariant to the order of the nodes;
* the team size is a tensor dimension rather than an architectural constant, so
  one trained policy runs unchanged at any number of agents or landmarks -- the
  property that makes zero-shot transfer across team size possible;
* finished agents and covered landmarks are masked out rather than encoded as
  special values.

Everything is plain torch: no torch_geometric dependency. With modest node
counts, dense masked attention is both simpler and faster on GPU than gather /
scatter message passing.

Interfaces mirror ``models/mappo_nets.py`` (``act``, ``evaluate_actions``,
``save_checkpoint``, ``load_checkpoint``) so the training loop is unchanged
apart from passing a dict of node tensors instead of a flat observation.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from config import device as DEFAULT_DEVICE

GNN_LR = 3e-4
INIT_STD = 0.3


class SceneEncoder(nn.Module):
    """Masked multi-head attention over agent and landmark nodes.

    Agents and landmarks are projected into a shared embedding space by
    type-specific linear layers (they carry different features), concatenated
    into one token sequence, and attended over. Each agent's output token is
    contextualised by every visible node, so the representation of "my
    situation" already includes the other agents and the remaining landmarks.

    Padding//masking: tokens for finished agents and covered landmarks are
    excluded from attention as *keys*, so they cannot influence others, but
    agent rows are still produced as *queries* -- the loop needs an embedding
    for every agent index, and masking the query rows instead would make the
    output shape depend on how many agents happen to be active.
    """

    def __init__(
        self,
        agent_features: int,
        landmark_features: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        use_entity_mask: bool = True,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim
        self.use_entity_mask = use_entity_mask

        self.agent_proj = nn.Linear(agent_features, embed_dim)
        self.landmark_proj = nn.Linear(landmark_features, embed_dim)
        # Learned type embeddings: without them the attention cannot tell an
        # agent token from a landmark token once both are in the shared space.
        self.type_embed = nn.Parameter(torch.zeros(2, embed_dim))

        self.layers = nn.ModuleList(
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(num_layers))
        self.ffns = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
            for _ in range(num_layers)
        )
        self.ffn_norms = nn.ModuleList(
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        )
        self.to(self.device)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return per-agent embeddings [B, N, embed_dim].

        Accepts either a single scene (``[N, F]``) or a batch (``[B, N, F]``).
        """
        a = obs["agent_nodes"]
        lm = obs["landmark_nodes"]
        a_mask = obs["agent_mask"]
        lm_mask = obs["landmark_mask"]

        squeeze_batch = a.dim() == 2
        if squeeze_batch:
            a, lm = a.unsqueeze(0), lm.unsqueeze(0)
            a_mask, lm_mask = a_mask.unsqueeze(0), lm_mask.unsqueeze(0)

        B, N, _ = a.shape
        L = lm.shape[1]

        at = self.agent_proj(a) + self.type_embed[0]
        lt = self.landmark_proj(lm) + self.type_embed[1]
        tokens = torch.cat([at, lt], dim=1)                      # [B, N+L, E]

        # key_padding_mask marks positions to IGNORE.
        ignore = torch.cat([~a_mask, ~lm_mask], dim=1)           # [B, N+L]
        if not self.use_entity_mask:
            ignore = torch.zeros_like(ignore)
        # A row that ignores everything yields NaN from softmax; keep such
        # rows fully visible instead (it only happens once the episode is over).
        all_ignored = ignore.all(dim=1, keepdim=True)
        ignore = ignore & ~all_ignored

        x = tokens
        for attn, norm, ffn, fnorm in zip(
            self.layers, self.norms, self.ffns, self.ffn_norms
        ):
            h, _ = attn(x, x, x, key_padding_mask=ignore, need_weights=False)
            x = norm(x + h)
            x = fnorm(x + ffn(x))

        out = x[:, :N, :]                                        # agent tokens
        return out.squeeze(0) if squeeze_batch else out


class DeepSetSceneEncoder(nn.Module):
    """Permutation-equivariant entity encoder with masked mean pooling.

    This control retains typed entities and completed-entity masking while
    replacing pairwise attention with token-wise transforms and a pooled scene
    summary.  ``embed_dim=160`` keeps the complete actor--critic parameter count
    within about one percent of the attention configuration.
    """

    def __init__(
        self,
        agent_features: int,
        landmark_features: int,
        embed_dim: int = 160,
        num_layers: int = 2,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim

        self.agent_proj = nn.Linear(agent_features, embed_dim)
        self.landmark_proj = nn.Linear(landmark_features, embed_dim)
        self.type_embed = nn.Parameter(torch.zeros(2, embed_dim))
        self.ffns = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        )
        self.context = nn.Linear(embed_dim * 2, embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.to(self.device)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        a = obs["agent_nodes"]
        lm = obs["landmark_nodes"]
        a_mask = obs["agent_mask"]
        lm_mask = obs["landmark_mask"]

        squeeze_batch = a.dim() == 2
        if squeeze_batch:
            a, lm = a.unsqueeze(0), lm.unsqueeze(0)
            a_mask, lm_mask = a_mask.unsqueeze(0), lm_mask.unsqueeze(0)

        n_agents = a.shape[1]
        at = self.agent_proj(a) + self.type_embed[0]
        lt = self.landmark_proj(lm) + self.type_embed[1]
        x = torch.cat([at, lt], dim=1)

        for ffn, norm in zip(self.ffns, self.norms):
            x = norm(x + ffn(x))

        visible = torch.cat([a_mask, lm_mask], dim=1).unsqueeze(-1)
        weights = visible.to(x.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (x * weights).sum(dim=1) / denom
        pooled = pooled.unsqueeze(1).expand(-1, n_agents, -1)
        out = self.context_norm(
            torch.relu(self.context(torch.cat([x[:, :n_agents], pooled], dim=-1)))
        )
        return out.squeeze(0) if squeeze_batch else out


class GraphGaussianActor(nn.Module):
    """Shared stochastic policy over graph observations."""

    def __init__(
        self,
        agent_features: int,
        landmark_features: int,
        action_dim: int,
        embed_dim: int = 128,
        encoder_type: str = "attention",
        use_entity_mask: bool = True,
        max_action: float = 1.0,
        lr: float = GNN_LR,
        init_std: float = INIT_STD,
        device: str = DEFAULT_DEVICE,
        chckpnt_file: str = "gnn_actor.pth",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.max_action = max_action
        if encoder_type == "attention":
            self.encoder = SceneEncoder(
                agent_features, landmark_features, embed_dim,
                use_entity_mask=use_entity_mask, device=device,
            )
        elif encoder_type == "deepsets":
            self.encoder = DeepSetSceneEncoder(
                agent_features, landmark_features, embed_dim, device=device,
            )
        else:
            raise ValueError(f"Unknown entity encoder: {encoder_type}")
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, action_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(torch.log(torch.tensor(init_std))))
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.chckpnt_file = chckpnt_file
        self.to(self.device)

    def forward(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.head(self.encoder(obs)) * self.max_action
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return mean, std

    def distribution(self, obs) -> torch.distributions.Normal:
        mean, std = self.forward(obs)
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        dist = self.distribution(obs)
        raw = dist.mean if deterministic else dist.sample()
        return raw, dist.log_prob(raw).sum(-1)

    def evaluate_actions(self, obs, action: torch.Tensor):
        dist = self.distribution(obs)
        return dist.log_prob(action.to(self.device)).sum(-1), dist.entropy().sum(-1)

    def save_checkpoint(self, filename: str = None, file_prefix: str = None):
        name = filename or self.chckpnt_file
        if file_prefix:
            name = f"{file_prefix}_{name}"
        torch.save({"model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()}, name)

    def load_checkpoint(self, filename: str = None):
        name = filename or self.chckpnt_file
        if not os.path.exists(name):
            return
        ck = torch.load(name, map_location=self.device, weights_only=False)
        self.load_state_dict(ck["model_state_dict"])
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])


class GraphCentralizedValue(nn.Module):
    """CTDE value head: per-agent values from the shared scene encoding.

    The parent project's critic consumed ``num_agents * obs_dim`` concatenated
    observations, so it was tied to one team size. Here the same encoder is
    applied and a per-agent scalar read off, which keeps the centralised
    information while remaining valid for any number of agents.
    """

    def __init__(
        self,
        agent_features: int,
        landmark_features: int,
        embed_dim: int = 128,
        encoder_type: str = "attention",
        use_entity_mask: bool = True,
        lr: float = GNN_LR,
        device: str = DEFAULT_DEVICE,
        chckpnt_file: str = "gnn_critic.pth",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        if encoder_type == "attention":
            self.encoder = SceneEncoder(
                agent_features, landmark_features, embed_dim,
                use_entity_mask=use_entity_mask, device=device,
            )
        elif encoder_type == "deepsets":
            self.encoder = DeepSetSceneEncoder(
                agent_features, landmark_features, embed_dim, device=device,
            )
        else:
            raise ValueError(f"Unknown entity encoder: {encoder_type}")
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.chckpnt_file = chckpnt_file
        self.to(self.device)

    def forward(self, obs) -> torch.Tensor:
        """Values [B, N] (or [N] for a single unbatched scene)."""
        return self.head(self.encoder(obs)).squeeze(-1)

    def save_checkpoint(self, filename: str = None, file_prefix: str = None):
        name = filename or self.chckpnt_file
        if file_prefix:
            name = f"{file_prefix}_{name}"
        torch.save({"model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()}, name)

    def load_checkpoint(self, filename: str = None):
        name = filename or self.chckpnt_file
        if not os.path.exists(name):
            return
        ck = torch.load(name, map_location=self.device, weights_only=False)
        self.load_state_dict(ck["model_state_dict"])
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
