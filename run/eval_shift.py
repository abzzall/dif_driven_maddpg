#!/usr/bin/env python3
"""Distribution-shift evaluation: sensing noise and actuation uncertainty.

Reviewer 2.4 asks for robustness to shifts beyond the training distribution.
Workspace scale is covered by run/eval_zeroshot.py --env-size. This script adds
the two remaining physically meaningful axes, both applied at EVALUATION time
only: no policy is retrained and no training distribution is altered.

  sensing noise      Gaussian noise added to the position fields of every entity
                     token before the actor sees them, in the normalised units of
                     the observation (position is divided by env_size/2, so a
                     sigma of 0.05 corresponds to 0.5 length units at the default
                     workspace, i.e. one coverage radius).
  actuation noise    Gaussian noise added to the commanded action, as a fraction
                     of the action bound, after the policy has acted and before
                     the simulator integrates it.

Both observation layouts place x, y first in every entity token (agents carry
x, y, sin, cos, v_lin, v_ang, done; landmarks carry x, y, covered), and the flat
layout is the concatenation of the same per-entity vectors, so one index mask
serves both arms.

Noise is drawn from a generator seeded by (condition, scenario), so the whole
study is reproducible across Python processes and every policy meets the same
perturbation sequence for a given condition and scenario.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_envs.diff_driven.graph_coverage_env import GraphCoverageEnv  # noqa: E402
from rl.mappo import MAPPO  # noqa: E402

AGENT_FEATURES = GraphCoverageEnv.AGENT_FEATURES
LANDMARK_FEATURES = GraphCoverageEnv.LANDMARK_FEATURES


def stable_noise_seed(condition: str, scenario_seed: int) -> int:
    """Stable 63-bit seed for reviewer-facing perturbation replay.

    The policy/run name is deliberately excluded, so every policy sees the same
    perturbation stream for a given condition and scenario. `hashlib` avoids
    Python's per-process hash randomisation.
    """
    payload = f"{condition}:{scenario_seed}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & (
        2**63 - 1
    )


def make_clustered_env(env, n_clusters=2, spread_frac=0.10):
    """Replace uniform landmark sampling with a clustered layout.

    Training samples landmarks uniformly subject to a minimum separation. This
    variant draws a small number of cluster centres and places landmarks around
    them, so the relative geometry seen after the usual PCA alignment and
    rescaling is tighter groups separated by gaps. Count, minimum separation,
    alignment, and rescaling are unchanged, so only the spatial distribution of
    landmarks shifts.
    """
    import torch as _t

    def _clustered(self):
        min_dist = 10 * self.agent_radius.item()
        spread = float(self.env_size) * spread_frac
        pts = []
        for _ in range(4000):
            if len(pts) >= self.num_landmarks:
                break
            centre_idx = int(_t.randint(0, n_clusters, (1,), device=self.device).item())
            angle = float(centre_idx) / n_clusters * 2 * math.pi
            centre = _t.tensor(
                [math.cos(angle), math.sin(angle)], device=self.device
            ) * (float(self.env_size) * 0.25)
            cand = centre + _t.randn(2, device=self.device) * spread
            if all(_t.norm(cand - e) >= min_dist for e in pts):
                pts.append(cand)
        if len(pts) < self.num_landmarks:
            raise RuntimeError("clustered layout could not place all landmarks")
        lm = _t.stack(pts, dim=0)
        origin = lm.mean(dim=0)
        centered = lm - origin
        cov = centered.T @ centered
        _, eigvecs = _t.linalg.eigh(cov)
        x_axis = eigvecs[:, -1] / _t.norm(eigvecs[:, -1])
        y_axis = _t.stack([-x_axis[1], x_axis[0]])
        aligned = centered @ _t.stack([x_axis, y_axis]).T
        min_xy, _u = aligned.min(dim=0)
        max_xy, _u2 = aligned.max(dim=0)
        extent = _t.max(_t.abs(min_xy), _t.abs(max_xy))
        scale = _t.min((self.env_size / 2 - 5) / extent)
        self.landmarks = aligned * scale

    env._init_landmarks = _clustered.__get__(env, type(env))
    return env


def build(run_dir: Path):
    meta = json.loads((run_dir / "meta.json").read_text())
    flat = meta.get("observation") != "graph"
    env = GraphCoverageEnv(
        num_agents=int(meta.get("n", 5)),
        num_obstacles=0,
        v_ang_max=float(meta.get("v_ang_max_float", math.pi / 2)),
        env_size=float(meta.get("env_size", 20.0)),
        flat_observations=flat,
    )
    kwargs = {}
    mlp_hidden = meta.get("mlp_hidden")
    if mlp_hidden is not None:
        # B_wide trains with hidden width 512; without this the constructor
        # defaults to 256 and load_state_dict fails with a shape mismatch.
        kwargs["mlp_hidden"] = int(mlp_hidden)
    algo = MAPPO(env, list(meta["reward_scales_used"]), **kwargs)
    cwd = os.getcwd()
    try:
        os.chdir(run_dir)
        algo.load_checkpoint()
    finally:
        os.chdir(cwd)
    return env, algo, meta, flat


def perturb_obs(obs, sigma, gen, flat, n_agents, n_landmarks, device):
    """Add Gaussian noise to the position fields only, leaving masks intact."""
    if sigma <= 0:
        return obs
    if flat:
        # flat_obs() is [N, 7N+3L]: each row carries the agent's own features,
        # then every other agent in index order, then all landmarks. Every row
        # therefore has the same position-column layout; noise is drawn per
        # (row, field), matching per-robot sensing error.
        out = obs.clone()
        cols = [0, 1]
        for j in range(n_agents - 1):
            base = AGENT_FEATURES + j * AGENT_FEATURES
            cols += [base, base + 1]
        off = AGENT_FEATURES * n_agents
        for j in range(n_landmarks):
            base = off + j * LANDMARK_FEATURES
            cols += [base, base + 1]
        cols = torch.tensor(cols, device=device, dtype=torch.long)
        rows = torch.arange(n_agents, device=device).unsqueeze(1).expand(
            n_agents, cols.numel())
        noise = torch.randn(n_agents, cols.numel(), generator=gen,
                            device=device) * sigma
        out[rows, cols] = out[rows, cols] + noise
        return out
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs.items()}
    for key in ("agent_nodes", "landmark_nodes"):
        t = out[key]
        noise = torch.randn(t.shape[0], 2, generator=gen, device=device) * sigma
        t[:, 0:2] = t[:, 0:2] + noise
    return out


def run_episode(env, algo, seed, max_steps, sensor_sigma, act_sigma, gen, flat):
    _, obs = env.reset_tensor(seed=seed)
    n = env.num_agents
    prev = env.agent_pos.clone()
    path = torch.zeros(n, device=env.device)
    steps = 0
    for _ in range(max_steps):
        seen = perturb_obs(obs, sensor_sigma, gen, flat, n, env.num_landmarks, env.device)
        raw, _ = algo.actor.act(seen, deterministic=True)
        raw = raw.clamp(-algo.max_action, algo.max_action)
        if act_sigma > 0:
            raw = raw + torch.randn(raw.shape, generator=gen, device=raw.device) * (
                act_sigma * algo.max_action)
            raw = raw.clamp(-algo.max_action, algo.max_action)
        _, obs, _, dones = env.step_tensor(raw)
        path += (env.agent_pos - prev).norm(dim=1)
        prev = env.agent_pos.clone()
        steps += 1
        if bool(dones.all()):
            break
    done = int(env.dones.sum().item())
    return dict(success=int(done == n), coverage=done / n, steps=steps,
                horizon_fail=int(steps >= max_steps and done < n),
                path_len=float(path.sum().item()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--sensor-noise", type=float, default=0.0)
    ap.add_argument("--actuation-noise", type=float, default=0.0)
    ap.add_argument("--condition", type=str, required=True)
    ap.add_argument("--landmark-dist", choices=["train", "clustered"], default="train",
                    help="'clustered' replaces uniform landmark sampling with a clustered "
                         "layout, an unseen landmark distribution")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = []
    for run_dir in a.runs:
        env, algo, meta, flat = build(run_dir)
        if a.landmark_dist == "clustered":
            env = make_clustered_env(env)
        name = run_dir.name
        gen = torch.Generator(device=env.device)
        eps = []
        for s in range(a.seed0, a.seed0 + a.episodes):
            gen.manual_seed(stable_noise_seed(a.condition, s))
            eps.append(run_episode(env, algo, s, a.max_steps,
                                   a.sensor_noise, a.actuation_noise, gen, flat))
        rows.append(dict(
            run=name,
            configuration=name.split("_n5_seed")[0],
            seed=int(name.rsplit("seed", 1)[1]),
            condition=a.condition,
            landmark_dist=a.landmark_dist,
            sensor_noise=a.sensor_noise,
            actuation_noise=a.actuation_noise,
            episodes=a.episodes,
            success_pct=100 * np.mean([e["success"] for e in eps]),
            coverage_pct=100 * np.mean([e["coverage"] for e in eps]),
            horizon_fail_pct=100 * np.mean([e["horizon_fail"] for e in eps]),
        ))
        del env, algo
        torch.cuda.empty_cache()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(rows)} rows; condition={a.condition})")


if __name__ == "__main__":
    main()
