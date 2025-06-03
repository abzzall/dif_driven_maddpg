from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import torch

from config import (
    env_name, num_agents, obs_dim, action_dim, obs_low, obs_high,
    act_low, act_high, render_mode,
    env_size, num_obstacles, v_lin_max, v_ang_max, dv_lin_max,
    dv_ang_max, agent_radius, safe_dist, sens_range, max_steps,
    obstacle_size_min, obstacle_size_max, collision_penalty_scale,
    device
)


class DiffDriveParallelEnv(ParallelEnv):
    metadata = {"render_modes": [render_mode], "name": env_name}

    def __init__(self):
        super().__init__()
        self._num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.possible_agents = self.agents[:]

        # World settings (converted to CUDA tensors)
        self.env_size = torch.tensor(env_size, device=device)
        self.num_landmarks = self._num_agents
        self.num_obstacles = num_obstacles
        self.v_lin_max = torch.tensor(v_lin_max, device=device)
        self.v_ang_max = torch.tensor(v_ang_max, device=device)
        self.dv_lin_max = torch.tensor(dv_lin_max, device=device)
        self.dv_ang_max = torch.tensor(dv_ang_max, device=device)
        self.agent_radius = torch.tensor(agent_radius, device=device)
        self.safe_dist = torch.tensor(safe_dist, device=device)
        self.sens_range = torch.tensor(sens_range, device=device)

        self.max_steps = max_steps
        self.timestep = 0

        # Observation and action spaces (remain on CPU)
        self.observation_spaces = {
            agent: spaces.Box(low=obs_low, high=obs_high, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=act_low, high=act_high, shape=(action_dim,), dtype=np.float32)
            for agent in self.agents
        }

        # Agent states (as tensors)
        self.agent_pos = torch.zeros((self._num_agents, 2), device=device)
        self.agent_vel_lin = torch.zeros((self._num_agents,), device=device)
        self.agent_vel_ang = torch.zeros((self._num_agents,), device=device)
        self.agent_dir = torch.zeros((self._num_agents,), device=device)

        # Landmarks and obstacles
        self.landmarks = torch.zeros((self.num_landmarks, 2), device=device)
        self.obstacle_pos = torch.zeros((self.num_obstacles, 2), device=device)
        self.obstacle_radius = torch.zeros((self.num_obstacles,), device=device)

        # Rendering
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.agents = self.possible_agents[:]
        # Reinitialize all entities
        self._init_agents()  # fills self.agent_pos, self.agent_vel_lin, etc.
        self._init_landmarks()  # fills self.landmarks
        self._init_obstacles()  # fills self.obstacle_pos, self.obstacle_radius
        # Get observations (in CUDA, then detach+cpu if gym requires)
        observations = self._get_all_observations()
        return observations

    def step(self, actions):
        self.timestep += 1

        # Convert actions (dict of np arrays) to a tensor on CUDA
        action_tensor = torch.stack(
            [torch.tensor(actions[agent], device=device, dtype=torch.float32) for agent in self.agents]
        )  # shape: (num_agents, 2)
        # Apply actions, update movement, handle collisions
        self._apply_actions(action_tensor)  # dV_lin and dV_ang
        self._update_positions()  # uses self.agent_vel_lin, self.agent_dir
        self._handle_collisions()  # modifies self.agent_pos if needed
        # Generate new observations and rewards
        observations = self._get_all_observations()  # dict of numpy arrays
        rewards = self._compute_rewards()  # dict of floats (per-agent)
        # Termination flags (fixed duration)
        terminations = {agent: self.timestep >= self.max_steps for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.clear()
        self.ax.set_xlim(0, self.env_size.item())
        self.ax.set_ylim(0, self.env_size.item())
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Step {self.timestep}")

        # Draw obstacles
        for i in range(self.num_obstacles):

            pos = self.obstacle_pos[i].detach().cpu().numpy()
            radius = self.obstacle_radius[i].item()
            circle = plt.Circle(pos, radius, color='gray', alpha=0.5)
            self.ax.add_patch(circle)

        # Draw landmarks
        for lm in self.landmarks.detach().cpu().numpy():
            self.ax.plot(lm[0], lm[1], 'rx', markersize=8)

        # Draw agents
        for i in range(self._num_agents):
            pos = self.agent_pos[i].detach().cpu().numpy()
            angle_deg = self.agent_dir[i].item()
            angle_rad = np.deg2rad(angle_deg)

            circle = plt.Circle(pos, self.agent_radius.item(), color='blue', alpha=0.6)
            self.ax.add_patch(circle)

            dx = self.agent_radius.item() * np.cos(angle_rad)
            dy = self.agent_radius.item() * np.sin(angle_rad)
            self.ax.arrow(pos[0], pos[1], dx, dy, head_width=0.2, head_length=0.2, fc='blue', ec='blue')

        plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def state(self):
        # Geometric center of landmarks
        lm_center = self.landmarks.mean(dim=0)

        # === Landmarks ===
        lm_vectors = self.landmarks - lm_center  # (N_lm, 2)
        lm_dists = torch.norm(lm_vectors, dim=1)
        lm_order = torch.argsort(lm_dists)
        lm_sorted = self.landmarks[lm_order]
        vectors = lm_sorted - lm_center  # reassign after sorting

        # Weighted circular mean for x-axis
        distances = torch.norm(vectors, dim=1)
        angles = torch.atan2(vectors[:, 1], vectors[:, 0])
        sin_sum = torch.sum(torch.sin(angles) * distances)
        cos_sum = torch.sum(torch.cos(angles) * distances)
        mean_angle = torch.atan2(sin_sum, cos_sum)

        # Rotation matrix
        cos_a = torch.cos(mean_angle)
        sin_a = torch.sin(mean_angle)
        rot_matrix = torch.stack([
            torch.stack([cos_a, sin_a]),
            torch.stack([-sin_a, cos_a])
        ])  # (2, 2)

        # === Landmarks in rotated frame ===
        rel_landmarks = (vectors @ rot_matrix.T)  # (N_lm, 2)

        # === Obstacles ===
        obs_vectors = self.obstacle_pos - lm_center
        obs_dists = torch.norm(obs_vectors, dim=1)
        obs_order = torch.argsort(obs_dists)
        rel_obstacles = torch.cat([
            (obs_vectors[obs_order] @ rot_matrix.T),  # positions
            self.obstacle_radius[obs_order].view(-1, 1)
        ], dim=1)  # (N_obs, 3)

        # === Agents ===
        agent_vectors = self.agent_pos - lm_center
        agent_dists = torch.norm(agent_vectors, dim=1)
        agent_order = torch.argsort(agent_dists)

        rel_agent_pos = (agent_vectors[agent_order] @ rot_matrix.T)  # (N, 2)
        rel_agent_data = torch.stack([
            self.agent_vel_lin[agent_order],  # (N,)
            torch.deg2rad(self.agent_dir[agent_order])  # (N,)
        ], dim=1)  # (N, 2)

        rel_agents = torch.cat([rel_agent_pos, rel_agent_data], dim=1)  # (N, 4)

        # === Final state ===
        full_state = torch.cat([
            rel_landmarks.flatten(),
            rel_obstacles.flatten(),
            rel_agents.flatten()
        ])

        return full_state.detach().cpu().numpy().astype(np.float32)

    def _init_agents(self):
        self.agent_pos = []
        self.agent_dir = []
        self.agent_vel_lin = []
        self.agent_vel_ang = []

        placed_positions = []

        for _ in range(self._num_agents):
            while True:
                pos = np.random.uniform(self.agent_radius.item(), self.env_size - self.agent_radius.item(), size=2)
                angle = np.random.uniform(0, 360)

                # Check collision with other agents
                collision = False
                for other_pos in placed_positions:
                    if np.linalg.norm(pos - other_pos) < 2 * self.agent_radius.item():
                        collision = True
                        break

                # Check collision with obstacles
                if not collision and all(
                        np.linalg.norm(pos - ob_pos.detach().cpu().numpy()) > self.agent_radius.item() + ob_rad.item()
                        for ob_pos, ob_rad in zip(self.obstacle_pos, self.obstacle_radius)
                ):
                    break

            placed_positions.append(pos)
            self.agent_pos.append(torch.tensor(pos, dtype=torch.float32, device=device))
            self.agent_dir.append(torch.tensor(angle, dtype=torch.float32, device=device))
            self.agent_vel_lin.append(torch.tensor(0.0, dtype=torch.float32, device=device))
            self.agent_vel_ang.append(torch.tensor(0.0, dtype=torch.float32, device=device))

        self.agent_pos = torch.stack(self.agent_pos)
        self.agent_dir = torch.stack(self.agent_dir)
        self.agent_vel_lin = torch.stack(self.agent_vel_lin)
        self.agent_vel_ang = torch.stack(self.agent_vel_ang)

    def _init_landmarks(self):
        self.landmarks = []
        attempts = 0
        max_attempts = 1000
        while len(self.landmarks) < self.num_landmarks and attempts < max_attempts:
            candidate = torch.rand(2, device=device) * self.env_size
            valid = True
            for existing in self.landmarks:
                if torch.norm(candidate - existing) < 2 * self.agent_radius:
                    valid = False
                    break
            if valid:
                self.landmarks.append(candidate)
            attempts += 1

        if len(self.landmarks) < self.num_landmarks:
            raise RuntimeError("Failed to place all landmarks with minimum separation.")

        self.landmarks = torch.stack(self.landmarks)

    def _init_obstacles(self):
        self.obstacle_pos = torch.rand((self.num_obstacles, 2), device=device) * self.env_size
        self.obstacle_radius = (
                torch.rand(self.num_obstacles, device=device) * (obstacle_size_max - obstacle_size_min)
                + obstacle_size_min
        )

    def _apply_actions(self, actions):
        # Extract actions into a tensor in correct order
        action_tensor = torch.tensor(
            [actions[agent] for agent in self.agents],
            dtype=torch.float32,
            device=device
        )  # Shape: (N, 2)

        dv_lin = action_tensor[:, 0].clamp(-self.dv_lin_max, self.dv_lin_max)
        dv_ang = action_tensor[:, 1].clamp(-self.dv_ang_max, self.dv_ang_max)

        # Update velocities with clamping
        self.agent_vel_lin = (self.agent_vel_lin + dv_lin).clamp(0, self.v_lin_max.item())
        self.agent_vel_ang = (self.agent_vel_ang + dv_ang).clamp(-self.v_ang_max, self.v_ang_max)

    def _update_positions(self):
        # Convert angles to radians
        theta_rad = torch.deg2rad(self.agent_dir)

        # Compute deltas
        dx = self.agent_vel_lin * torch.cos(theta_rad)
        dy = self.agent_vel_lin * torch.sin(theta_rad)
        delta = torch.stack([dx, dy], dim=1)  # shape: (N, 2)

        # Update positions
        self.agent_pos = self.agent_pos + delta

        # Clamp to bounds
        self.agent_pos = torch.clamp(
            self.agent_pos,
            min=self.agent_radius,
            max=self.env_size - self.agent_radius,
        )

        # Update angle
        self.agent_dir = (self.agent_dir + self.agent_vel_ang) % 360

    def _handle_collisions(self):
        # --- Agent-Agent Collisions ---
        pos_i = self.agent_pos.unsqueeze(1)  # (N, 1, 2)
        pos_j = self.agent_pos.unsqueeze(0)  # (1, N, 2)
        diff = pos_j - pos_i  # (N, N, 2)
        dists = torch.norm(diff, dim=-1)  # (N, N)

        # Create mask for actual collisions (avoid self)
        N = self.agent_pos.shape[0]
        collision_mask = (dists < 2 * self.agent_radius) & (dists > 0)

        # Process collisions
        i_indices, j_indices = collision_mask.nonzero(as_tuple=True)
        for i, j in zip(i_indices.tolist(), j_indices.tolist()):
            vec = self.agent_pos[j] - self.agent_pos[i]
            dist = torch.norm(vec)
            if dist > 0:
                direction = vec / dist
                overlap = 2 * self.agent_radius - dist
                self.agent_pos[i] -= direction * (overlap / 2)
                self.agent_pos[j] += direction * (overlap / 2)
                self.agent_vel_lin[i] = 0.0
                self.agent_vel_lin[j] = 0.0

        # --- Agent-Obstacle Collisions ---
        for i in range(self._num_agents):
            pos_i = self.agent_pos[i]
            for j in range(self.num_obstacles):
                vec = self.obstacle_pos[j] - pos_i
                dist = torch.norm(vec)
                min_dist = self.agent_radius + self.obstacle_radius[j]
                if dist < min_dist:
                    if dist > 0:
                        direction = -vec / dist
                        overlap = min_dist - dist
                        self.agent_pos[i] += direction * overlap
                    self.agent_vel_lin[i] = 0.0

    def _get_all_observations(self):
        observations = {}

        for idx, agent_id in enumerate(self.agents):
            pos = self.agent_pos[idx]  # (2,)
            angle_rad = torch.deg2rad(self.agent_dir[idx])  # scalar

            # Rotation matrix for local frame (2x2)
            rot = torch.stack([
                torch.stack([torch.cos(angle_rad), torch.sin(angle_rad)]),
                torch.stack([-torch.sin(angle_rad), torch.cos(angle_rad)])
            ])

            # === Relative positions of other agents ===
            mask = torch.arange(len(self.agents), device=pos.device) != idx
            other_pos = self.agent_pos[mask]  # (N-1, 2)
            rel_agents = other_pos - pos.unsqueeze(0)
            agent_dists = torch.norm(rel_agents, dim=1)
            agent_order = torch.argsort(agent_dists)
            rel_agents = (rel_agents[agent_order] @ rot.T)  # (N-1, 2)

            # === Relative positions of landmarks ===
            rel_landmarks = self.landmarks - pos.unsqueeze(0)
            landmark_dists = torch.norm(rel_landmarks, dim=1)
            landmark_order = torch.argsort(landmark_dists)
            rel_landmarks = (rel_landmarks[landmark_order] @ rot.T)  # (L, 2)

            # === Obstacle sensed distances ===
            rel_obstacle = self.obstacle_pos - pos.unsqueeze(0)
            dist_to_obs = torch.norm(rel_obstacle, dim=1)
            edge_dists = dist_to_obs - self.obstacle_radius  # (M,)
            obs_order = torch.argsort(dist_to_obs)
            edge_dists_sorted = edge_dists[obs_order]

            sensed_dists = torch.where(
                edge_dists_sorted < self.sens_range,
                edge_dists_sorted,
                torch.tensor(0.0, device=pos.device)
            )  # (M,)

            # === Final observation vector ===
            flat_obs = torch.cat([
                rel_agents.flatten(),
                rel_landmarks.flatten(),
                sensed_dists
            ]).to(torch.float32)

            observations[agent_id] = flat_obs

        return observations

    def _compute_rewards(self):
        N = self._num_agents
        device = self.agent_pos.device

        # ----- Hungarian Assignment: agents <-> landmarks -----
        agent_pos_np = self.agent_pos.cpu().numpy()
        landmark_np = self.landmarks.cpu().numpy()
        cost_matrix = np.linalg.norm(
            agent_pos_np[:, None, :] - landmark_np[None, :, :], axis=2
        )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_distance = cost_matrix[row_ind, col_ind].sum()
        base_reward = -total_distance / N

        # ----- Collision Penalties -----
        penalties = torch.zeros(N, device=device)

        # Agent–Agent collision penalty
        delta = self.agent_pos.unsqueeze(1) - self.agent_pos.unsqueeze(0)  # (N, N, 2)
        dist_matrix = torch.norm(delta, dim=2)  # (N, N)
        mask = (dist_matrix < self.safe_dist) & (~torch.eye(N, dtype=torch.bool, device=device))
        penalty_matrix = torch.exp(-dist_matrix) * mask  # (N, N)
        penalties -= 10 * penalty_matrix.sum(dim=1)

        # Agent–Obstacle penalty
        for i in range(N):
            dist = torch.norm(self.agent_pos[i] - self.obstacle_pos, dim=1) - self.obstacle_radius  # (M,)
            close = dist < self.safe_dist
            penalties[i] -= collision_penalty_scale * torch.sum(torch.exp(-dist[close]))

        # ----- Combine Base + Penalty -----
        rewards = {
            self.agents[i]: (base_reward + penalties[i].item())
            for i in range(N)
        }
        return rewards

