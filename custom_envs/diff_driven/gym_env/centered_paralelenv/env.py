from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import torch

from config import (
    env_name, num_agents, obs_low, obs_high,
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
        self.device=device
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


        self.obs_dim=2*self.num_obstacles+2*self.num_landmarks+(self.num_agents-1)*5
        self.action_dim=2
        self.state_dim=5*self.num_agents+2*self.num_landmarks+3*self.num_obstacles

        # Observation and action spaces (remain on CPU)
        self.observation_spaces = {
            agent: spaces.Box(low=obs_low, high=obs_high, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=act_low, high=act_high, shape=(self.action_dim,), dtype=np.float32)
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

    def _reset_episode(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.timestep = 0
        self.agents = self.possible_agents[:]
        # Reinitialize all entities
        self._init_agents()  # fills self.agent_pos, self.agent_vel_lin, etc.
        self._init_landmarks()  # fills self.landmarks
        self._init_obstacles()  # fills self.obstacle_pos, self.obstacle_radius

    def reset_tensor(self, seed=None):
        self._reset_episode(seed)
        state = self.state()
        observations=self.get_all_obs_tensor()
        return state, observations

    def reset(self, seed=None, options=None):
        self._reset_episode()
        # Get observations (in CUDA, then detach+cpu if gym requires)
        observations = self.get_all_obs_dict()
        return observations

    def step(self, actions):
        # Convert actions (dict of np arrays) to a tensor on CUDA
        action_tensor = torch.stack([
            torch.as_tensor(actions[agent], device=device, dtype=torch.float32)
            for agent in self.agents
        ])

        self._make_step(action_tensor)
        # Generate new observations and rewards
        observations=self.get_all_obs_dict()
        rewards_tensor=self._compute_rewards_tensor()
        rewards={agent_id:rewards_tensor[idx] for idx, agent_id in enumerate(self.agents)}
        done=self.done()
        terminations={ agent_id: done for agent_id in self.agents}
        truncations={ agent_id: False for agent_id in self.agents}
        infos={ agent_id: {} for agent_id in self.agents}
        return observations, rewards, terminations, truncations, infos


    def step_tensor(self, actions_tensor):
        self._make_step(actions_tensor)
        state = self.state()
        observations = self.get_all_obs_tensor()
        rewards = self._compute_rewards_tensor()
        dones = torch.full((self.num_agents,),self.timestep >= self.max_steps , dtype=torch.bool)
        return state, observations, rewards, dones

    def _make_step(self, action_tensor):
        # Apply actions, update movement, handle collisions
        self._apply_actions(action_tensor)  # dV_lin and dV_ang
        self._update_positions()  # uses self.agent_vel_lin, self.agent_dir
        self._handle_collisions()  # modifies self.agent_pos if needed
        self.timestep += 1

    def get_all_obs_dict(self):
        return {
            agent_id: self.get_observation(idx)
            for idx, agent_id in enumerate(self.agents)
        }



    def get_all_obs_tensor(self):
        return torch.stack([self.get_observation(idx) for idx in range(self._num_agents)], dim=0)

    def done(self):
        return self.timestep >= self.max_steps

    def get_dones_tensor(self):
        return torch.full((self.num_agents,),self.timestep >= self.max_steps , dtype=torch.bool)

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
            angle_rad = angle_deg

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
        # === Define reference frame ===
        lm_center = self.landmarks.mean(dim=0)  # (2,)
        lm_vectors = self.landmarks - lm_center  # (L, 2)
        lm_dists = torch.norm(lm_vectors, dim=1)

        # Weighted circular mean angle
        angles = torch.atan2(lm_vectors[:, 1], lm_vectors[:, 0])
        sin_sum = torch.sum(torch.sin(angles) * lm_dists)
        cos_sum = torch.sum(torch.cos(angles) * lm_dists)
        mean_angle = torch.atan2(sin_sum, cos_sum)

        # Rotation matrix to align with weighted direction
        cos_a, sin_a = torch.cos(mean_angle), torch.sin(mean_angle)
        rot_matrix = torch.stack([
            torch.stack([cos_a, sin_a]),
            torch.stack([-sin_a, cos_a])
        ])  # (2, 2)

        # === Landmarks (sorted by distance) ===
        lm_order = torch.argsort(lm_dists)
        lm_rel_pos = (lm_vectors[lm_order] @ rot_matrix.T)  # (L, 2)

        # === Agents (sorted by distance) ===
        ag_vectors = self.agent_pos - lm_center  # (N, 2)
        ag_dists = torch.norm(ag_vectors, dim=1)
        ag_order = torch.argsort(ag_dists)

        ag_rel_pos = (ag_vectors[ag_order] @ rot_matrix.T)  # (N, 2)
        ag_dirs = self.agent_dir[ag_order].unsqueeze(1)  # (N, 1)
        ag_vlin = self.agent_vel_lin[ag_order].unsqueeze(1)  # (N, 1)
        ag_vang = self.agent_vel_ang[ag_order].unsqueeze(1)  # (N, 1)

        ag_state = torch.cat([ag_rel_pos, ag_dirs, ag_vlin, ag_vang], dim=1)  # (N, 5)

        # === Obstacles (sorted by distance) ===
        ob_vectors = self.obstacle_pos - lm_center  # (M, 2)
        ob_dists = torch.norm(ob_vectors, dim=1)
        ob_order = torch.argsort(ob_dists)

        ob_rel_pos = (ob_vectors[ob_order] @ rot_matrix.T)  # (M, 2)
        ob_radii = self.obstacle_radius[ob_order].unsqueeze(1)  # (M, 1)

        ob_state = torch.cat([ob_rel_pos, ob_radii], dim=1)  # (M, 3)

        # === Final state ===
        full_state = torch.cat([
            lm_rel_pos.flatten(),  # 2L
            ag_state.flatten(),  # 5N
            ob_state.flatten()  # 3M
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
                angle = np.random.uniform(-torch.pi, torch.pi, )

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

    def _apply_actions(self, action_tensor):
        dv_lin = action_tensor[:, 0].clamp(-self.dv_lin_max, self.dv_lin_max)
        dv_ang = action_tensor[:, 1].clamp(-self.dv_ang_max, self.dv_ang_max)

        # Update velocities with clamping
        self.agent_vel_lin = (self.agent_vel_lin + dv_lin).clamp(0, self.v_lin_max.item())
        self.agent_vel_ang = (self.agent_vel_ang + dv_ang).clamp(-self.v_ang_max, self.v_ang_max)

    def _update_positions(self):
        # Convert angles to radians
        theta_rad = self.agent_dir

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
        self.agent_dir = (self.agent_dir + self.agent_vel_ang + torch.pi) % (2 * torch.pi) - torch.pi


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

    def get_observation(self, idx):
        pos = self.agent_pos[idx]  # (2,)
        angle_rad = self.agent_dir[idx]  # scalar

        # Local rotation matrix
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)
        rot = torch.stack([
            torch.stack([cos_a, sin_a]),
            torch.stack([-sin_a, cos_a])
        ])  # (2, 2)

        # === Own motion ===
        own_lin = self.agent_vel_lin[idx].unsqueeze(0)  # (1,)
        own_ang = self.agent_vel_ang[idx].unsqueeze(0)  # (1,)

        # === Other agents ===
        mask = torch.arange(self._num_agents, device=device) != idx
        other_pos = self.agent_pos[mask]
        rel_vec = other_pos - pos
        dists = torch.norm(rel_vec, dim=1)
        order = torch.argsort(dists)
        rel_pos = (rel_vec[order] @ rot.T)
        other_dir = self.agent_dir[mask][order]
        rel_dir = other_dir - angle_rad
        rel_dir = torch.atan2(torch.sin(rel_dir), torch.cos(rel_dir))
        lin_vels = self.agent_vel_lin[mask][order]
        ang_vels = self.agent_vel_ang[mask][order]

        # === Landmarks ===
        rel_lm = self.landmarks - pos
        lm_dists = torch.norm(rel_lm, dim=1)
        lm_order = torch.argsort(lm_dists)
        rel_lm_local = (rel_lm[lm_order] @ rot.T)

        # === Obstacles ===
        obs_vec = self.obstacle_pos - pos
        center_dists = torch.norm(obs_vec, dim=1)
        edge_dists = center_dists - self.obstacle_radius
        in_range = edge_dists < self.sens_range
        obs_idx = torch.argsort(edge_dists)
        obs_idx = obs_idx[in_range[obs_idx]]

        edge_dists_in_range = edge_dists[obs_idx]
        obs_vec_local = obs_vec[obs_idx] @ rot.T
        obs_angles = torch.atan2(obs_vec_local[:, 1], obs_vec_local[:, 0])

        # === Pad obstacle distances and angles ===
        pad_len = self.num_obstacles - len(obs_idx)
        if pad_len > 0:
            edge_dists_in_range = torch.cat([
                edge_dists_in_range,
                torch.zeros(pad_len, device=device)
            ])
            obs_angles = torch.cat([
                obs_angles,
                torch.zeros(pad_len, device=device)
            ])

        # === Final observation ===
        return torch.cat([
            own_lin,
            own_ang,
            rel_pos.flatten(),
            rel_dir,
            lin_vels,
            ang_vels,
            rel_lm_local.flatten(),
            edge_dists_in_range,
            obs_angles
        ], dim=0).to(torch.float32)

    def _compute_rewards_tensor(self):
        N = self._num_agents
        device = self.agent_pos.device

        # ----- Hungarian Assignment: agents <-> landmarks -----
        # Done on CPU because scipy does not support GPU tensors
        cost_matrix = torch.cdist(
            self.agent_pos.detach().cpu(), self.landmarks.detach().cpu()
        ).numpy()  # (N, N)

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
        penalties -= 10.0 * penalty_matrix.sum(dim=1)

        # Agent–Obstacle collision penalty (vectorized)
        # Shape: agent_pos: (N, 2), obstacle_pos: (M, 2)
        ap = self.agent_pos.unsqueeze(1)  # (N, 1, 2)
        ob = self.obstacle_pos.unsqueeze(0)  # (1, M, 2)
        dist_ap_ob = torch.norm(ap - ob, dim=2)  # (N, M)
        effective_dist = dist_ap_ob - self.obstacle_radius.unsqueeze(0)  # (N, M)

        close_mask = effective_dist < self.safe_dist  # (N, M)
        penalties -= collision_penalty_scale * torch.sum(
            torch.exp(-effective_dist) * close_mask, dim=1
        )

        # Final rewards: base_reward + individual penalties
        rewards = base_reward + penalties  # shape: (N,)

        return rewards  # 1D tensor of shape (N,)

    def get_list_from_dict_by_agent_id(self, d:dict):
        return torch.stack([
            torch.tensor(d[agent], dtype=torch.float32, device=self.device)
            for agent in self.agents
        ])
    def get_dict_from_list_by_agent_id(self, l:list):
        return {
            f"agent_{i}": l[i]
            for i in range(min(self.num_agents, len(l)))
        }
