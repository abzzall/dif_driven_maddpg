from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import torch
from typing import Union, Optional, Tuple, Dict

from config import (
    env_name, num_agents, obs_low, obs_high,
    act_low, act_high, render_mode,
    env_size, num_obstacles, v_lin_max, v_ang_max, dv_lin_max,
    dv_ang_max, agent_radius, safe_dist, sens_range, max_steps,
    obstacle_size_min, obstacle_size_max, collision_penalty_scale,
    device, normalise
)


class DiffDriveParallelEnv(ParallelEnv):
    metadata = {"render_modes": [render_mode], "name": env_name}

    def __init__(
            self,
            num_agents: int = num_agents,
            obs_low: float = obs_low,
            obs_high: float = obs_high,
            act_low: float = act_low,
            act_high: float = act_high,
            env_size: float = env_size,
            num_obstacles: int = num_obstacles,
            v_lin_max: float = v_lin_max,
            v_ang_max: float = v_ang_max,
            dv_lin_max: float = dv_lin_max,
            dv_ang_max: float = dv_ang_max,
            agent_radius: float = agent_radius,
            safe_dist: float = safe_dist,
            sens_range: float = sens_range,
            max_steps: int = max_steps,
            obstacle_size_min: float = obstacle_size_min,
            obstacle_size_max: float = obstacle_size_max,
            collision_penalty_scale: float = collision_penalty_scale,
            device: Union[str, torch.device] = device,
            normalise=normalise
    ):

        super().__init__()
        self.normalise=normalise
        self.device=device
        self.obstacle_size_min=obstacle_size_min
        self.obstacle_size_max=obstacle_size_max
        self.collision_penalty_scale=collision_penalty_scale
        self._num_agents = num_agents
        self.agents:list[str] = [f"agent_{i}" for i in range(self._num_agents)]
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


        self.action_dim=2
        if normalise:
            self.obs_dim = 3 * self.num_obstacles + 2 * self.num_landmarks + (self.num_agents - 1) * 6 + self.action_dim
            self.state_dim = 6 * self.num_agents + 2 * self.num_landmarks + 3 * self.num_obstacles
        else:
            self.obs_dim=2*self.num_obstacles+2*self.num_landmarks+(self.num_agents-1)*5+self.action_dim
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
        self.score=torch.zeros(self.num_agents, device=device)

    def _reset_episode(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.timestep = 0
        self._init_landmarks()  # fills self.landmarks
        self.agents = self.possible_agents[:]
        # Reinitialize all entities
        self._init_agents()  # fills self.agent_pos, self.agent_vel_lin, etc.
        self._init_obstacles()  # fills self.obstacle_pos, self.obstacle_radius
        self.score=torch.zeros(self.num_agents, device=device)
        self._init_static_state_part()

    def reset_tensor(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self._reset_episode(seed)
        state = self.state_tensor()
        observations=self.get_all_obs_tensor()
        return state, observations

    def reset(self, seed: Optional[int] = None, options=None) -> Dict[str, np.ndarray]:
        self._reset_episode()
        # Get observations (in CUDA, then detach+cpu if gym requires)
        observations = self.get_all_obs_dict()
        return observations

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],  # rewards
        Dict[str, bool],  # terminations
        Dict[str, bool],  # truncations
        Dict[str, dict]  # infos
    ]:
        # Convert actions (dict of np arrays) to a tensor on CUDA
        action_tensor = torch.stack([
            torch.as_tensor(actions[agent], device=self.device, dtype=torch.float32)
            for agent in self.agents
        ])

        rewards_tensor=self._make_step(action_tensor)
        # Generate new observations and rewards
        observations=self.get_all_obs_dict()
        rewards={agent_id:rewards_tensor[idx].item() for idx, agent_id in enumerate(self.agents)}
        done=self.done()
        terminations={ agent_id: done for agent_id in self.agents}
        truncations={ agent_id: False for agent_id in self.agents}
        infos={ agent_id: {} for agent_id in self.agents}
        return observations, rewards, terminations, truncations, infos

    def step_tensor(self, actions_tensor: torch.Tensor) -> Tuple[
        torch.Tensor,  # state: [state_dim]
        torch.Tensor,  # observations: [N, obs_dim]
        torch.Tensor,  # rewards: [N]
        torch.Tensor  # dones: [N], dtype=bool
    ]:
        rewards=self._make_step(actions_tensor)
        state = self.state_tensor()
        observations = self.get_all_obs_tensor()

        dones = torch.full((self.num_agents,),self.timestep >= self.max_steps , dtype=torch.bool)
        return state, observations, rewards, dones

    def _make_step(self, action_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the action tensor to update the environment state.

        Args:
            action_tensor (torch.Tensor): Tensor of shape [num_agents, 2],
                containing (dv_lin, dv_ang) for each agent on the same device.

        Returns:
            torch.Tensor: Per-agent reward vector of shape [num_agents],
                on the same device as the input.
        """
        # Apply actions, update movement, handle collisions
        self._apply_actions(action_tensor)  # dV_lin and dV_ang
        self._update_positions()  # uses self.agent_vel_lin, self.agent_dir
        self._handle_collisions()  # modifies self.agent_pos if needed
        self.timestep += 1
        rewards= self._compute_rewards_tensor()
        self.score += rewards
        return rewards

    def get_all_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns per-agent observations as a dictionary (CPU numpy arrays)."""
        return {
            agent_id: self.get_observation(idx).detach().cpu().numpy()
            for idx, agent_id in enumerate(self.agents)
        }

    def get_all_obs_tensor(self) -> torch.Tensor:
        """Returns stacked per-agent observations as a CUDA tensor of shape [num_agents, obs_dim]."""
        return torch.stack([self.get_observation(idx) for idx in range(self._num_agents)], dim=0)

    def done(self) -> bool:
        """Returns whether episode has reached max steps."""
        return self.timestep >= self.max_steps

    def get_dones_tensor(self) -> torch.Tensor:
        """Returns done mask as a tensor of shape [num_agents], dtype=torch.bool."""
        return torch.full((self.num_agents,),self.timestep >= self.max_steps , dtype=torch.bool)

    def render(self) -> None:
        """
        Renders the environment using matplotlib:
        - Draws agents with direction arrows
        - Obstacles as gray circles
        - Landmarks as red crosses
        """
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

    def _init_static_state_part(self):
        """
        Computes and stores the static part of the global state tensor.

        Includes:
            - Landmark positions: shape [num_landmarks, 2]
            - Obstacle positions and radii: shape [num_obstacles, 3]

        Normalization if self.normalise is True:
            - Positions in [-env_size/2, env_size/2] → divided by (env_size / 2)
            - Radii in [obstacle_size_min, obstacle_size_max] → min-max normalized

        Stores:
            self.static_state_tensor: 1D tensor of shape [2L + 3M], on self.device
        """
        device = self.device
        normalize = self.normalise

        # === 1. Landmarks ===
        lm_dists = torch.norm(self.landmarks, dim=1)
        lm_order = torch.argsort(lm_dists)
        lm_sorted = self.landmarks[lm_order]  # shape: [L, 2]

        if normalize:
            lm_sorted = lm_sorted / (self.env_size / 2)  # → [-1, 1]

        lm_flat = lm_sorted.flatten()

        # === 2. Obstacles ===
        ob_dists = torch.norm(self.obstacle_pos, dim=1)
        ob_order = torch.argsort(ob_dists)
        ob_pos_sorted = self.obstacle_pos[ob_order]  # shape: [M, 2]
        ob_radii_sorted = self.obstacle_radius[ob_order].unsqueeze(1)  # shape: [M, 1]

        if normalize:
            ob_pos_sorted = ob_pos_sorted / (self.env_size / 2)  # → [-1, 1]
            ob_radii_sorted = (ob_radii_sorted - self.obstacle_size_min) / (
                    self.obstacle_size_max - self.obstacle_size_min + 1e-8
            )  # → [0, 1]

        ob_state = torch.cat([ob_pos_sorted, ob_radii_sorted], dim=1).flatten()  # shape: [3M]

        # === Final static state ===
        self.static_state_tensor = torch.cat([lm_flat, ob_state], dim=0).to(torch.float32).to(device)

    def state_tensor(self) -> torch.Tensor:
        """
        Returns the full global state tensor for the current timestep.

        Structure (sorted by distance to origin for consistency):
            [landmarks (2L) | obstacles (3M) | agents (5N)]

        Agent fields:
            - position: (x, y)
            - direction: θ (as sin, cos if normalized)
            - linear velocity
            - angular velocity

        Normalization if self.normalise is True:
            - Positions: divided by (env_size / 2)
            - Velocities: divided by v_lin_max / v_ang_max
            - Angle: converted to sin(θ), cos(θ)

        Returns:
            torch.Tensor: 1D state vector of shape [2L + 3M + (5 or 6)×N], on self.device
        """
        device = self.device
        normalize = self.normalise

        # Sort agents by distance to origin
        ag_dists = torch.norm(self.agent_pos, dim=1)
        ag_order = torch.argsort(ag_dists)

        ag_pos = self.agent_pos[ag_order]  # (N, 2)
        ag_dir = self.agent_dir[ag_order]  # (N,)
        ag_vlin = self.agent_vel_lin[ag_order].unsqueeze(1)  # (N, 1)
        ag_vang = self.agent_vel_ang[ag_order].unsqueeze(1)  # (N, 1)

        if normalize:
            ag_pos = ag_pos / (self.env_size / 2)  # → [-1, 1]
            ag_dir_sin = torch.sin(ag_dir).unsqueeze(1)
            ag_dir_cos = torch.cos(ag_dir).unsqueeze(1)
            ag_vlin = ag_vlin / self.v_lin_max
            ag_vang = ag_vang / self.v_ang_max
            ag_state = torch.cat([ag_pos, ag_dir_sin, ag_dir_cos, ag_vlin, ag_vang], dim=1).flatten()  # (6N,)
        else:
            ag_dir = ag_dir.unsqueeze(1)
            ag_state = torch.cat([ag_pos, ag_dir, ag_vlin, ag_vang], dim=1).flatten()  # (5N,)

        # Combine with static state
        return torch.cat([self.static_state_tensor, ag_state], dim=0).float().to(device)

    def state(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Global state as float32 numpy array on CPU.
        """
        return self.state_tensor().detach().cpu().numpy().astype(np.float32)


    def _init_agents(self):
        """
        Initializes agent positions, directions, and velocities while avoiding collisions
        with other agents and obstacles. All values are stored as CUDA tensors.
        """

        self.agent_pos = []
        self.agent_dir = []
        self.agent_vel_lin = []
        self.agent_vel_ang = []

        placed_positions = []

        for _ in range(self._num_agents):
            while True:
                half = self.env_size.item() / 2 - self.agent_radius.item()
                pos = np.random.uniform(-half, half, size=2)
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
            self.agent_pos.append(torch.tensor(pos, dtype=torch.float32, device=self.device))
            self.agent_dir.append(torch.tensor(angle, dtype=torch.float32, device=self.device))
            self.agent_vel_lin.append(torch.tensor(0.0, dtype=torch.float32, device=self.device))
            self.agent_vel_ang.append(torch.tensor(0.0, dtype=torch.float32, device=self.device))

        self.agent_pos = torch.stack(self.agent_pos)
        self.agent_dir = torch.stack(self.agent_dir)
        self.agent_vel_lin = torch.stack(self.agent_vel_lin)
        self.agent_vel_ang = torch.stack(self.agent_vel_ang)

    def _init_landmarks(self):
        """
        Initializes landmark positions with minimum separation and transforms them
        into a new coordinate system centered at the landmarks' centroid and aligned
        with the principal axis of their spatial distribution.

        After this, all coordinates are assumed to be in the new system,
        and the original world frame is discarded.
        """
        landmarks = []
        attempts = 0
        max_attempts = 1000
        min_dist = 2 * self.agent_radius.item()

        while len(landmarks) < self.num_landmarks and attempts < max_attempts:
            candidate = (torch.rand(2, device=self.device) - 0.5) * self.env_size
            valid = True
            for existing in landmarks:
                if torch.norm(candidate - existing) < min_dist:
                    valid = False
                    break
            if valid:
                landmarks.append(candidate)
            attempts += 1

        if len(landmarks) < self.num_landmarks:
            raise RuntimeError("Failed to place all landmarks with minimum separation.")

        # Stack into a tensor (L, 2) — original frame
        landmarks_tensor = torch.stack(landmarks, dim=0)

        # Define new coordinate system:
        # Step 1: origin = centroid
        origin = landmarks_tensor.mean(dim=0)

        # Step 2: PCA for principal direction
        centered = landmarks_tensor - origin
        cov = centered.T @ centered
        eigvals, eigvecs = torch.linalg.eigh(cov)
        x_axis = eigvecs[:, -1] / torch.norm(eigvecs[:, -1])
        y_axis = torch.stack([-x_axis[1], x_axis[0]])
        rot_matrix = torch.stack([x_axis, y_axis])  # (2, 2)

        # Step 3: Transform landmarks into new global coordinate frame
        landmarks_aligned = centered @ rot_matrix.T  # (L, 2)

        # Save landmarks in new frame
        self.landmarks = landmarks_aligned

    def _init_obstacles(self):
        """
        Initializes obstacle positions and radii uniformly within environment bounds.
        """
        self.obstacle_pos = (torch.rand((self.num_obstacles, 2), device=self.device) - 0.5) * self.env_size
        self.obstacle_radius = (
                torch.rand(self.num_obstacles, device=self.device) * (self.obstacle_size_max - self.obstacle_size_min)
                + self.obstacle_size_min
        )

    def _apply_actions(self, action_tensor: torch.Tensor):
        """
        Applies delta actions to agent velocities (linear and angular),
        with clamping based on environment constraints.
        """
        # Assumes action_tensor ∈ [-1, 1]
        dv_lin = action_tensor[:, 0] * self.dv_lin_max
        dv_ang = action_tensor[:, 1] * self.dv_ang_max

        # Update velocities with clamping
        self.agent_vel_lin = (self.agent_vel_lin + dv_lin).clamp(0, self.v_lin_max.item())
        self.agent_vel_ang = (self.agent_vel_ang + dv_ang).clamp(-self.v_ang_max, self.v_ang_max)

    def _update_positions(self):
        """
        Updates agent positions and orientations based on current velocities.
        Ensures positions are clamped within environment boundaries.
        """

        # Convert angles to radians
        theta_rad = self.agent_dir

        # Compute deltas
        dx = self.agent_vel_lin * torch.cos(theta_rad)
        dy = self.agent_vel_lin * torch.sin(theta_rad)
        delta = torch.stack([dx, dy], dim=1)  # shape: (N, 2)

        # Update positions
        self.agent_pos = self.agent_pos + delta

        # Clamp to bounds
        half = self.env_size / 2
        self.agent_pos = torch.clamp(
            self.agent_pos,
            min=-half + self.agent_radius,
            max=half - self.agent_radius,
        )

        # Update angle
        self.agent_dir = (self.agent_dir + self.agent_vel_ang + torch.pi) % (2 * torch.pi) - torch.pi


    def _handle_collisions(self):
        """
        Detects and resolves collisions:
        - Between agents (mutual repulsion)
        - Between agents and obstacles (agent repelled from obstacle)

        Sets linear velocity to zero on collision to simulate impact.
        """

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

    def get_observation(self, idx: int) -> torch.Tensor:
        device = self.device
        normalize = self.normalise

        pos = self.agent_pos[idx].to(device)  # (2,)
        heading = self.agent_dir[idx].to(device)

        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        rot = torch.stack([
            torch.stack([cos_h, sin_h]),
            torch.stack([-sin_h, cos_h])
        ])  # (2, 2)

        # Own motion
        own_lin = self.agent_vel_lin[idx].unsqueeze(0).to(device)
        own_ang = self.agent_vel_ang[idx].unsqueeze(0).to(device)
        if normalize:
            own_lin = own_lin / self.v_lin_max
            own_ang = own_ang / self.v_ang_max

        # Other agents
        mask = torch.arange(self._num_agents, device=device) != idx
        rel_pos = (self.agent_pos[mask] - pos) @ rot.T
        rel_dir = self.agent_dir[mask] - heading
        rel_dir = torch.atan2(torch.sin(rel_dir), torch.cos(rel_dir))

        if normalize:
            rel_pos = rel_pos / self.env_size
            rel_dir_sin = torch.sin(rel_dir).unsqueeze(1)
            rel_dir_cos = torch.cos(rel_dir).unsqueeze(1)
        else:
            rel_dir = rel_dir.unsqueeze(1)

        lin_vels = self.agent_vel_lin[mask].unsqueeze(1).to(device)
        ang_vels = self.agent_vel_ang[mask].unsqueeze(1).to(device)
        if normalize:
            lin_vels = lin_vels / self.v_lin_max
            ang_vels = ang_vels / self.v_ang_max

        # Landmarks
        rel_lm = (self.landmarks - pos) @ rot.T
        if normalize:
            rel_lm = rel_lm / self.env_size

        # Obstacles
        obs_vec = self.obstacle_pos - pos
        center_dists = torch.norm(obs_vec, dim=1)
        edge_dists = center_dists - self.obstacle_radius
        in_range = edge_dists < self.sens_range
        obs_idx = torch.argsort(edge_dists)
        obs_idx = obs_idx[in_range[obs_idx]]

        edge_dists = edge_dists[obs_idx]
        if normalize:
            edge_dists = edge_dists / self.sens_range

        obs_vec_local = obs_vec[obs_idx] @ rot.T
        obs_angles = torch.atan2(obs_vec_local[:, 1], obs_vec_local[:, 0])

        if normalize:
            obs_angle_sin = torch.sin(obs_angles)
            obs_angle_cos = torch.cos(obs_angles)

        # Pad obstacles
        pad_len = self.num_obstacles - len(obs_idx)

        def pad(x, value=0):
            if pad_len > 0:
                return torch.cat([x, torch.full((pad_len,), value, device=device)], dim=0)
            return x

        edge_dists = pad(edge_dists)
        if normalize:
            obs_angle_sin = pad(obs_angle_sin)
            obs_angle_cos = pad(obs_angle_cos)
        else:
            obs_angles = pad(obs_angles)

        # Final concat
        components = [
            own_lin, own_ang,
            rel_pos.flatten(),
            rel_dir_sin.flatten(), rel_dir_cos.flatten() if normalize else rel_dir.flatten(),
            lin_vels.flatten(),
            ang_vels.flatten(),
            rel_lm.flatten(),
            edge_dists,
            obs_angle_sin, obs_angle_cos if normalize else obs_angles
        ]

        return torch.cat(components, dim=0).float().to(device)

    def _compute_rewards_tensor(self) -> torch.Tensor:
        """
        Computes rewards for all agents based on:
          - Hungarian assignment to nearest landmarks (minimizing global distance)
          - Penalties for close proximity to other agents
          - Penalties for collisions or closeness to obstacles

        Returns:
            torch.Tensor: A 1D tensor of shape (num_agents,) representing individual rewards.
        """

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
        penalties -= self.collision_penalty_scale * torch.sum(
            torch.exp(-effective_dist) * close_mask, dim=1
        )

        # Final rewards: base_reward + individual penalties
        rewards = base_reward + penalties  # shape: (N,)

        return rewards  # 1D tensor of shape (N,)