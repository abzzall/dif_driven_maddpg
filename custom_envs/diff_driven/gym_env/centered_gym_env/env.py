import gym
import numpy as np
import torch
from gym import spaces
from config import (
    NUM_AGENTS, NUM_OBSTACLES, WORLD_SIZE,
    V_LIN_MAX, V_ANG_MAX, DV_LIN_MAX, DV_ANG_MAX,
    SENSING_RANGE, AGENT_RADIUS, SAFE_DISTANCE, device
)
from scipy.optimize import linear_sum_assignment

def _min_distance_valid(pos, existing, min_dist):
    return all(torch.norm(pos - e) > min_dist for e in existing)

class DiffDriveEnv(gym.Env):
    def __init__(
        self,
        num_agents=NUM_AGENTS,
        num_obstacles=NUM_OBSTACLES,
        world_size=WORLD_SIZE,
        v_lin_max=V_LIN_MAX,
        v_ang_max=V_ANG_MAX,
        dv_lin_max=DV_LIN_MAX,
        dv_ang_max=DV_ANG_MAX,
        sens_range=SENSING_RANGE,
        agent_radius=AGENT_RADIUS,
        safe_dist=SAFE_DISTANCE
    ):
        super().__init__()
        self.num_agents = num_agents
        self.num_landmarks = num_agents
        self.num_obstacles = num_obstacles
        self.world_size = world_size

        self.vLinMax = v_lin_max
        self.vAngMax = v_ang_max
        self.dvLinMax = dv_lin_max
        self.dvAngMax = dv_ang_max
        self.agentRadius = agent_radius
        self.sensRange = sens_range
        self.safeDist = safe_dist

        self.agents = {}

        self.action_space = spaces.Box(
            low=np.array([-self.dvLinMax, -self.dvAngMax], dtype=np.float32),
            high=np.array([self.dvLinMax, self.dvAngMax], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        obs_dim = 2 * (self.num_agents - 1) + 2 * self.num_landmarks + self.num_obstacles
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_world()
        return {f"agent_{i}": self._get_obs(i) for i in range(self.num_agents)}

    def step(self, actions):
        for i, action in actions.items():
            self._apply_action(i, action)

        self._move_agents()

        rewards = {f"agent_{i}": self._get_reward(i) for i in range(self.num_agents)}
        obs = {f"agent_{i}": self._get_obs(i) for i in range(self.num_agents)}
        terminated = {f"agent_{i}": False for i in range(self.num_agents)}
        terminated["__all__"] = False

        return obs, rewards, terminated, {}

    def _initialize_world(self):
        self.agents = {
            f"agent_{i}": {
                "pos": torch.rand(2, device=device) * self.world_size,
                "theta": torch.rand(1, device=device) * 2 * torch.pi - torch.pi,
                "vlin": torch.tensor(0.0, device=device),
                "vang": torch.tensor(0.0, device=device)
            } for i in range(self.num_agents)
        }

        landmarks = []
        while len(landmarks) < self.num_landmarks:
            pos = torch.rand(2, device=device) * self.world_size
            if _min_distance_valid(pos, landmarks, 2 * self.agentRadius):
                landmarks.append(pos)
        self.landmarks = torch.stack(landmarks)

        self.obstacles = [
            {
                "pos": torch.rand(2, device=device) * self.world_size,
                "radius": torch.rand(1, device=device) * 2.5 + 0.5
            }
            for _ in range(self.num_obstacles)
        ]

    def _apply_action(self, agent_id, action):
        agent = self.agents[f"agent_{agent_id}"]
        dv_lin, dv_ang = torch.tensor(action, device=device)
        agent["vlin"] = torch.clamp(agent["vlin"] + dv_lin, 0, self.vLinMax)
        agent["vang"] = torch.clamp(agent["vang"] + dv_ang, -self.vAngMax, self.vAngMax)

    def _move_agents(self):
        updated_positions = {}
        updated_speeds = {}

        for agent_id, agent in self.agents.items():
            dx = agent["vlin"] * torch.cos(agent["theta"])
            dy = agent["vlin"] * torch.sin(agent["theta"])
            move_vec = torch.cat([dx, dy])
            proposed_pos = agent["pos"] + move_vec
            collided = False

            for other_id, other in self.agents.items():
                if other_id == agent_id:
                    continue
                rel_vec = proposed_pos - other["pos"]
                dist = torch.norm(rel_vec)
                if dist < 2 * self.agentRadius:
                    contact_dist = 2 * self.agentRadius
                    ratio = dist / (contact_dist + 1e-6)
                    corrected_move = move_vec * ratio / 2
                    updated_positions[agent_id] = agent["pos"] + corrected_move
                    updated_speeds[agent_id] = torch.tensor(0.0, device=device)
                    collided = True
                    break

            if not collided:
                for obs in self.obstacles:
                    rel_vec = proposed_pos - obs["pos"]
                    dist = torch.norm(rel_vec)
                    if dist < self.agentRadius + obs["radius"]:
                        contact_dist = self.agentRadius + obs["radius"]
                        ratio = dist / (contact_dist + 1e-6)
                        corrected_move = move_vec * ratio
                        updated_positions[agent_id] = agent["pos"] + corrected_move
                        updated_speeds[agent_id] = torch.tensor(0.0, device=device)
                        collided = True
                        break

            if not collided:
                updated_positions[agent_id] = proposed_pos
                updated_speeds[agent_id] = agent["vlin"]

        for agent_id, new_pos in updated_positions.items():
            self.agents[agent_id]["pos"] = new_pos
            self.agents[agent_id]["vlin"] = updated_speeds[agent_id]
            self.agents[agent_id]["vang"] = torch.tensor(0.0, device=device)
            self.agents[agent_id]["theta"] = (self.agents[agent_id]["theta"] + self.agents[agent_id]["vang"] + torch.pi) % (2 * torch.pi) - torch.pi

    def _get_obs(self, agent_idx):
        agent_id = f"agent_{agent_idx}"
        self_pos = self.agents[agent_id]["pos"]
        self_theta = self.agents[agent_id]["theta"]
        rot_matrix = torch.tensor([
            [torch.cos(-self_theta), -torch.sin(-self_theta)],
            [torch.sin(-self_theta),  torch.cos(-self_theta)]
        ], device=device).squeeze()

        agents_rel = []
        for i, agent in self.agents.items():
            if i == agent_id: continue
            delta = agent["pos"] - self_pos
            agents_rel.append(rot_matrix @ delta)

        landmarks_rel = []
        for landmark in self.landmarks:
            delta = landmark - self_pos
            landmarks_rel.append(rot_matrix @ delta)

        obs_dists = []
        for obs in self.obstacles:
            d = torch.norm(obs["pos"] - self_pos) - obs["radius"]
            obs_dists.append(torch.clamp(d, max=self.sensRange))

        obs_tensor = torch.cat(agents_rel + landmarks_rel + obs_dists).float()
        return obs_tensor.cpu().numpy()

    def _get_reward(self, agent_idx):
        agent_id = f"agent_{agent_idx}"
        agent_pos = self.agents[agent_id]["pos"]

        agent_positions = [a["pos"].cpu().numpy() for a in self.agents.values()]
        landmark_positions = self.landmarks.cpu().numpy()
        cost_matrix = np.zeros((self.num_agents, self.num_landmarks))

        for i, agent in enumerate(agent_positions):
            for j, landmark in enumerate(landmark_positions):
                cost_matrix[i, j] = np.linalg.norm(agent - landmark)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        global_reward = -torch.tensor(cost_matrix[row_ind, col_ind].sum(), device=device)

        local_penalty = 0.0
        for other_id, other in self.agents.items():
            if other_id == agent_id:
                continue
            d = torch.norm(agent_pos - other["pos"])
            if d < self.safeDist:
                local_penalty -= 10 * torch.exp(-d)

        for obs in self.obstacles:
            d = torch.norm(agent_pos - obs["pos"]) - obs["radius"]
            if d < self.safeDist:
                local_penalty -= 10 * torch.exp(-d)

        return (global_reward + local_penalty).item()
