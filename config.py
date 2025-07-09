import torch
import numpy as np


import torch

# ====== Environment settings ======
env_name = "diff_drive_multiagent"
render_mode = "human"  # or "none"

# ====== Environment settings ======
# Number of agents (should match number of landmarks)
num_agents = 6

# Number of obstacles (moderate complexity)
num_obstacles = 4  # 1 obstacle per agent for balanced navigation

# Environment size (2D continuous space)
env_size = 100  # Map is 10x10 meters

# Episode length
max_steps = 1000  # Long enough for full behavior to emerge, short enough for stable training

# ====== Observation, State, Action dimensions ======
# obs_dim = 78     # Per-agent observation dimension
# state_dim = 81   # Global state used by the critic
# action_dim = 2   # dVlin and dVang per agent

# Observation and action space bounds
obs_low = -1.0
obs_high = 1.0
act_low = -1.0
act_high = 1.0

# ====== Motion parameters ======
v_lin_max = 1.0          # Max linear velocity
v_ang_max = torch.pi/2        # Max angular velocity (degrees)
dv_lin_max = 0.1 * v_lin_max  # Max delta linear velocity per step
dv_ang_max = torch.pi/12        # Max delta angular velocity per step

# Sensor range (LIDAR)
sens_range = 5 * v_lin_max  # How far agents can detect obstacles

# ====== Reward settings ======
collision_penalty_scale = 30.0  # Multiplied by exp(-d) if d < safe_dist

# ====== Replay Buffer and Training ======
replay_buffer_size = 100_000  # Large enough to avoid overfitting, safe for 6GB GPU (store on CPU)
batch_size = 128               # Optimized for 6GB GPU, adjust if needed
start_training_after = 2048 # 1000   # Sooner training for small buffer

# ====== Network architecture ======
# hidden_dim_actor = 256      # Sufficient for obs_dim=78, 2-layer ReLU net
# hidden_dim_critic = 256     # For input_dim = state_dim + joint_actions = 99
# num_critic_heads = 3        # Average over 3 heads for ensemble stability

# ====== Exploration Noise ======
std_scale = 0.3             # Gaussian noise std = 0.3 * max_action
use_noise = True            # Use noise during training, not evaluation

# CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Agent Size and Safety Settings
agent_radius = v_lin_max
safe_dist = v_lin_max
sens_range = 5 * v_lin_max


obstacle_size_min=1
obstacle_size_max=5

# === Training and Replay ===
gamma = 0.99                        # Discount factor
tau = 0.005                         # For soft update of target networks


#n_games
n_games = 25_000  # Total number of games to train the agents
train_each=100

critic_lr: float = 1e-3
critic_ckpt: str = 'shared_critic.pth'
actor_lr = 1e-3

normalise=True # normalise the observations and state

# === Training Configuration ===



patience = 200                   # Max episodes with no improvement before early stopping
min_episodes_before_early_stop = 100  # Minimum number of episodes before early stopping is considered

score_avg_window = 50            # Number of recent episodes to average for performance evaluation
