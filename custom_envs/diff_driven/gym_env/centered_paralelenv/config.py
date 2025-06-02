import torch
import numpy as np

# CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic Environment Info
env_name = "diffdrive_parallel_env_v0"
render_mode = "human"

# Agent and Entity Counts
num_agents = 9
num_obstacles = 9

# Environment Size
env_size = 10.0  # 10x10 environment

# Observation and Action Dimensions
obs_dim = 4  # Placeholder, to be updated
action_dim = 2  # dV_lin, dV_ang

# Observation Space Limits
obs_low = -1.0
obs_high = 1.0

# Action Space Limits
act_low = -1.0
act_high = 1.0

# Agent Dynamics Parameters
v_lin_max = 1.0  # Max linear velocity
v_ang_max = 90.0  # Max angular velocity (degrees)
dv_lin_max = 0.1 * v_lin_max  # Max linear velocity adjustment
dv_ang_max = 15.0  # Max angular velocity adjustment

# Agent Size and Safety Settings
agent_radius = v_lin_max
safe_dist = v_lin_max
sens_range = 5 * v_lin_max

# Episode Settings
max_steps = 500