import torch
import numpy as np

# CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_name = "my_parallel_env_v0"
num_agents = 3
obs_dim = 4
action_dim = 2
obs_low = -1.0
obs_high = 1.0
act_low = -1.0
act_high = 1.0
render_mode = "human"
