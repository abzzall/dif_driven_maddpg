import torch

# Number of agents in the environment
NUM_AGENTS = 9

# Number of landmarks (same as agents by default)
NUM_LANDMARKS = NUM_AGENTS

# Number of circular obstacles in the environment
NUM_OBSTACLES = 9

# Size of the 2D environment (world is a square of WORLD_SIZE x WORLD_SIZE)
WORLD_SIZE = 10.0

# Maximum linear velocity of an agent (in meters per second)
V_LIN_MAX = 1.0

# Maximum angular velocity of an agent (in radians per second)
V_ANG_MAX = torch.pi / 2  # 90 degrees

# Maximum adjustment (delta) for linear velocity per step
DV_LIN_MAX = 0.1 * V_LIN_MAX

# Maximum adjustment (delta) for angular velocity per step
DV_ANG_MAX = torch.pi / 12  # 15 degrees

# Agent's sensing range (e.g., how far it can detect obstacles)
SENSING_RANGE = 5 * V_LIN_MAX

# Radius of each agent (used for collision detection)
AGENT_RADIUS = V_LIN_MAX

# Minimum safe distance to avoid collisions (same as radius)
SAFE_DISTANCE = V_LIN_MAX

# CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")