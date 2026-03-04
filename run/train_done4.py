import torch
import numpy as np
from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnvDone, \
    DiffDriveParallelEnvAssignFirstDone
from models.simplecritic import SharedCritic
from rl.maddpg import MADDPGSharedActorCriticIndependent, MADDPGSharedActorCriticIndependentQmean, IDDPG, IDDPGWithoutS

seed=9832
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

scale=[
            1.0,      # progressive
            1.0,      # distance
            0.0,      # base
            1000,   # reached goal
            200.0,    # agent collision
            200.0,    # obstacle collision
            1,      # v_lin
            1.,      # v_ang
            1.       # time
        ]

env=DiffDriveParallelEnvAssignFirstDone(
    v_ang_max=torch.pi/2,
    num_agents=6,
    num_obstacles=0


)
maddpg=IDDPGWithoutS(
    env,
    reward_scales=scale,
    batch_size=128,
    replay_buffer_size=5000,
)

maddpg.train_loop(
    start_training_after=100,
    train_each=50,
    patience=256,
    min_episodes_before_early_stop=2000,
    score_avg_window=256,
    max_steps=500
)
