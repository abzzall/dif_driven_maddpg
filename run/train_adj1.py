import torch
import numpy as np

from config import v_ang_max
from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnvDone, \
    DiffDriveParallelEnvDoneAdj
from rl.maddpg import MADDPGSharedActorCriticIndependent, IDDPGWithoutS

seed=9832
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

scale=[
            1.0,      # progressive
            1.0,      # distance
            0.0,      # base
            10.0,   # reached goal
            10.0,    # agent collision
            10.0,    # obstacle collision
            1.0,      # v_lin
            1.0,      # v_ang
            1.0       # time
        ]
v_ang_max=torch.pi/2
dv_ang_max=v_ang_max/12
env=DiffDriveParallelEnvDoneAdj(
    v_ang_max=v_ang_max,
    v_lin_max=1,
    dv_ang_max=dv_ang_max,
    dv_lin_max=0.1
)

maddpg=IDDPGWithoutS(
    env,
    reward_scales=scale,
    batch_size=128,
    replay_buffer_size=5000,
)

maddpg.main_loop(
    start_training_after=100,
    train_each=50,
    patience=256,
    min_episodes_before_early_stop=10000,
    score_avg_window=256,
    max_steps=500
)
