import torch
import numpy as np
from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnvDone, \
    DiffDriveParallelEnvDoneAdj
from rl.maddpg import MADDPGSharedActorCriticIndependent


seed=9832
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

scale=[
            1.0,      # progressive
            0.3,      # distance
            0.0,      # base
            10,   # reached goal
            10.0,    # agent collision
            10.0,    # obstacle collision
            0.2,      # v_lin
            0.7,      # v_ang
            0.5       # time
        ]
env=DiffDriveParallelEnvDone(
    v_ang_max=torch.pi/2,
    v_lin_max=1,
)

maddpg=MADDPGSharedActorCriticIndependent(
    env,
    reward_scales=scale,
    batch_size=128,
    replay_buffer_size=5000,
)

maddpg.main_loop(
    start_training_after=100,
    train_each=50,
    patience=256,
    min_episodes_before_early_stop=2000,
    score_avg_window=256,
    max_steps=500
)
