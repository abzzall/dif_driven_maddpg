import torch
import numpy as np
from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnvDone
from models.simplecritic import SharedCritic
from rl.maddpg import MADDPGSharedActorCriticIndependent, MADDPGSharedActorCriticIndependentQmean
import os


def main(seed=9832):

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
                0,      # v_lin
                0.,      # v_ang
                1       # time
            ]

    env=DiffDriveParallelEnvDone(
        v_ang_max=torch.pi/12,
        num_agents=6,
        num_obstacles=10

    )

    maddpg=MADDPGSharedActorCriticIndependentQmean(
        env,
        reward_scales=scale,
        batch_size=128,
        replay_buffer_size=5000,
    )

    # maddpg.replay_buffer.load("replay_buffer.pkl")
    #
    # new_reward_scales = torch.tensor(scale).to(maddpg.device)
    # maddpg.train_critic_only(reward_scales=new_reward_scales, num_passes=10)
    #
    # maddpg.actor.load_checkpoint()
    # maddpg.actor_target.load_state_dict(maddpg.actor.state_dict())
    # maddpg.save_checkpoint()
    maddpg.train_loop(
        start_training_after=500,
        train_each=100,
        patience=256,
        min_episodes_before_early_stop=1000,
        score_avg_window=256,
        max_steps=500,
        n_games=1000

    )
    # os.chdir('..')
    # i=i+1

main()