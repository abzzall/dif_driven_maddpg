import time

from pettingzoo import ParallelEnv

from config import *
from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnv
from models.simpleactor import SimpleActor
from models.simplecritic import SharedCritic
from replay_buffer import ReplayBuffer
from abc import ABC, abstractmethod
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from config import *
from typing import Union, Optional
import matplotlib.pyplot as plt
import numpy as np




@torch.no_grad()
def plot_trajectory(
    env: DiffDriveParallelEnv,
    actor_weights_path: str,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    save_path: str = "trajectory.png"
) -> None:
    """
    Simulates and saves a trajectory plot of agents using a pre-trained actor policy.

    Args:
        env (DiffDriveParallelEnv): The environment instance.
        actor_weights_path (str): Path to the actor weights file (.pt).
        seed (int, optional): Seed for reproducibility.
        steps (int, optional): Number of steps to simulate. Defaults to env.max_steps.
        save_path (str): File path to save the generated PNG image.
    """
    steps = steps or env.max_steps
    state, obs = env.reset_tensor(seed)

    num_agents = env._num_agents
    obs_dim = env.obs_dim
    act_dim = env.action_dim
    device = env.device

    # Load trained actor
    actor = SimpleActor(obs_dim, act_dim).to(device)
    actor.load_checkpoint(actor_weights_path)
    actor.eval()

    # Collect trajectories
    traj = [env.agent_pos.clone().detach().cpu().numpy()]  # list of [N, 2] arrays
    for _ in range(steps):
        actions = actor(obs).clamp(-1, 1)  # shape: [N, 2]
        state, obs, _, dones = env.step_tensor(actions)
        traj.append(env.agent_pos.clone().detach().cpu().numpy())
        if dones.all():
            break
    traj = np.stack(traj, axis=1)  # shape: [N, T, 2]

    # === Plot ===
    fig, ax = plt.subplots(figsize=(7, 7))
    half = env.env_size.item() / 2
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect('equal')
    ax.set_title("Agent Trajectories")

    # Obstacles
    for i in range(env.num_obstacles):
        pos = env.obstacle_pos[i].cpu().numpy()
        rad = env.obstacle_radius[i].item()
        circle = plt.Circle(pos, rad, color='gray', alpha=0.5)
        ax.add_patch(circle)

    # Landmarks
    for lm in env.landmarks.cpu().numpy():
        ax.plot(lm[0], lm[1], 'rx', markersize=8, label='Landmark')

    # Agent trajectories
    colors = plt.cm.get_cmap('tab10', num_agents)
    for i in range(num_agents):
        path = traj[i]
        ax.plot(path[:, 0], path[:, 1], color=colors(i), linewidth=1.5)
        ax.plot(path[0, 0], path[0, 1], 'o', color='blue')   # start
        ax.plot(path[-1, 0], path[-1, 1], 'o', color='green')  # end

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


class MADDPGBase(ABC):
    def __init__(
            self,
            env: DiffDriveParallelEnv,
            device: Union[str, torch.device] = device,
            replay_buffer_size: int = replay_buffer_size
    ) -> None:
        """
        Initializes the base MADDPG class.

        Args:
            env (DiffDriveParallelEnv): The multi-agent environment.
            device (str | torch.device): Target device for computation (e.g., 'cuda' or 'cpu').
            replay_buffer_size (int): Maximum capacity of the replay buffer.

        Sets:
            self.env: The environment instance.
            self.replay_buffer: A ReplayBuffer instance.
            self.obs_dim: Observation dimension.
            self.state_dim: Global state dimension.
            self.device: Device used for all models and data.
        """
        self.env=env
        self.replay_buffer=ReplayBuffer(obs_dim=env.obs_dim, state_dim=env.state_dim, action_dim=env.action_dim,
                                        device=device, num_agents=env.num_agents, replay_buffer_size=replay_buffer_size)
        self.obs_dim=env.obs_dim
        self.state_dim=env.state_dim

        self.device=device

    @staticmethod
    @torch.no_grad()
    def update_params_vectorized(
            network: torch.nn.Module,
            target_network: torch.nn.Module,
            tau: float
    ) -> None:
        """
        Performs a soft update of target network parameters:
            θ_target ← τ * θ_online + (1 - τ) * θ_target

        Args:
            network (torch.nn.Module): Source network whose parameters are used in update.
            target_network (torch.nn.Module): Target network to be softly updated.
            tau (float): Soft update coefficient in [0, 1].

        Notes:
            Assumes both networks:
            - Have identical architectures,
            - Reside on the same device.
        """
        vec_net = parameters_to_vector(network.parameters())
        vec_target = parameters_to_vector(target_network.parameters())
        updated = tau * vec_net + (1.0 - tau) * vec_target
        vector_to_parameters(updated, target_network.parameters())

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def load_actor(self):
        pass

    @abstractmethod
    def choose_actions(self, obs_list, use_noise=True):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    def main_loop(
            self,
            evaluate: bool = False,
            n_games: int = n_games,
            train_each: int = train_each
    ) -> None:
        """
        Runs training or evaluation episodes.

        Args:
            evaluate (bool): If True, disables learning and enables rendering.
            n_games (int): Number of episodes to run.
            train_each (int): Frequency (in steps) to trigger learning.

        Flow:
            - Resets environment each episode.
            - Chooses and executes actions.
            - Stores transitions in replay buffer.
            - Learns after `train_each` steps (unless in evaluation mode).
            - Saves best-performing model (if not evaluating).
        """

        if evaluate:
            self.load_actor()
        total_steps = 0
        score_history=[]
        best_score=-np.inf
        for i in range(n_games):
            state, observations = self.env.reset_tensor()

            done = torch.full((self.env.num_agents,),False , dtype=torch.bool)

            while not any(done):
                if evaluate:
                    self.env.render()
                    time.sleep(0.1)  # to slow down the action for the video
                actions= self.choose_actions(observations)
                next_state, next_observations, rewards, done = self.env.step_tensor(actions)
                self.replay_buffer.add(state, observations, actions, rewards,
                                       next_state, next_observations, done)
                if total_steps>=train_each and not evaluate:
                    self.learn()
                    total_steps=0
                state=next_state
                observations=next_observations
                total_steps+=1
            score_history.append(self.env.score.mean())
            avg_score= np.mean(score_history[-100:])
            if not evaluate and avg_score>best_score:
                best_score=avg_score
                self.save_checkpoint()
                print('checkpoint saved')
            print('episode', i, 'score %.1f' % self.env.score.mean(), 'avg score %.1f' % avg_score)


class MADDPGSharedActorCritic(MADDPGBase):
    def __init__(
        self,
        env: DiffDriveParallelEnv,
        device: Union[str, torch.device] = device
    ) -> None:
        """
        Initializes MADDPG agent with shared actor and critic networks.

        Args:
            env (DiffDriveParallelEnv): Environment instance.
            device (str | torch.device): Device for model placement.

        Sets:
            self.critic (SharedCritic): Centralized critic for all agents.
            self.critic_target (SharedCritic): Target critic network.
            self.actor (SimpleActor): Shared actor across all agents.
            self.actor_target (SimpleActor): Target actor network.
        """
        super().__init__(env, device=device)
        self.critic=SharedCritic(env.state_dim, env.action_dim, env.num_agents, device=self.device)
        self.critic_target=SharedCritic(env.state_dim, env.action_dim,env.num_agents,  device=self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor=SimpleActor(env.obs_dim, env.action_dim, device=device)
        self.actor_target=SimpleActor(env.obs_dim, env.action_dim, device=device)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def learn(self):
        """
        Performs one training step for both actor and critic using MADDPG.

        Steps:
            - Samples a batch of experiences from replay buffer.
            - Computes target Q-values using target critic and target actor.
            - Computes critic loss and updates critic.
            - Computes actor loss (policy gradient) and updates actor.
            - Soft-updates target networks.

        Requires:
            - Replay buffer must be filled.
            - All networks and tensors must reside on the same device.

        Shapes:
            B: Batch size
            N: Number of agents
            obs:        [B, N, obs_dim]
            state:      [B, state_dim]
            actions:    [B, N, act_dim]
            rewards:    [B, N]
            dones:      [B, N]
        """
        if not self.replay_buffer.filled():
            print('memory not ready')
            return
        obs, next_obs, state, next_state, action, reward, done=self.replay_buffer.sample()
        B, N, act_dim = action.shape
        with torch.no_grad():
            next_action=self.actor_target(next_obs)                             #pi'(o')
            joint_next_action=next_action.view(B, N*act_dim)                    #pi'(o') joint
            next_q=self.critic_target(next_state, joint_next_action)                  #Q'(s', pi'(o'))
            y=reward+gamma*next_q*(1-done)                               #r+gamma*Q'(s', pi'(o'))*(1-done)
        joint_action=action.view(B, N*act_dim)                                   #a joint
        q_pred=self.critic(state, joint_action)                                      #Q(s, pi(o))
        critic_loss=F.mse_loss(q_pred, y)                                           #Q(s, pi(o))-r-g

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        pred_actions=self.actor(obs)                                                #pi(o)
        joint_pred_actions=pred_actions.view(B, N*act_dim)                        #pi(o) joint
        actor_loss=-self.critic(state, joint_pred_actions).mean()                #- Q(s, pi(o)) mean
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_params_vectorized(self.critic, self.critic_target, tau)
        self.update_params_vectorized(self.actor, self.actor_target, tau)

    def  load_actor(self):
        """
        Loads the saved weights of the actor network from the checkpoint path.

        Assumes:
            - self.actor is an instance of SimpleActor.
            - SimpleActor has a method `.load_checkpoint()` that handles device mapping.
        """
        self.actor.load_checkpoint()

    def choose_actions(
            self,
            obs_list: torch.Tensor,  # shape: [num_agents, obs_dim], on self.device
            use_noise: bool = True
    ) -> torch.Tensor:  # shape: [num_agents, act_dim], on self.device
        """
        Selects actions for all agents using the shared actor.

        Args:
            obs_list (torch.Tensor): A tensor of observations for each agent.
                - Shape: [num_agents, obs_dim]
                - Type: torch.FloatTensor
                - Must be on the same device as the actor network (e.g., CUDA or CPU)

            use_noise (bool): Whether to include exploration noise during action selection.
                - True during training for exploration
                - False during evaluation

        Returns:
            torch.Tensor: Actions for all agents.
                - Shape: [num_agents, act_dim]
                - Type: torch.FloatTensor
                - On the same device as self.actor
        """
        return self.actor.choose_action(obs_list, use_noise=use_noise, eval_mode=True)

    def save_checkpoint(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()
        self.actor_target.save_checkpoint()


