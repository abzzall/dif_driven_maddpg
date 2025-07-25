import time

from pettingzoo import ParallelEnv


from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnv
from models.simpleactor import SimpleActor
from models.simplecritic import SharedCritic
from replay_buffer import ReplayBuffer
from abc import ABC, abstractmethod
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from config import *
from typing import Union, Optional, Tuple, Any, Callable
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


class MADDPGBase(ABC):
    def __init__(
            self,
            env: DiffDriveParallelEnv,
            device: Union[str, torch.device] = device,
            replay_buffer_size: int = replay_buffer_size,
            batch_size: int = batch_size
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
                                        device=device, num_agents=env.num_agents, replay_buffer_size=replay_buffer_size, batch_size=batch_size)
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
    def learn(self, buffer = None):
        pass

    @abstractmethod
    def load_actor(self):
        pass

    @abstractmethod
    def choose_actions(self, obs_list, use_noise=True):
        pass

    @abstractmethod
    def save_checkpoint(self, file_pref=None):
        pass
    @abstractmethod
    def load_checkpoint(self):
        pass

    def main_loop(self, n_games=n_games, train_each=train_each, evaluate=False,
                   checkpoint_path="training_state.pkl", patience=patience,
                   score_avg_window=score_avg_window, max_steps=max_steps) -> None:

        # === Load previous training state if exists ===
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                state_dict = pickle.load(f)
            start_episode = state_dict["episode"]
            self.score_history = state_dict["score_history"]
            best_score = state_dict["best_score"]
            self.replay_buffer.load("replay_buffer.pkl")
            episodes_without_improvement = state_dict["episodes_without_improvement"]
            total_steps = state_dict['total_steps']
            self.actor_losses = state_dict['actor_losses']
            self.critic_losses = state_dict['critic_losses']
            self.load_checkpoint()
            print(f"Resuming from episode {start_episode}")
        else:
            start_episode = 0
            self.score_history = []
            best_score = -float("inf")
            episodes_without_improvement = 0
            total_steps = 0
            self.actor_losses = []
            self.critic_losses = []

        for i in range(start_episode, n_games):
            state, obs = self.env.reset_tensor()
            done = torch.full((self.env.num_agents,), False, dtype=torch.bool)
            episode_trajectory = [self.env.agent_pos.cpu().clone().detach().numpy()]
            initial_headings = self.env.agent_dir.clone().detach().to(device=self.device)
            j = 0
            while j < max_steps and not done.all():
                if evaluate:
                    self.env.render()
                    time.sleep(0.1)

                actions = self.choose_actions(obs)
                next_state, next_obs, rewards, done = self.env.step_tensor(actions)
                self.replay_buffer.add(state, obs, actions, rewards, next_state, next_obs, done)

                state = next_state
                obs = next_obs

                total_steps += 1

                if not evaluate and total_steps % train_each == 0:
                    critic_loss, actor_loss = self.learn()
                    self.actor_losses.append(actor_loss)
                    self.critic_losses.append(critic_loss)
                    print(f"Trained at step {total_steps}, actor loss: {actor_loss}, critic loss: {critic_loss}")

                episode_trajectory.append(self.env.agent_pos.cpu().clone().detach().numpy())
                j += 1

            # === Episode statistics
            score = self.env.score.mean().cpu().item()
            self.score_history.append(score)
            avg_score = torch.tensor(self.score_history[-score_avg_window:], device='cpu').float().mean().item()
            print(f"Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}")

            if not evaluate and avg_score > best_score and i > min_episodes_before_early_stop:
                best_score = avg_score
                self.save_checkpoint(file_pref=f'best_{best_score}_{i}')
                print("Checkpoint saved (best model)")
                episodes_without_improvement = 0
            else:
                if i > min_episodes_before_early_stop:
                    episodes_without_improvement += 1






            # === Every 5 episodes: Save progress and plot
            if not evaluate and i % 5 == 0:
                print("Training progress saving.")
                state_dict = {
                    "episode": i + 1,
                    "score_history": self.score_history,
                    "best_score": best_score,
                    'episodes_without_improvement': episodes_without_improvement,
                    'total_steps': total_steps,
                    'actor_losses': self.actor_losses,
                    'critic_losses': self.critic_losses
                }
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(state_dict, f)

                self.replay_buffer.save("replay_buffer.pkl")
                self.save_checkpoint()
                print("Training progress saved.")

                # --- Plots ---
                self.plot_learning_curve()
                self.plot_actor_loss()
                self.plot_critic_loss()
                self.plot_episode_gone_trajectory(np.stack(episode_trajectory), episode=i)

            # === Offline replay training for this episode
            trajectories = torch.tensor(np.stack(episode_trajectory), dtype=torch.float32, device=self.device)
            trajectory=self.offline_replay(trajectories, initial_headings)
            if not evaluate and i % 5 == 0:
                # --- Replay trajectory of optimized offline version
                self.plot_episode_gone_trajectory(
                    np.stack(trajectory),  # Last state from offline replay
                    episode=0,

                )
            # Train 100 times from offline buffer

            for k in range(100):
                critic_loss, actor_loss = self.learn(buffer=self.offline_buffer)
                print(f"Trained offline at episode {i}: step {k}, actor loss: {actor_loss}, critic loss: {critic_loss}")

                self.actor_losses.append(actor_loss)
                self.critic_losses.append(critic_loss)
            if episodes_without_improvement >= patience:
                print(f"\n🛑 Early stopping triggered: No improvement in {patience} episodes.")
                self.plot_learning_curve()
                self.plot_actor_loss()
                self.plot_critic_loss()
                self.plot_episode_gone_trajectory(np.stack(episode_trajectory), episode=i)
                break

        print("Training complete.")

    def plot_learning_curve(self, episode=0):
        plt.figure(figsize=(8, 5))
        plt.plot(self.score_history)
        plt.title('Learning Curve (Score)')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.grid(True)
        plt.savefig(f'learning_curve_episode_{episode}.png', dpi=300)
        plt.close()

    def plot_actor_loss(self, episode=0):
        plt.figure(figsize=(8, 5))
        plt.plot(self.actor_losses)
        plt.title('Actor Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'actor_loss_episode_{episode}.png', dpi=300)
        plt.close()

    def plot_critic_loss(self, episode=0):
        plt.figure(figsize=(8, 5))
        plt.plot(self.critic_losses)
        plt.title('Critic Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'critic_loss_episode_{episode}.png', dpi=300)
        plt.close()

    def try_actor(self):
        print('trying current actor')
        done = torch.full((self.env.num_agents,), False, dtype=torch.bool)
        j = 0
        state, obs = self.env.reset_tensor()

        episode_trajectory = [self.env.agent_pos.cpu().clone().detach().numpy()]
        while j < max_steps and not done.all():
            actions = self.choose_actions(obs, use_noise=False)
            state, obs, rewards, done = self.env.step_tensor(actions)
            episode_trajectory.append(self.env.agent_pos.cpu().clone().detach().numpy())
            j = j + 1
            if j%100==0:
                print(f'{j} steps passed')
        score = self.env.score.mean().cpu().item()
        return episode_trajectory, score


    def plot_episode_gone_trajectory(self, trajectory, episode):
        # trajectory shape: (T, num_agents, 2)
        num_agents = trajectory.shape[1]
        env = self.env
        fig, ax = plt.subplots(figsize=(7, 7))
        half = env.env_size.item() / 2
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect('equal')
        ax.set_title(f"Episode {episode} Trajectories")

        # Obstacles
        for i in range(env.num_obstacles):
            pos = env.obstacle_pos[i].cpu().numpy()
            rad = env.obstacle_radius[i].item()
            circle = plt.Circle(pos, rad, color='gray', alpha=0.5)
            ax.add_patch(circle)

        # Agent trajectories
        colors = plt.cm.get_cmap('tab10', num_agents)
        for agent_idx in range(num_agents):
            path = trajectory[:, agent_idx]
            ax.plot(path[:, 0], path[:, 1], color=colors(agent_idx), linewidth=1.5)
            ax.plot(path[0, 0], path[0, 1], 'o', color='blue')  # start
            ax.plot(path[-1, 0], path[-1, 1], 'o', color='green')  # end
        # Landmarks
        for lm in env.landmarks.cpu().numpy():
            ax.plot(lm[0], lm[1], 'rx', markersize=8, label='Landmark')


        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'trajectory_episode_{episode}.png', dpi=300)
        plt.close()

    def plot_episode_new_trajectory(self, episode):
        trajectory, _ = self.try_actor()
        self.plot_episode_gone_trajectory(np.stack(trajectory), episode)
    def save_actor(self, file_name:str = 'simple_actor.pth'):
        pass

    def train_loop(
            self,
            n_games: int = n_games,
            train_each: int = train_each,
            evaluate: bool = False,
            checkpoint_path: str = "training_state.pkl",
            patience=patience,
            score_avg_window=score_avg_window,
            max_steps=max_steps,
            start_training_after=start_training_after,
            rescale_env_rewards:Callable[[DiffDriveParallelEnv, int], None]=None,
            min_episodes_before_early_stop:int=min_episodes_before_early_stop
    ) -> None:
        """
        Resumable training loop with checkpointing.

        Args:
            n_games (int): Total episodes to run.
            train_each (int): Frequency to trigger learning.
            evaluate (bool): If True, only evaluates.
            checkpoint_path (str): Path to save/load training state.
        """

        # === Load previous training state if exists ===
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                state_dict = pickle.load(f)
            start_episode = state_dict["episode"]
            self.score_history = state_dict["score_history"]
            best_score = state_dict["best_score"]
            self.replay_buffer.load("replay_buffer.pkl")
            episodes_without_improvement = state_dict["episodes_without_improvement"]
            total_steps = state_dict['total_steps']
            self.actor_losses=state_dict['actor_losses']
            self.critic_losses=state_dict['critic_losses']
            self.load_checkpoint()

            print(f"Resuming from episode {start_episode}")
        else:
            start_episode = 0
            self.score_history = []
            best_score = -float("inf")
            episodes_without_improvement = 0
            total_steps = 0
            self.actor_losses=[]
            self.critic_losses=[]


        for i in range(start_episode, n_games):
            if rescale_env_rewards!=None:
                rescale_env_rewards(self.env, i)
            state, obs = self.env.reset_tensor()
            done = torch.full((self.env.num_agents,), False, dtype=torch.bool)
            episode_trajectory = [self.env.agent_pos.cpu().clone().detach().numpy()]

            j=0
            while j <max_steps and not done.all():
                # print(f'step: {total_steps}')
                if evaluate:
                    self.env.render()
                    time.sleep(0.1)
                # print(f'chosing action by obs')
                actions = self.choose_actions(obs)
                # print(f'taking actions')
                next_state, next_obs, rewards, done = self.env.step_tensor(actions)
                # print(f'saving rb')
                self.replay_buffer.add(state, obs, actions, rewards, next_state, next_obs, done)

                state = next_state
                obs = next_obs

                total_steps += 1

                if not evaluate and total_steps % train_each == 0:
                    if total_steps<start_training_after:
                        print(f'replay buffer not ready: {total_steps}/{start_training_after}')
                    else:
                        critic_loss, actor_loss  = self.learn()
                        self.actor_losses.append(actor_loss)
                        self.critic_losses.append(critic_loss)
                        print(f"Trained at step {total_steps}, actor loss: {actor_loss}, critic loss : {critic_loss}")
                episode_trajectory.append(self.env.agent_pos.cpu().clone().detach().numpy())
                j=j+1
            score = self.env.score.mean().cpu().item()
            self.score_history.append(score)
            avg_score = torch.tensor(self.score_history[-score_avg_window:], device='cpu').float().mean().item()

            print(f"Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}")

            if not evaluate and avg_score > best_score and i>min_episodes_before_early_stop:
                best_score = avg_score
                self.save_checkpoint(file_pref=f'best_{best_score}_{i}')
                print("Checkpoint saved (best model)")
                episodes_without_improvement=0
            else:
                if i>min_episodes_before_early_stop:
                    episodes_without_improvement+=1
            if not (i>0):
                continue
            # Save training state every 5 episodes
            if i>0:

                print("Training progress saving.")
                state_dict = {
                    "episode": i + 1,
                    "score_history": self.score_history,
                    "best_score": best_score,
                'episodes_without_improvement': episodes_without_improvement,
                    'total_steps':total_steps,
                    'actor_losses':self.actor_losses,
                    'critic_losses':self.critic_losses


                }
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(state_dict, f)


                self.replay_buffer.save("replay_buffer.pkl")

                self.save_checkpoint()

                self.plot_learning_curve()
                self.plot_actor_loss()
                self.plot_critic_loss()
                # self.save_actor(f'simple_actor_{i}.pth')
                self.plot_episode_gone_trajectory(np.stack(episode_trajectory), i)
                print("Training progress saved.")
            if i%10==0:
                self.save_checkpoint(file_pref=f'episode_{i}_')
                self.replay_buffer.save(f"replay_buffer_{i}.pkl")
                with open(f'episode_{i}_{checkpoint_path}', "wb") as f:
                    pickle.dump(state_dict, f)
                self.plot_episode_new_trajectory(episode=i+1)
                self.plot_episode_new_trajectory(episode=i+2)
                self.plot_episode_new_trajectory(episode=i+3)

                print("plotted")

            if episodes_without_improvement >= patience:
                print(f"\n🛑 Early stopping triggered: No improvement in {patience} episodes.")
                print("Training progress saving.")
                state_dict = {
                    "episode": i + 1,
                    "score_history": self.score_history,
                    "best_score": best_score,
                    'episodes_without_improvement': episodes_without_improvement,
                    'total_steps': total_steps,
                    'actor_losses': self.actor_losses,
                    'critic_losses': self.critic_losses

                }
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(state_dict, f)

                self.replay_buffer.save("replay_buffer.pkl")
                self.save_checkpoint()
                print("Training progress saved.")

                self.plot_learning_curve()
                self.plot_actor_loss()
                self.plot_critic_loss()
                self.plot_episode_gone_trajectory(np.stack(episode_trajectory), episode=i)
                break
        print("Training complete.")

    def offline_replay(self, trajectories, init_headings):
        new_landmarks = []
        replay_actions_list=[]
        print('Optimizing roads:')
        for agent_id in range(self.env.num_agents):
            traj = trajectories[:, agent_id].to(self.device)
            heading = init_headings[agent_id].to(self.device)
            steps_counts, dists, prev_inx, last_dir, actions, furthest_idx = self.env.graph_search_cuda(traj, heading)
            print(f'road for agent_{agent_id} optimized')
            new_landmarks.append(traj[furthest_idx])
            ind=furthest_idx
            replay_actions=[]
            while(ind > 0):
                replay_actions.append(actions[ind])
                ind=prev_inx[ind]
            replay_actions.reverse()  # reverse to forward order
            replay_actions_list.append(replay_actions)
        new_landmarks = torch.stack(new_landmarks)
        print(f'reforming env: ')
            # Step 1: Compute centroid as new origin
        origin = new_landmarks.mean(dim=0)

            # Step 2: PCA for principal axis alignment
        centered = new_landmarks - origin
        cov = centered.T @ centered
        eigvals, eigvecs = torch.linalg.eigh(cov)
        x_axis = eigvecs[:, -1] / torch.norm(eigvecs[:, -1])
        y_axis = torch.stack([-x_axis[1], x_axis[0]])
        rot_matrix = torch.stack([x_axis, y_axis]).to(device=self.device)  # (2, 2)

            # Step 3: Apply translation and rotation to landmarks, agents, obstacles
        def transform(pos):
                return (pos - origin.to(device)) @ rot_matrix.T

            # Transform all components
        self.env.landmarks = transform(new_landmarks)
        self.env.agent_pos = transform(trajectories[0])
        self.env.obstacle_pos = transform(self.env.obstacle_pos)

                # Rotate headings accordingly (assumes 2D vectors for directions)
        self.env.agent_dir = (rot_matrix @ torch.stack([
                    torch.cos(init_headings),
                    torch.sin(init_headings)
                ])).T
        self.env.agent_dir = torch.atan2(self.env.agent_dir[:, 1], self.env.agent_dir[:, 0])

        self.env._init_static_state_part()
        self.env._reset_hungarian()

        observation =self.env.get_all_obs_tensor()
        state=self.env.state_tensor()
        done = torch.full((self.env.num_agents,), False, dtype=torch.bool, device=self.device)
        # --- Prepare actions tensor with padding ---
        max_len = max(len(actions) for actions in replay_actions_list)
        print(f'Starting replay: maxlenth of the replaying episode : {max_len}')
        padded_actions = torch.ones((max_len, self.env.num_agents, 2), device=device)

        for agent_id, actions_list in enumerate(replay_actions_list):
            for t, action in enumerate(actions_list):
                padded_actions[t, agent_id] = action.to(device)

        self.offline_buffer=ReplayBuffer(obs_dim=self.env.obs_dim, state_dim=self.env.state_dim, action_dim=self.env.action_dim,
                                        device=device, num_agents=self.env.num_agents, replay_buffer_size=max_len)
        j=0
        resulted_trajectory=[self.env.agent_pos.cpu().clone().detach().numpy()]
        while j < max_len and not done.all():
            actions = padded_actions[j]
            actions = torch.nan_to_num(actions, nan=1.0)
            # print(f'taking actions')
            next_state, next_obs, rewards, done = self.env.step_tensor(actions)
            # print(f'saving rb')
            self.offline_buffer.add(state, observation, actions, rewards, next_state, next_obs, done)
            resulted_trajectory.append(self.env.agent_pos.cpu().clone().detach().numpy())

            state = next_state
            observation = next_obs
            j=j+1
        print(f'replay successfull')
        return resulted_trajectory

class MADDPGSharedActorCritic(MADDPGBase):
    def __init__(
        self,
        env: DiffDriveParallelEnv,
            batch_size: int = batch_size,
            replay_buffer_size: int = replay_buffer_size,
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
        super().__init__(env, device=device, batch_size = batch_size, replay_buffer_size = replay_buffer_size,)
        self.critic=SharedCritic(env.state_dim, env.action_dim, env.num_agents, device=self.device, chckpnt_file='shared_critic.pth')
        self.critic_target=SharedCritic(env.state_dim, env.action_dim,env.num_agents,  device=self.device, chckpnt_file='shared_critic_target.pth')
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor=SimpleActor(env.obs_dim, env.action_dim, device=device, chckpnt_file='shared_actor.pth')
        self.actor_target=SimpleActor(env.obs_dim, env.action_dim, device=device, chckpnt_file='shared_actor_target.pth')
        self.actor_target.load_state_dict(self.actor.state_dict())

    def learn(self, buffer=None) -> tuple[None, None] | tuple[float, float]:
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
        #
        # if not self.replay_buffer.ready():
        #     print(
        #         f'memory not ready, current size = {self.replay_buffer.size}/{self.replay_buffer.start_training_after}')
        #     return None, None
        buffer = buffer or self.replay_buffer
        obs, next_obs, state, next_state, action, reward, done = self.replay_buffer.sample()
        B, N, act_dim = action.shape

        # ------------------ Critic Update ----------------------
        done_all = done.all(dim=1, keepdim=True).float()  # [B, 1], episode done if all agents done

        with torch.no_grad():
            next_action = self.actor_target(next_obs)  # pi'(o')  -> [B, N, act_dim]
            joint_next_action = next_action.view(B, N * act_dim)  # joint action -> [B, N * act_dim]
            next_q = self.critic_target(next_state, joint_next_action)  # Q'(s', pi'(o')) -> [B, 1]
            y = reward.sum(dim=1, keepdim=True) + gamma * next_q * (1 - done_all)  # joint reward + discount

        joint_action = action.view(B, N * act_dim)
        q_pred = self.critic(state, joint_action)
        critic_loss = F.mse_loss(q_pred, y)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # ------------------ Actor Update ----------------------
        pred_actions = self.actor(obs)  # pi(o) -> [B, N, act_dim]

        # Mask done agents' actions (agents marked done stop learning their actions)
        masked_actions = pred_actions * (~done).unsqueeze(-1).float()  # [B, N, act_dim]

        joint_pred_actions = masked_actions.view(B, N * act_dim)
        actor_loss = -self.critic(state, joint_pred_actions).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # ------------------ Target Networks -------------------
        self.update_params_vectorized(self.critic, self.critic_target, tau)
        self.update_params_vectorized(self.actor, self.actor_target, tau)

        return critic_loss.item(), actor_loss.item()

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

    def save_checkpoint(self, file_pref=None):
        self.actor.save_checkpoint(file_prefix=file_pref)
        self.critic.save_checkpoint(file_prefix=file_pref)
        self.critic_target.save_checkpoint(file_prefix=file_pref)
        self.actor_target.save_checkpoint(file_prefix=file_pref)
    def load_checkpoint(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()
        self.actor_target.load_checkpoint()
    def save_actor(self, file_name:str = 'simple_actor.pth'):
        self.actor.save_checkpoint(file_name)


