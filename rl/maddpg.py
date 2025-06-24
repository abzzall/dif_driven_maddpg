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

class MADDPGBase(ABC):
    def __init__(self, env: DiffDriveParallelEnv,  device=device, replay_buffer_size=replay_buffer_size):
        self.env=env
        self.replay_buffer=ReplayBuffer(obs_dim=env.obs_dim, state_dim=env.state_dim, action_dim=env.action_dim,
                                        device=device, num_agents=env.num_agents, replay_buffer_size=replay_buffer_size)
        self.obs_dim=env.obs_dim
        self.state_dim=env.state_dim

        self.device=device



    @staticmethod
    @torch.no_grad()
    def update_params_vectorized(network, target_network, tau):
        """
        Vectorized soft update: target ← τ * network + (1 - τ) * target
        Assumes both networks are on the same device.
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



    def main_loop(self, evaluate=False, n_games=n_games, train_each=train_each):
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
    def __init__(self, env: DiffDriveParallelEnv,  device=device):
        super().__init__(env, device=device)
        self.critic=SharedCritic(env.state_dim, env.action_dim, env.num_agents, device=self.device)
        self.critic_target=SharedCritic(env.state_dim, env.action_dim,env.num_agents,  device=self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor=SimpleActor(env.obs_dim, env.action_dim, device=device)
        self.actor_target=SimpleActor(env.obs_dim, env.action_dim, device=device)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def learn(self):
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
        actor_loss=-self.critic(state, joint_pred_actions).mean()                #Q(s, pi(o)) mean
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_params_vectorized(self.critic, self.critic_target, tau)
        self.update_params_vectorized(self.actor, self.actor_target, tau)

    def  load_actor(self):
        self.actor.load_checkpoint()

    def choose_actions(self, obs_list, use_noise=True):
        return self.actor.choose_action(obs_list, use_noise=use_noise, eval_mode=True)

    def save_checkpoint(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()
        self.actor_target.save_checkpoint()


