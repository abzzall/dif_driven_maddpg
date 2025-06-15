from pettingzoo import ParallelEnv

from config import *
from models.critic import SimpleCritic
from replay_buffer import ReplayBuffer
from abc import ABC, abstractmethod

class MADDPGBase(ABC):
    def __init__(self, env: ParallelEnv, replay_buffer: ReplayBuffer, device=device):
       self.env=env
       self.replay_buffer=replay_buffer
       self.device=device


    @staticmethod
    def update_params(network, target_network, tau=tau, device=device):
        """
            Soft updates target_network parameters toward network parameters using interpolation factor tau.

            Args:
                network (torch.nn.Module): Source network (e.g., actor or critic).
                target_network (torch.nn.Module): Target network to be updated.
                tau (float): Soft update interpolation factor (0 < tau <= 1).
                device (torch.device): Device to perform the update on.
            """
        with torch.no_grad():
            for target_param, param in zip(target_network.parameters(), network.parameters()):
                target_param.data.copy_(
                    tau * param.data.to(device) + (1.0 - tau) * target_param.data.to(device)
                )
    @abstractmethod
    def learn(self):
        pass

