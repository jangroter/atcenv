import torch
import torch.nn.functional as F
import torch.optim as optim

from typing import Tuple, Optional, List
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import atcenv.src.functions as fn
from atcenv.src.environment_objects.flight import Flight

from atcenv.src.models.model import Model
from atcenv.src.models.actor import Actor
from atcenv.src.models.critic_q import Critic_Q
from atcenv.src.models.critic_v import Critic_V
from atcenv.src.models.replay_buffer import ReplayBuffer

class SAC(Model):

    def __init__(self,
                 action_dim: int, 
                 buffer: ReplayBuffer, 
                 actor: Actor, 
                 critic_q_1: Critic_Q,
                 critic_q_2: Critic_Q,
                 critic_v: Critic_V,
                 critic_v_target: Critic_V,
                 alpha_lr: float = 3e-3,
                 actor_lr: float = 3e-4,
                 critic_q_lr: float = 3e-3, 
                 critic_v_lr: float = 3e-3,
                 gamma: float = 0.995,
                 tau: float = 5e-3,
                 policy_update_freq: int = 1,
                 initial_random_steps: int = 0):

        self.transform_action = True
        self.test = False
        
        self.device = torch.device("cpu")

        self.action_dim = action_dim
        self.buffer = buffer

        self.target_alpha = -np.prod((self.action_dim)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.actor = actor.to(device=self.device)

        self.critic_q_1 = critic_q_1.to(device=self.device)
        self.critic_q_2 = critic_q_2.to(device=self.device)

        self.critic_v = critic_v.to(device=self.device)
        self.critic_v_target = critic_v_target.to(device=self.device)
        self.critic_v_target.load_state_dict(self.critic_v_target.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_q_1_optimizer = optim.Adam(self.critic_q_1.parameters(), lr=critic_q_lr)
        self.critic_q_2_optimizer = optim.Adam(self.critic_q_2.parameters(), lr=critic_q_lr)
        self.critic_v_optimizer = optim.Adam(self.critic_v.parameters(), lr=critic_v_lr)

        self.gamma = gamma
        self.tau = tau
        self.policy_update_freq = policy_update_freq

        self.initial_random_steps = initial_random_steps
        self.total_steps = 0

    def get_action(self, observation: dict) -> np.ndarray:
        observation = observation["observation"]

        if self.total_steps < self.initial_random_steps and not self.test:
            action = np.random.standard_normal((len(observation),self.action_dim)) * 0.33
        else:
            action = self.actor(torch.FloatTensor(np.array(observation)).to(self.device))[0].detach().cpu().numpy()
            action = np.array(action)
            action = np.clip(action, -1, 1)
        
        self.total_steps += 1
        return action
    
    def store_transition(self,observation,action,new_observation,reward,done) -> None:
        if not self.test:
            done = False
            transition = [observation, action, reward, new_observation, done]
            self.buffer.store(*transition)
        if (self.total_steps % self.policy_update_freq == 0 and 
            len(self.buffer) >  self.buffer.batch_size and 
            self.total_steps > self.initial_random_steps and 
            not self.test):
                self.update_model()

    def new_episode(self, test: bool) -> None:
        self.test = test

    def setup_model(self, experiment_folder: str) -> None:
        pass

    def update_model(self):
        device = self.device

        samples = self.buffer.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"]).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        b,n = reward.size()
        reward = reward.view(b,n,1)
        new_action, log_prob = self.actor(state)
        alpha_loss = ( -self.log_alpha.exp() * (log_prob + self.target_alpha).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        #mask = 1 - done
        q1_pred = self.critic_q_1(state, action)
        q2_pred = self.critic_q_2(state, action)
        vf_target = self.critic_v_target(next_state)
        q_target = reward + self.gamma * vf_target #* mask
        qf1_loss = F.mse_loss(q_target.detach(), q1_pred)
        qf2_loss = F.mse_loss(q_target.detach(), q2_pred)

        v_pred = self.critic_v(state)
        q_pred = torch.min(
            self.critic_q_1(state, new_action), self.critic_q_2(state, new_action)
        )
        v_target = q_pred - alpha * log_prob
        v_loss = F.mse_loss(v_pred, v_target.detach())

        if self.total_steps % self.policy_update_freq == 0:
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)
        
        self.critic_q_1_optimizer.zero_grad()
        qf1_loss.backward()
        self.critic_q_1_optimizer.step()
        self.critic_q_2_optimizer.zero_grad()
        qf2_loss.backward()
        self.critic_q_2_optimizer.step()

        qf_loss = qf1_loss + qf2_loss

        self.critic_v_optimizer.zero_grad()
        v_loss.backward()
        self.critic_v_optimizer.step()

    def _target_soft_update(self):
        for t_param, l_param in zip(
            self.critic_v_target.parameters(), self.critic_v.parameters()
        ):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)