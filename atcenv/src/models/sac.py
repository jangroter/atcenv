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

import time

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
                 policy_update_freq: int = 10,
                 initial_random_steps: int = 0):

        self.transform_action = True
        self.test = False
        
        # device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device("cpu")

        self.action_dim = action_dim
        self.buffer = buffer

        self.target_alpha = -np.prod((self.action_dim)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.qf1_lossarr = np.array([])
        self.qf2_lossarr = np.array([])

        self.actor = actor.to(device=self.device)

        self.critic_q_1 = critic_q_1.to(device=self.device)
        self.critic_q_2 = critic_q_2.to(device=self.device)

        self.critic_v = critic_v.to(device=self.device)
        self.critic_v_target = critic_v_target.to(device=self.device)
        # self.critic_v_target.load_state_dict(self.critic_v_target.state_dict())

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
            action = self.actor(torch.FloatTensor(np.array([observation])).to(self.device))[0].detach().cpu().numpy()
            action = np.array(action[0])
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
        super().setup_model(experiment_folder)

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

        self.qf1_lossarr = np.append(self.qf1_lossarr,qf1_loss.detach().cpu().numpy())
        self.qf2_lossarr = np.append(self.qf2_lossarr,qf2_loss.detach().cpu().numpy())
   
        v_pred = self.critic_v(state.detach())
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

class SAC_V2(Model):

    def __init__(self,
                 action_dim: int, 
                 buffer: ReplayBuffer, 
                 actor: Actor, 
                 critic_q_1: Critic_Q,
                 critic_q_2: Critic_Q,
                 alpha_lr: float = 3e-3,
                 actor_lr: float = 3e-4,
                 critic_q_lr: float = 3e-3, 
                 gamma: float = 0.995,
                 tau: float = 5e-3,
                 policy_update_freq: int = 10,
                 initial_random_steps: int = 0):

        self.transform_action = True
        self.test = False
        
        # device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device("cpu")

        self.action_dim = action_dim
        self.buffer = buffer

        self.target_alpha = -np.prod((self.action_dim)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.qf1_lossarr = np.array([])
        self.qf2_lossarr = np.array([])

        self.actor = actor.to(device=self.device)

        self.critic_q_1 = critic_q_1.to(device=self.device)
        self.critic_q_2 = critic_q_2.to(device=self.device)

        self.critic_q_target_1 = critic_q_1.to(device=self.device)
        self.critic_q_target_2 = critic_q_2.to(device=self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_q_1_optimizer = optim.Adam(self.critic_q_1.parameters(), lr=critic_q_lr)
        self.critic_q_2_optimizer = optim.Adam(self.critic_q_2.parameters(), lr=critic_q_lr)

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
            action = self.actor(torch.FloatTensor(np.array([observation])).to(self.device))[0].detach().cpu().numpy()
            action = np.array(action[0])
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
        super().setup_model(experiment_folder)

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

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            qf1_next_target = self.critic_q_target_1(next_state, next_action)
            qf2_next_target = self.critic_q_target_2(next_state, next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_prob
            q_target = reward + self.gamma * (min_qf_next_target)

        #mask = 1 - done

        q1_pred = self.critic_q_1(state, action)
        q2_pred = self.critic_q_2(state, action)

        qf1_loss = F.mse_loss(q_target.detach(), q1_pred)
        qf2_loss = F.mse_loss(q_target.detach(), q2_pred)

        self.qf1_lossarr = np.append(self.qf1_lossarr,qf1_loss.detach().cpu().numpy())
        self.qf2_lossarr = np.append(self.qf2_lossarr,qf2_loss.detach().cpu().numpy())

        q_pred = torch.min(
            self.critic_q_1(state, new_action), self.critic_q_2(state, new_action)
        )

        
        if self.total_steps % self.policy_update_freq == 0:
            actor_loss = (alpha * log_prob - q_pred).mean()

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

    def _target_soft_update(self):
        for t_param, l_param in zip(
            self.critic_q_target_1.parameters(), self.critic_q_1.parameters()
        ):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)
        for t_param, l_param in zip(
            self.critic_q_target_2.parameters(), self.critic_q_2.parameters()
        ):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)

class SAC_V3(Model):
    def __init__(self,
                 action_dim: int, 
                 buffer: ReplayBuffer, 
                 actor: Actor, 
                 critic_q: Critic_Q,
                 critic_q_target: Critic_Q,
                 alpha_lr: float = 3e-3,
                 actor_lr: float = 3e-4,
                 critic_q_lr: float = 3e-3, 
                 gamma: float = 0.995,
                 tau: float = 5e-3,
                 policy_update_freq: int = 10,
                 initial_random_steps: int = 0):

        self.gamma = gamma
        self.tau = tau
        self.transform_action = True
        self.test = False
        self.initial_random_steps = initial_random_steps
        self.total_steps = 0
        
        # device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device("cpu")

        self.action_dim = action_dim
        self.buffer = buffer

        self.target_alpha = -np.prod((self.action_dim)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.qf1_lossarr = np.array([])
        self.qf2_lossarr = np.array([])


        self.policy_update_freq = policy_update_freq

        self.actor = actor.to(device=self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_q = critic_q.to(device=self.device)
        self.critic_optim = optim.Adam(self.critic_q.parameters(), lr=critic_q_lr)
        self.critic_q_target = critic_q_target.to(device=self.device)

    
        self.hard_update(self.critic_q_target, self.critic_q)

    def get_action(self, observation: dict) -> np.ndarray:
        observation = observation["observation"]

        if self.total_steps < self.initial_random_steps and not self.test:
            action = np.random.standard_normal((len(observation),self.action_dim)) * 0.33
        else:
            action = self.actor(torch.FloatTensor(np.array([observation])).to(self.device))[0].detach().cpu().numpy()
            action = np.array(action[0])
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
        super().setup_model(experiment_folder)

    def update_model(self):
        # Sample a batch from memory
        device = self.device

        samples = self.buffer.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"]).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        b,n = reward.size()
        reward = reward.view(b,n,1)

        alpha = self.log_alpha.exp()

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.actor(next_state)
            qf_target = self.critic_q_target(next_state, next_state_action)
            qf1_next_target = qf_target[:,:,0]
            qf2_next_target = qf_target[:,:,1]
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi.flatten(start_dim=-2,end_dim=-1)
            next_q_value = reward.flatten(start_dim=-2,end_dim=-1) + self.gamma * (min_qf_next_target)

        qf= self.critic_q(state, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1 = qf[:,:,0]
        qf2 = qf[:,:,1]
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.qf1_lossarr = np.append(self.qf1_lossarr,qf1_loss.detach().cpu().numpy())
        self.qf2_lossarr = np.append(self.qf2_lossarr,qf2_loss.detach().cpu().numpy())

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi = self.actor(state)

        qf_pi = self.critic_q(state, pi)
        qf1_pi = qf_pi[:,:,0]
        qf2_pi = qf_pi[:,:,1]
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        if self.total_steps % self.policy_update_freq == 0:
            policy_loss = ((alpha * log_pi.flatten(start_dim=-2,end_dim=-1)) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.critic_q_target, self.critic_q, self.tau)
        else:
            policy_loss = torch.zeros(1)
        

        alpha_loss = -(self.log_alpha * (log_pi + self.target_alpha).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone() # For TensorboardX logs

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)