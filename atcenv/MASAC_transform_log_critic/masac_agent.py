from math import gamma
import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from atcenv.MASAC_transform_log_critic.buffer import ReplayBuffer
from atcenv.MASAC_transform_log_critic.mactor_critic import Actor, CriticQ, CriticV
from torch.nn.utils.clip_grad import clip_grad_norm_


GAMMMA = 1.00
TAU =5e-3 
INITIAL_RANDOM_STEPS = 100
POLICY_UPDATE_FREQUENCE = 1
NUM_AGENTS = 10

BUFFER_SIZE = 1000000
BATCH_SIZE = 256

ACTION_DIM = 2
STATE_DIM = 7

NUM_HEADS = 5

class MaSacAgent:
    def __init__(self):                
        self.memory = ReplayBuffer(STATE_DIM,ACTION_DIM, NUM_AGENTS, BUFFER_SIZE, BATCH_SIZE)

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('DEVICE USED: ', torch.cuda.device(torch.cuda.current_device()), torch.cuda.get_device_name(0))
    
        except:
            # Cuda isn't available
            self.device = torch.device("cpu")
            print('DEVICE USED: CPU')
        
        self.target_alpha = -np.prod((ACTION_DIM,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-3)

        self.qf1_lossarr = np.array([])
        self.qf2_lossarr = np.array([])

        self.actor = Actor(STATE_DIM, ACTION_DIM, num_heads=NUM_HEADS).to(self.device)

        self.vf = CriticV(STATE_DIM, num_heads=NUM_HEADS).to(self.device)
        self.vf_target = CriticV(STATE_DIM, num_heads=NUM_HEADS).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())

        self.qf1 = CriticQ(STATE_DIM + ACTION_DIM, num_heads=NUM_HEADS).to(self.device)
        self.qf2 = CriticQ(STATE_DIM + ACTION_DIM, num_heads=NUM_HEADS).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=1e-2)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=1e-2)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=1e-2)

        self.transition = []

        self.total_step = 0
        self.print = True

        self.is_test = False
        
        #print('DEVICE USED', torch.cuda.device(torch.cuda.current_device()), torch.cuda.get_device_name(0))
    
    def do_step(self, state, max_speed, min_speed, test = False, batch = False):

        if self.total_step < INITIAL_RANDOM_STEPS and not self.is_test:
            selected_action = np.random.uniform(-1, 1, (len(state), ACTION_DIM))
        else:
            state = [state]
            action = self.actor(torch.FloatTensor(state).to(self.device))[0].detach().cpu().numpy()
            selected_action = action[0]
            selected_action = np.array(selected_action)
            selected_action = np.clip(selected_action, -1, 1)

        self.total_step += 1
        return selected_action.tolist()
    
    def setResult(self,episode_name, state, new_state, reward, action, done):       
        if not self.is_test:
            #for i in range(len(state)):         
            self.transition = [state, action, reward, new_state, done]
            self.memory.store(*self.transition)

        if (len(self.memory) >  BATCH_SIZE and self.total_step > INITIAL_RANDOM_STEPS):
            self.update_model()
    
    def update_model(self):
        device = self.device

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"]).to(device)
        b,n = reward.size()
        reward = reward.view(b,n,1)
        done = torch.FloatTensor(samples["done"]).to(device)
        new_action, log_prob = self.actor(state)
        alpha_loss = ( -self.log_alpha.exp() * (log_prob + self.target_alpha).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        #mask = 1 - done
        q1_pred = self.qf1(state, action)
        q2_pred = self.qf2(state, action)
        vf_target = self.vf_target(next_state)
        #print(mask.shape)
        q_target = -1*torch.log(-1*(reward -0.000001 + GAMMMA*-1*torch.exp(vf_target*-1)))
        # print('qt',q_target[torch.isnan(q_target)])
        qf1_loss = F.mse_loss(q_target.detach(), q1_pred)
        qf2_loss = F.mse_loss(q_target.detach(), q2_pred)

        self.qf1_lossarr = np.append(self.qf1_lossarr,qf1_loss.detach().cpu().numpy())
        self.qf2_lossarr = np.append(self.qf2_lossarr,qf2_loss.detach().cpu().numpy())

        v_pred = self.vf(state)
        q_pred = torch.min(
            self.qf1(state, new_action), self.qf2(state, new_action)
        )
        # print('qp',q_pred[torch.isnan(q_pred)])
        # v_target = -1*torch.log(-1*(-1*torch.exp(q_pred*-1)- alpha * torch.exp(log_prob) -0.000001))
        v_target = -1*(-1*torch.exp(q_pred*-1)- alpha * log_prob)
        v_target[v_target < 0] = 0.000001
        v_target = -1*(-1*torch.exp(v_target))
        # print('vt',v_target[torch.isnan(v_target)])
        # print('v_target q',-1*torch.exp(q_pred[torch.isnan(v_target)]*-1))
        temp = alpha * log_prob
        # print('v_target alpha', temp[torch.isnan(v_target)])
        # print('vp',v_pred[torch.isnan(v_pred)])
        v_loss = F.mse_loss(v_pred, v_target.detach())
        # print('vl',v_loss)

        if self.total_step % POLICY_UPDATE_FREQUENCE== 0:
            advantage = (torch.exp(-1*q_pred)*-1) - (torch.exp(-1*v_pred.detach())*-1)
            # print('a',advantage[torch.isnan(advantage)])
            actor_loss = (alpha * log_prob - advantage).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)
        
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        qf_loss = qf1_loss + qf2_loss

        self.vf_optimizer.zero_grad()
        v_loss.backward()
        self.vf_optimizer.step()

        return actor_loss.data, qf_loss.data, v_loss.data, alpha_loss.data
    
    def save_models(self):
        torch.save(self.actor.state_dict(), "results/mactor.pt")
        torch.save(self.qf1.state_dict(), "results/mqf1.pt")
        torch.save(self.qf2.state_dict(), "results/mqf2.pt")
        torch.save(self.vf.state_dict(), "results/mvf.pt")       
    
    def _target_soft_update(self):
        for t_param, l_param in zip(
            self.vf_target.parameters(), self.vf.parameters()
        ):
            t_param.data.copy_(TAU * l_param.data + (1.0 - TAU) * t_param.data)

    def normalizeState(self, s_t, max_speed, min_speed):
        s_t[0] = s_t[0]/210000 # 210000 is the maximum observed x value for the default environment
        s_t[1] = s_t[1]/210000 

        s_t[2] = s_t[2]/max_speed
        s_t[3] = s_t[3]/max_speed
        s_t[4] = ((s_t[4]-(min_speed+max_speed)/2)/((max_speed-min_speed)/2))

        s_t[5] = s_t[5]/math.pi
        s_t[6] = s_t[6]/math.pi
        return s_t