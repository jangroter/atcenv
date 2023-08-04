from math import gamma
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from atcenv.MASAC.buffer import ReplayBuffer
from atcenv.MASAC.mactor_critic import Actor, CriticQ, CriticV
from torch.nn.utils.clip_grad import clip_grad_norm_


GAMMMA = 0.99
TAU =5e-3
INITIAL_RANDOM_STEPS = 100
POLICY_UPDATE_FREQUENCE = 2
NUM_AGENTS = 10

BUFFER_SIZE = 1000000
BATCH_SIZE = 256

ACTION_DIM = 2
STATE_DIM = 14
NUMBER_INTRUDERS_STATE = 2

MEANS = [57000,57000,0,0,0,0,0,0]
STDS = [31500,31500,100000,100000,1,1,1,1]

class MaSacAgent:
    def __init__(self):                
        self.memory = ReplayBuffer(STATE_DIM,ACTION_DIM, BUFFER_SIZE, BATCH_SIZE)
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('DEVICE USED: ', torch.cuda.device(torch.cuda.current_device()), torch.cuda.get_device_name(0))
    
        except:
            # Cuda isn't available
            self.device = torch.device("cpu")
            print('DEVICE USED: CPU')
        
        self.target_alpha = -np.prod((ACTION_DIM,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.actor = Actor(STATE_DIM, ACTION_DIM).to(self.device)

        self.vf = CriticV(STATE_DIM).to(self.device)
        self.vf_target = CriticV(STATE_DIM).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())

        self.qf1 = CriticQ(STATE_DIM + ACTION_DIM).to(self.device)
        self.qf2 = CriticQ(STATE_DIM + ACTION_DIM).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=3e-4)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=3e-4)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=3e-4)

        self.transition = [[] for i in range(NUM_AGENTS)]

        self.total_step = 0

        self.is_test = False
        
    def do_step(self, state, max_speed, min_speed, test = False, batch = False):

        if not test and self.total_step < INITIAL_RANDOM_STEPS and not self.is_test:
            selected_action = np.random.uniform(-1, 1, (len(state), ACTION_DIM))
        else:
            selected_action = []
            for i in range(len(state)):
                action = self.actor(torch.FloatTensor(state[i]).to(self.device))[0].detach().cpu().numpy()
                selected_action.append(action)
            selected_action = np.array(selected_action)
            selected_action = np.clip(selected_action, -1, 1)

        self.total_step += 1
        return selected_action.tolist()
    
    def setResult(self,episode_name, state, new_state, reward, action, done):       
        if not self.is_test:
            for i in range(len(state)):               
                self.transition[i] = [state[i], action[i], reward, new_state[i], done]
                self.memory.store(*self.transition[i])

        if (len(self.memory) >  BATCH_SIZE and self.total_step > INITIAL_RANDOM_STEPS):
            self.update_model()
    
    def update_model(self):
        device = self.device

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"].reshape(-1, ACTION_DIM)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        new_action, log_prob = self.actor(state)

        alpha_loss = ( -self.log_alpha.exp() * (log_prob + self.target_alpha).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        mask = 1 - done
        q1_pred = self.qf1(state, action)
        q2_pred = self.qf2(state, action)
        vf_target = self.vf_target(next_state)
        q_target = reward + GAMMMA * vf_target * mask
        qf1_loss = F.mse_loss(q_target.detach(), q1_pred)
        qf2_loss = F.mse_loss(q_target.detach(), q2_pred)

        v_pred = self.vf(state)
        q_pred = torch.min(
            self.qf1(state, new_action), self.qf2(state, new_action)
        )
        v_target = q_pred - alpha * log_prob
        v_loss = F.mse_loss(v_pred, v_target.detach())

        if self.total_step % POLICY_UPDATE_FREQUENCE== 0:
            advantage = q_pred - v_pred.detach()
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

    def load_models(self):
        # The models were trained on a CUDA device
        # If you are running on a CPU-only machine, use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
        self.actor.load_state_dict(torch.load("results/mactor.pt", map_location=torch.device('cpu')))
        self.qf1.load_state_dict(torch.load("results/mqf1.pt", map_location=torch.device('cpu')))
        self.qf2.load_state_dict(torch.load("results/mqf2.pt", map_location=torch.device('cpu')))
        self.vf.load_state_dict(torch.load("results/mvf.pt", map_location=torch.device('cpu')))
    
    def _target_soft_update(self):
        for t_param, l_param in zip(
            self.vf_target.parameters(), self.vf.parameters()
        ):
            t_param.data.copy_(TAU * l_param.data + (1.0 - TAU) * t_param.data)

    def normalizeState(self, s_t, max_speed, min_speed):
         # distance to closest #NUMBER_INTRUDERS_STATE intruders
        for i in range(0, NUMBER_INTRUDERS_STATE):
            s_t[i] = (s_t[i]-MEANS[0])/(STDS[0]*2)

        # relative bearing to closest #NUMBER_INTRUDERS_STATE intruders
        for i in range(NUMBER_INTRUDERS_STATE, NUMBER_INTRUDERS_STATE*2):
            s_t[i] = (s_t[i]-MEANS[1])/(STDS[1]*2)

        for i in range(NUMBER_INTRUDERS_STATE*2, NUMBER_INTRUDERS_STATE*3):
            s_t[i] = (s_t[i]-MEANS[2])/(STDS[2]*2)
        
        for i in range(NUMBER_INTRUDERS_STATE*3, NUMBER_INTRUDERS_STATE*4):
            s_t[i] = (s_t[i]-MEANS[3])/(STDS[3]*2)

        for i in range(NUMBER_INTRUDERS_STATE*4, NUMBER_INTRUDERS_STATE*5):
            s_t[i] = (s_t[i])/(3.1415)

        # current bearing
        
        # current speed
        s_t[NUMBER_INTRUDERS_STATE*5] = ((s_t[NUMBER_INTRUDERS_STATE*5]-min_speed)/(max_speed-min_speed))*2 - 1
        # optimal speed
        s_t[NUMBER_INTRUDERS_STATE*5 + 1] = ((s_t[NUMBER_INTRUDERS_STATE*5 + 1]-min_speed)/(max_speed-min_speed))*2 - 1
        # # distance to target
        # s_t[NUMBER_INTRUDERS_STATE*2 + 2] = s_t[NUMBER_INTRUDERS_STATE*2 + 2]/MAX_DISTANCE
        # # bearing to target
        s_t[NUMBER_INTRUDERS_STATE*5+2] = s_t[NUMBER_INTRUDERS_STATE*5+2]
        s_t[NUMBER_INTRUDERS_STATE*5+3] = s_t[NUMBER_INTRUDERS_STATE*5+3]

        # s_t[0] = s_t[0]/MAX_BEARING

        return s_t