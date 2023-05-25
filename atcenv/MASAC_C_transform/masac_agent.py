from math import gamma
import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from atcenv.MASAC_C_transform.buffer import ReplayBuffer
from atcenv.MASAC_C_transform.mactor_critic import Actor, CriticQ, CriticV
from torch.nn.utils.clip_grad import clip_grad_norm_


GAMMMA = 0.99
TAU =5e-3 #was 5e-3
INITIAL_RANDOM_STEPS = 100
POLICY_UPDATE_FREQUENCE = 2
NUM_AGENTS = 10

BUFFER_SIZE = 100000
BATCH_SIZE = 256

ACTION_DIM = 2
STATE_DIM = 7

NUM_HEADS = 3

NUMBER_INTRUDERS_STATE = 2

MEANS = [57000,57000,0,0,0,0,0,0]
STDS = [31500,31500,100000,100000,1,1,1,1]

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
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.qf1_lossarr = np.array([])
        self.qf2_lossarr = np.array([])

        self.actor = Actor(STATE_DIM, ACTION_DIM, num_heads=NUM_HEADS).to(self.device)

        self.vf = CriticV(14 * NUM_AGENTS).to(self.device)
        self.vf_target = CriticV(14 * NUM_AGENTS).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())

        self.qf1 = CriticQ(14 * NUM_AGENTS + ACTION_DIM * NUM_AGENTS).to(self.device)
        self.qf2 = CriticQ(14 * NUM_AGENTS + ACTION_DIM * NUM_AGENTS).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=3e-3)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=3e-3)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=3e-3)

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
    
    def setResult(self,episode_name, state, qstate, new_state, qnew_state, reward, action, done):       
        if not self.is_test:
            #for i in range(len(state)):         
            self.transition = [state, qstate, action, reward, new_state, qnew_state, done]
            self.memory.store(*self.transition)

        if (len(self.memory) >  BATCH_SIZE and self.total_step > INITIAL_RANDOM_STEPS):
            self.update_model()
    
    def update_model(self):
        device = self.device

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        qstate = torch.FloatTensor(samples['qobs']).to(device)
        q_state = torch.reshape(qstate,(BATCH_SIZE,14*NUM_AGENTS))
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        qnext_state = torch.FloatTensor(samples["qnext_obs"]).to(device)
        q_next_state = torch.reshape(qnext_state,(BATCH_SIZE,14*NUM_AGENTS))
        action = torch.FloatTensor(samples["acts"]).to(device)
        q_action = torch.reshape(action,(BATCH_SIZE,ACTION_DIM*NUM_AGENTS))
        reward = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        new_action, log_prob = self.actor(state)
        q_new_action = torch.reshape(new_action,(BATCH_SIZE,ACTION_DIM*NUM_AGENTS))
        q_log_prob = torch.mean(log_prob,1)
        alpha_loss = ( -self.log_alpha.exp() * (log_prob + self.target_alpha).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        #mask = 1 - done
        q1_pred = self.qf1(q_state, q_action)
        q2_pred = self.qf2(q_state, q_action)
        vf_target = self.vf_target(q_next_state)
        q_target = reward + GAMMMA * vf_target #* mask
        qf1_loss = F.mse_loss(q_target.detach(), q1_pred)
        qf2_loss = F.mse_loss(q_target.detach(), q2_pred)

        v_pred = self.vf(q_state)
        q_pred = torch.min(
            self.qf1(q_state, q_new_action), self.qf2(q_state, q_new_action)
        )
        v_target = q_pred - alpha * q_log_prob
        v_loss = F.mse_loss(v_pred, v_target.detach())


        if self.total_step % POLICY_UPDATE_FREQUENCE== 0:
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * q_log_prob - advantage).mean()

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

        self.qf1_lossarr = np.append(self.qf1_lossarr,qf1_loss.detach().cpu().numpy())
        self.qf2_lossarr = np.append(self.qf2_lossarr,qf2_loss.detach().cpu().numpy())

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
    
    def qnormalizeState(self, s_t, max_speed, min_speed):
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