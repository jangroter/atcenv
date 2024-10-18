import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from typing import Tuple

from abc import ABC, abstractmethod

import atcenv.src.models.transformer as transformer

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

class Actor(ABC, nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.test = False
    
    @abstractmethod
    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def set_test(self, test: bool) -> None:
        self.test = test

class FeedForwardActor(Actor):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.layers = nn.ModuleList()

        for layer in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        log_std_layer = nn.Linear(hidden_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(hidden_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = observation
        for layer in self.layers:
            x = layer(x)

        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob
        
        return action, log_prob

class RelTransformerActor(Actor):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_heads: int = 3,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()

        self.num_heads = num_heads
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.tokeys    = nn.Linear(4, 4*num_heads, bias=False)
        self.toqueries = nn.Linear(in_dim, 4*num_heads, bias=False)
        self.tovalues  = nn.Linear(4, 4*num_heads, bias=False)

        self.layers = nn.ModuleList()
        in_dim = 4 * num_heads + in_dim
        for layer in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim

        log_std_layer = nn.Linear(hidden_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(hidden_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:  

        rel_state_init = state[:,:,0:4]

        h = self.num_heads
        b, t, k = rel_state_init.size()
        
        # Subtracts this vector from itself to generate a relative state matrix 
        rel_state = rel_state_init.view(b,1,t,k) - rel_state_init.view(b,1,t,k).transpose(1,2)

        # Create a boolean mask to exclude the diagonal elements
        mask = ~torch.eye(t, dtype=bool).unsqueeze(0).repeat(b,1,1)
        rel_state = rel_state[mask].view(b, t, t-1, k)
    
        b, t,_, k = rel_state.size()

        # transformer operations
        queries =  self.toqueries(state).view(b, t, h, k)
        keys =  self.tokeys(rel_state.view(b,t*(t-1),k)).view(b,t*(t-1),h,k)
        values =  self.tovalues(rel_state.view(b,t*(t-1),k)).view(b,t*(t-1),h,k)

        # Fold heads into the batch dimension
        queries = queries.transpose(1, 2).reshape(b * h, t, k)
        keys = keys.transpose(1, 2).reshape(b * h, t*(t-1), k)
        values = values.transpose(1, 2).reshape(b * h, t*(t-1), k)

        queries = queries.view(b*h,t,1,k).transpose(2,3)
        keys = keys.view(b*h,t,(t-1),k)
        values = values.view(b*h,t,(t-1),k)

        w_prime = torch.matmul(keys,queries)
        w_prime = w_prime / (k ** (1 / 2))
        w = F.softmax(w_prime, dim=2)
        
        x = torch.matmul(w.transpose(2,3),values).view(b*h,t,k)

        # Retrieve heads from batch dimension
        x = x.view(b,h,t,k)

        # Swap h, t back, unify heads
        x = x.transpose(1, 2).reshape(b, t, h * k)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)

        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob

class RelDistTransformerActor(Actor):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_heads: int = 3,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()

        self.num_heads = num_heads
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.tokeys    = nn.Linear(5, 5*num_heads, bias=False)
        self.toqueries = nn.Linear(in_dim, 5*num_heads, bias=False)
        self.tovalues  = nn.Linear(5, 5*num_heads, bias=False)

        self.layers = nn.ModuleList()
        in_dim = 5 * num_heads + in_dim
        for layer in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim

        log_std_layer = nn.Linear(hidden_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(hidden_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:  

        rel_state_init = state[:,:,0:4]

        h = self.num_heads
        b, t, k = rel_state_init.size()
        
        # Subtracts this vector from itself to generate a relative state matrix 
        rel_state = rel_state_init.view(b,1,t,k) - rel_state_init.view(b,1,t,k).transpose(1,2)

        # Create a boolean mask to exclude the diagonal elements
        mask = ~torch.eye(t, dtype=bool).unsqueeze(0).repeat(b,1,1)
        rel_state = rel_state[mask].view(b, t, t-1, k)

        # Calculate absolute distance matrix of size (batch,n_agents,n_agents-1,1)
        r = torch.sqrt(rel_state[:,:,:,0].clone()**2 + rel_state[:,:,:,1].clone()**2).view(b,t,t-1,1)
        
        # Apply transformation to distance to have a distance closer to zero have a higher value, assymptotically going to zero.
        r_trans = (1/(1+torch.exp(-1+5.*(r-0.2))))

        # divide x and y by the distance to get the values as a function of the unit circle
        rel_state[:,:,:,0:2] = rel_state[:,:,:,0:2].clone()/r
        
        # add transformed distance vector to the state vector
        rel_state = torch.cat((rel_state,r_trans),dim=-1)
    
        b, t,_, k = rel_state.size()

        # transformer operations
        queries =  self.toqueries(state).view(b, t, h, k)
        keys =  self.tokeys(rel_state.view(b,t*(t-1),k)).view(b,t*(t-1),h,k)
        values =  self.tovalues(rel_state.view(b,t*(t-1),k)).view(b,t*(t-1),h,k)

        # Fold heads into the batch dimension
        queries = queries.transpose(1, 2).reshape(b * h, t, k)
        keys = keys.transpose(1, 2).reshape(b * h, t*(t-1), k)
        values = values.transpose(1, 2).reshape(b * h, t*(t-1), k)

        queries = queries.view(b*h,t,1,k).transpose(2,3)
        keys = keys.view(b*h,t,(t-1),k)
        values = values.view(b*h,t,(t-1),k)

        w_prime = torch.matmul(keys,queries)
        w_prime = w_prime / (k ** (1 / 2))
        w = F.softmax(w_prime, dim=2)
        
        x = torch.matmul(w.transpose(2,3),values).view(b*h,t,k)

        # Retrieve heads from batch dimension
        x = x.view(b,h,t,k)

        # Swap h, t back, unify heads
        x = x.transpose(1, 2).reshape(b, t, h * k)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)

        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob

class AbsTransformerActor(Actor):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_heads: int = 3,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()

        self.num_heads = num_heads

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.tokeys    = nn.Linear(4, 4*num_heads, bias=False)
        self.toqueries = nn.Linear(4, 4*num_heads, bias=False)
        self.tovalues  = nn.Linear(4, 4*num_heads, bias=False)

        self.layers = nn.ModuleList()
        in_dim = 4 * num_heads + in_dim
        for layer in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim

        log_std_layer = nn.Linear(hidden_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(hidden_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:  
        
        t_state = state[:,:,0:4]

        b, t, k = t_state.size()
        h = self.num_heads

        # Project state to queries, keys and values
        queries = self.toqueries(t_state).view(b, t, h, k)
        keys    = self.tokeys(t_state).view(b, t, h, k)
        values  = self.tovalues(t_state).view(b, t, h, k)

        # Fold heads into the batch dimension
        keys = keys.transpose(1, 2).reshape(b * h, t, k)
        queries = queries.transpose(1, 2).reshape(b * h, t, k)
        values = values.transpose(1, 2).reshape(b * h, t, k)

        # Compute attention weights
        w_prime = torch.bmm(queries, keys.transpose(1, 2))
        w_prime = w_prime / (k ** (1 / 2))
        w = F.softmax(w_prime, dim=2)

        # Apply the self-attention to the values
        x = torch.bmm(w, values).view(b, h, t, k)

        # Swap h, t back, unify heads
        x = x.transpose(1, 2).reshape(b, t, h * k)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)

        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob

class OwnRefTransformerActor(Actor):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_heads: int = 3,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()

        self.num_heads = num_heads
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.tokeys    = nn.Linear(5, 5*num_heads, bias=False)
        self.toqueries = nn.Linear(in_dim, 5*num_heads, bias=False)
        self.tovalues  = nn.Linear(5, 5*num_heads, bias=False)

        self.layers = nn.ModuleList()
        in_dim = 5 * num_heads + in_dim
        for layer in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim

        log_std_layer = nn.Linear(hidden_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(hidden_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:  

        """ State should be [batch,n_agents,[x,y,vx,vy,v,cos(drift),sin(drift),track]]"""

        rel_state_init = state[:,:,0:4]

        h = self.num_heads
        b, t, k = rel_state_init.size()
        
        # Subtracts this vector from itself to generate a relative state matrix 
        rel_state = rel_state_init.view(b,1,t,k) - rel_state_init.view(b,1,t,k).transpose(1,2)
        
        # Create x,y tensors for each aircraft by doubling the k dimension
        xy_state = rel_state.view(b,t,t,2,2).transpose(3,4)
        angle = state[:,:,-1]
        angle = -angle + 0.5 * torch.pi
        angle = angle.view(b,t,1)
        x_new = xy_state[:,:,:,0,:] * torch.cos(angle.view(b,1,t,1)) - xy_state[:,:,:,1,:]*torch.sin(angle.view(b,1,t,1))
        y_new = xy_state[:,:,:,0,:] * torch.sin(angle.view(b,1,t,1)) + xy_state[:,:,:,1,:]*torch.cos(angle.view(b,1,t,1))
        rel_state = torch.cat((x_new,y_new),dim=3).view(b,t,t,2,2).transpose(3,4).reshape(b,t,t,4)

        # Create a boolean mask to exclude the diagonal elements
        mask = ~torch.eye(t, dtype=bool).unsqueeze(0).repeat(b,1,1)
        rel_state = rel_state[mask].view(b, t, t-1, k)

        # Calculate absolute distance matrix of size (batch,n_agents,n_agents-1,1)
        r = torch.sqrt(rel_state[:,:,:,0].clone()**2 + rel_state[:,:,:,1].clone()**2).view(b,t,t-1,1)
        
        # Apply transformation to distance to have a distance closer to zero have a higher value, assymptotically going to zero.
        r_trans = (1/(1+torch.exp(-1+5.*(r-0.2))))

        # divide x and y by the distance to get the values as a function of the unit circle
        rel_state[:,:,:,0:2] = rel_state[:,:,:,0:2].clone()/r
        
        # add transformed distance vector to the state vector
        rel_state = torch.cat((rel_state,r_trans),dim=-1)
    
        b, t,_, k = rel_state.size()

        # transformer operations
        queries =  self.toqueries(state).view(b, t, h, k)
        keys =  self.tokeys(rel_state.view(b,t*(t-1),k)).view(b,t*(t-1),h,k)
        values =  self.tovalues(rel_state.view(b,t*(t-1),k)).view(b,t*(t-1),h,k)

        # Fold heads into the batch dimension
        queries = queries.transpose(1, 2).reshape(b * h, t, k)
        keys = keys.transpose(1, 2).reshape(b * h, t*(t-1), k)
        values = values.transpose(1, 2).reshape(b * h, t*(t-1), k)

        queries = queries.view(b*h,t,1,k).transpose(2,3)
        keys = keys.view(b*h,t,(t-1),k)
        values = values.view(b*h,t,(t-1),k)

        w_prime = torch.matmul(keys,queries)
        w_prime = w_prime / (k ** (1 / 2))
        w = F.softmax(w_prime, dim=2)
        
        x = torch.matmul(w.transpose(2,3),values).view(b*h,t,k)

        # Retrieve heads from batch dimension
        x = x.view(b,h,t,k)

        # Swap h, t back, unify heads
        x = x.transpose(1, 2).reshape(b, t, h * k)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)

        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob

class OwnRefTransformerActor_final(Actor):

    def __init__(self,
                 q_dim: int,
                 kv_dim: int,
                 out_dim: int,
                 num_heads: int = 3,
                 num_blocks: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()

        self.num_heads = num_heads
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.tokeys    = nn.Linear(kv_dim, kv_dim*num_heads, bias=False)
        self.toqueries = nn.Linear(q_dim, kv_dim*num_heads, bias=False)
        self.tovalues  = nn.Linear(kv_dim, kv_dim*num_heads, bias=False)

        self.layers = nn.ModuleList()
        in_dim = kv_dim * num_heads + q_dim
        for layer in range(2):
            self.layers.append(nn.Linear(in_dim, 256))
            self.layers.append(nn.ReLU())
            in_dim = 256

        log_std_layer = nn.Linear(in_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(in_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:  

        """ State should be [batch,n_agents,[x,y,vx,vy,v,cos(drift),sin(drift),track]]"""

        rel_state_init = state[:,:,0:4]

        h = self.num_heads
        b, t, k = rel_state_init.size()
        
        # Subtracts this vector from itself to generate a relative state matrix 
        rel_state = rel_state_init.view(b,1,t,k) - rel_state_init.view(b,1,t,k).transpose(1,2)
        
        # Create x,y tensors for each aircraft by doubling the k dimension
        xy_state = rel_state.view(b,t,t,2,2).transpose(3,4)
        angle = state[:,:,-1]
        angle = -angle + 0.5 * torch.pi
        angle = angle.view(b,t,1)
        x_new = xy_state[:,:,:,0,:] * torch.cos(angle.view(b,1,t,1)) - xy_state[:,:,:,1,:]*torch.sin(angle.view(b,1,t,1))
        y_new = xy_state[:,:,:,0,:] * torch.sin(angle.view(b,1,t,1)) + xy_state[:,:,:,1,:]*torch.cos(angle.view(b,1,t,1))
        rel_state = torch.cat((x_new,y_new),dim=3).view(b,t,t,2,2).transpose(3,4).reshape(b,t,t,4)

        # Create a boolean mask to exclude the diagonal elements
        mask = ~torch.eye(t, dtype=bool).unsqueeze(0).repeat(b,1,1)
        rel_state = rel_state[mask].view(b, t, t-1, k)

        # Calculate absolute distance matrix of size (batch,n_agents,n_agents-1,1)
        r = torch.sqrt(rel_state[:,:,:,0].clone()**2 + rel_state[:,:,:,1].clone()**2).view(b,t,t-1,1)
        
        # Apply transformation to distance to have a distance closer to zero have a higher value, assymptotically going to zero.
        r_trans = (1/(1+torch.exp(-1+5.*(r-0.2))))

        # divide x and y by the distance to get the values as a function of the unit circle
        rel_state[:,:,:,0:2] = rel_state[:,:,:,0:2].clone()/r
        
        # add transformed distance vector to the state vector
        rel_state = torch.cat((rel_state,r_trans),dim=-1)
    
        b, t,_, k = rel_state.size()

        # transformer operations
        queries =  self.toqueries(state).view(b, t, h, k)
        keys =  self.tokeys(rel_state.view(b,t*(t-1),k)).view(b,t*(t-1),h,k)
        values =  self.tovalues(rel_state.view(b,t*(t-1),k)).view(b,t*(t-1),h,k)

        # Fold heads into the batch dimension
        queries = queries.transpose(1, 2).reshape(b * h, t, k)
        keys = keys.transpose(1, 2).reshape(b * h, t*(t-1), k)
        values = values.transpose(1, 2).reshape(b * h, t*(t-1), k)

        queries = queries.view(b*h,t,1,k).transpose(2,3)
        keys = keys.view(b*h,t,(t-1),k)
        values = values.view(b*h,t,(t-1),k)

        w_prime = torch.matmul(keys,queries)
        w_prime = w_prime / (k ** (1 / 2))
        w = F.softmax(w_prime, dim=2)
        
        x = torch.matmul(w.transpose(2,3),values).view(b*h,t,k)

        # Retrieve heads from batch dimension
        x = x.view(b,h,t,k)

        # Swap h, t back, unify heads
        x = x.transpose(1, 2).reshape(b, t, h * k)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)

        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob

class OwnRefTransformerSkipActor(Actor):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_heads: int = 3,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()

        self.num_heads = num_heads
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.tokeys    = nn.Linear(5, 5*num_heads, bias=False)
        self.toqueries = nn.Linear(in_dim, 5*num_heads, bias=False)
        self.tovalues  = nn.Linear(5, 5*num_heads, bias=False)

        self.layers = nn.ModuleList()
        input = 5 * num_heads + in_dim
        for layer in range(num_layers):
            self.layers.append(nn.Linear(input, hidden_dim))
            input = hidden_dim+in_dim+5*num_heads

        log_std_layer = nn.Linear(input, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(input, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:  

        """ State should be [batch,n_agents,[x,y,vx,vy,v,cos(drift),sin(drift),track]]"""

        rel_state_init = state[:,:,0:4]

        h = self.num_heads
        b, t, k = rel_state_init.size()
        
        # Subtracts this vector from itself to generate a relative state matrix 
        rel_state = rel_state_init.view(b,1,t,k) - rel_state_init.view(b,1,t,k).transpose(1,2)
        
        # Create x,y tensors for each aircraft by doubling the k dimension
        xy_state = rel_state.view(b,t,t,2,2).transpose(3,4)
        angle = state[:,:,-1]
        angle = -angle + 0.5 * torch.pi
        angle = angle.view(b,t,1)
        x_new = xy_state[:,:,:,0,:] * torch.cos(angle.view(b,1,t,1)) - xy_state[:,:,:,1,:]*torch.sin(angle.view(b,1,t,1))
        y_new = xy_state[:,:,:,0,:] * torch.sin(angle.view(b,1,t,1)) + xy_state[:,:,:,1,:]*torch.cos(angle.view(b,1,t,1))
        rel_state = torch.cat((x_new,y_new),dim=3).view(b,t,t,2,2).transpose(3,4).reshape(b,t,t,4)

        # Create a boolean mask to exclude the diagonal elements
        mask = ~torch.eye(t, dtype=bool).unsqueeze(0).repeat(b,1,1)
        rel_state = rel_state[mask].view(b, t, t-1, k)

        # Calculate absolute distance matrix of size (batch,n_agents,n_agents-1,1)
        r = torch.sqrt(rel_state[:,:,:,0].clone()**2 + rel_state[:,:,:,1].clone()**2).view(b,t,t-1,1)
        
        # Apply transformation to distance to have a distance closer to zero have a higher value, assymptotically going to zero.
        r_trans = (1/(1+torch.exp(-1+5.*(r-0.2))))

        # divide x and y by the distance to get the values as a function of the unit circle
        rel_state[:,:,:,0:2] = rel_state[:,:,:,0:2].clone()/r
        
        # add transformed distance vector to the state vector
        rel_state = torch.cat((rel_state,r_trans),dim=-1)
    
        b, t,_, k = rel_state.size()

        # transformer operations
        queries =  self.toqueries(state).view(b, t, h, k)
        keys =  self.tokeys(rel_state.view(b,t*(t-1),k)).view(b,t*(t-1),h,k)
        values =  self.tovalues(rel_state.view(b,t*(t-1),k)).view(b,t*(t-1),h,k)

        # Fold heads into the batch dimension
        queries = queries.transpose(1, 2).reshape(b * h, t, k)
        keys = keys.transpose(1, 2).reshape(b * h, t*(t-1), k)
        values = values.transpose(1, 2).reshape(b * h, t*(t-1), k)

        queries = queries.view(b*h,t,1,k).transpose(2,3)
        keys = keys.view(b*h,t,(t-1),k)
        values = values.view(b*h,t,(t-1),k)

        w_prime = torch.matmul(keys,queries)
        w_prime = w_prime / (k ** (1 / 2))
        w = F.softmax(w_prime, dim=2)
        
        x = torch.matmul(w.transpose(2,3),values).view(b*h,t,k)

        # Retrieve heads from batch dimension
        x = x.view(b,h,t,k)

        # Swap h, t back, unify heads
        x = x.transpose(1, 2).reshape(b, t, h * k)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN
        skip = x

        # Forward pass
        for layer in self.layers:
            x = F.relu(layer(x))
            x = torch.cat((x,skip),dim=2)

        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob

class TransformerTestActor(Actor):
    def __init__(self,
                 q_dim: int,
                 kv_dim: int,
                 out_dim: int,
                 num_heads: int = 3,
                 num_blocks: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()

        self.num_heads = num_heads
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.block1 = transformer.OwnRefTransformerBlock(q_dim,kv_dim,num_heads)

        self.layers = nn.ModuleList()
        in_dim = kv_dim * num_heads + q_dim
        for layer in range(2):
            self.layers.append(nn.Linear(in_dim, 256))
            self.layers.append(nn.ReLU())
            in_dim = 256


        log_std_layer = nn.Linear(in_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(in_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)
    
    def forward(self, state):

        x = self.block1(state)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)


        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob

class MultiHeadAdditiveActor(Actor):
    def __init__(self,
                 q_dim: int,
                 kv_dim: int,
                 out_dim: int,
                 num_heads: int = 3,
                 num_blocks: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.block1 = transformer.MultiHeadAdditiveAttentionBlock(q_dim,kv_dim,num_heads)

        self.layers = nn.ModuleList()
        in_dim = kv_dim * num_heads + q_dim
        for layer in range(2):
            self.layers.append(nn.Linear(in_dim, 256))
            self.layers.append(nn.ReLU())
            in_dim = 256


        log_std_layer = nn.Linear(in_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(in_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)

    def forward(self, state):

        x = self.block1(state)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)


        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob

class MultiHeadAdditiveActorBasic(Actor):
    def __init__(self,
                 q_dim: int,
                 kv_dim: int,
                 out_dim: int,
                 num_heads: int = 3,
                 num_blocks: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.block1 = transformer.MultiHeadAdditiveAttentionBlockBasic(q_dim,kv_dim,num_heads)

        self.layers = nn.ModuleList()
        in_dim = kv_dim * num_heads + q_dim
        for layer in range(2):
            self.layers.append(nn.Linear(in_dim, 256))
            self.layers.append(nn.ReLU())
            in_dim = 256


        log_std_layer = nn.Linear(in_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(in_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)

    def forward(self, state):

        x = self.block1(state)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)


        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob

class MultiHeadAdditiveActorBasicV2(Actor):
    def __init__(self,
                 q_dim: int,
                 kv_dim: int,
                 out_dim: int,
                 num_heads: int = 3,
                 num_blocks: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.block1 = transformer.MultiHeadAdditiveAttentionBlockBasic(q_dim,kv_dim,num_heads)

        self.layers = nn.ModuleList()
        in_dim = kv_dim * num_heads + q_dim
        for layer in range(2):
            self.layers.append(nn.Linear(in_dim, 256))
            self.layers.append(nn.ReLU())
            in_dim = 256


        log_std_layer = nn.Linear(in_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(in_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)

    def forward(self, state):

        x = self.block1(state)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)


        mu =  self.mu_layer(x)

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob

class TransformerTestActorAdditive(Actor):
    def __init__(self,
                 q_dim: int,
                 kv_dim: int,
                 out_dim: int,
                 num_blocks: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.block1 = transformer.OwnRefTransformerBlockAdditive(q_dim,kv_dim)

        self.layers = nn.ModuleList()
        in_dim = kv_dim + q_dim
        for layer in range(2):
            self.layers.append(nn.Linear(in_dim, 256))
            self.layers.append(nn.ReLU())
            in_dim = 256


        log_std_layer = nn.Linear(in_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(in_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)
    
    def forward(self, state):

        x = self.block1(state)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)


        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob

class MultiHeadNeuralNetworkActorBasic(Actor):
    def __init__(self,
                 q_dim: int,
                 kv_dim: int,
                 out_dim: int,
                 num_heads: int = 3,
                 num_blocks: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.block1 = transformer.MultiHeadNeuralNetworkBlockBasic(q_dim,kv_dim,num_heads)

        self.layers = nn.ModuleList()
        in_dim = kv_dim * num_heads + q_dim
        for layer in range(2):
            self.layers.append(nn.Linear(in_dim, 256))
            self.layers.append(nn.ReLU())
            in_dim = 256


        log_std_layer = nn.Linear(in_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(in_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)

    def forward(self, state):

        x = self.block1(state)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)


        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob

class MultiHeadNeuralNetworkActorBasicScore(Actor):
    def __init__(self,
                 q_dim: int,
                 kv_dim: int,
                 out_dim: int,
                 num_heads: int = 3,
                 num_blocks: int = 2,
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.block1 = transformer.MultiHeadNeuralNetworkBlockBasicScore(q_dim,kv_dim,num_heads)

        self.layers = nn.ModuleList()
        in_dim = kv_dim * num_heads + q_dim
        for layer in range(2):
            self.layers.append(nn.Linear(in_dim, 256))
            self.layers.append(nn.ReLU())
            in_dim = 256


        log_std_layer = nn.Linear(in_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(in_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)

    def forward(self, state):

        x = self.block1(state)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)


        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob

class LSTMActor_final(Actor):
    """
    Long short-term memory recurrent unit which has the following update rule:
        it ​= σ(W_xi * ​xt ​+ b_xi ​+ W_hi * ​h(t−1) ​+ b_hi​)
        ft​ = σ(W_xf * ​xt ​+ b_xf ​+ W_hf * ​h(t−1) ​+ b_hf​)
        gt ​= tanh(W_xg * ​xt ​+ b_xg ​+ W_hg * ​h(t−1) ​+ b_hg​)
        ot ​= σ(W_xo * ​xt ​+ b_xo​ + W_ho ​h(t−1) ​+ b_ho​)
        ct ​= ft​ ⊙ c(t−1) ​+ it ​⊙ gt​
        ht ​= ot​ ⊙ tanh(ct​)​
    """
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int,
                 sorting_strategy: str = 'random',
                 log_std_min: float= -20,
                 log_std_max: float=2):
        super().__init__()

        self.hidden_size = 15

        self.sorting_strategy = sorting_strategy

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # LSTM parameters
        self.weight_xh = nn.Parameter(torch.Tensor(in_dim, 4*15))
        self.weight_hh = nn.Parameter(torch.Tensor(15, 4*15))
        self.bias_xh = nn.Parameter(torch.Tensor(4*15))
        self.bias_hh = nn.Parameter(torch.Tensor(4*15))

        self.layers = nn.ModuleList()
        in_dim = 15 + 8
        for layer in range(2):
            self.layers.append(nn.Linear(in_dim, 256))
            self.layers.append(nn.ReLU())
            in_dim = 256

        log_std_layer = nn.Linear(in_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(in_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)
    

        # Initialize parameters
        self.reset_params()


    def reset_params(self):
        """
        Initialize network parameters.
        """

        std = 1.0 / torch.sqrt(torch.tensor(self.hidden_size))
        self.weight_xh.data.uniform_(-std, std)
        self.weight_hh.data.uniform_(-std, std)
        self.bias_xh.data.uniform_(-std, std)
        self.bias_hh.data.uniform_(-std, std)


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: input with shape (N, T, D) where N is number of samples, T is
                number of aircraft and D is the state vector size which must be equal to
                self.input_size.
        
        State should be [batch,n_agents,[x,y,vx,vy,v,cos(drift),sin(drift),track]]

        Returns:
            y: output with a shape of (N, T, H) where H is hidden size
        """

        rel_state_init = state[:,:,0:4]

        b, t, k = rel_state_init.size()
        
        # Subtracts this vector from itself to generate a relative state matrix 
        rel_state = rel_state_init.view(b,1,t,k) - rel_state_init.view(b,1,t,k).transpose(1,2)
        
        # Create x,y tensors for each aircraft by doubling the k dimension
        xy_state = rel_state.view(b,t,t,2,2).transpose(3,4)
        angle = state[:,:,-1]
        angle = -angle + 0.5 * torch.pi
        angle = angle.view(b,t,1)
        x_new = xy_state[:,:,:,0,:] * torch.cos(angle.view(b,1,t,1)) - xy_state[:,:,:,1,:]*torch.sin(angle.view(b,1,t,1))
        y_new = xy_state[:,:,:,0,:] * torch.sin(angle.view(b,1,t,1)) + xy_state[:,:,:,1,:]*torch.cos(angle.view(b,1,t,1))
        rel_state = torch.cat((x_new,y_new),dim=3).view(b,t,t,2,2).transpose(3,4).reshape(b,t,t,4)

        # Create a boolean mask to exclude the diagonal elements
        mask = ~torch.eye(t, dtype=bool).unsqueeze(0).repeat(b,1,1)
        rel_state = rel_state[mask].view(b, t, t-1, k)

        # Calculate absolute distance matrix of size (batch,n_agents,n_agents-1,1)
        r = torch.sqrt(rel_state[:,:,:,0].clone()**2 + rel_state[:,:,:,1].clone()**2).view(b,t,t-1,1)
        
        # Apply transformation to distance to have a distance closer to zero have a higher value, assymptotically going to zero.
        r_trans = (1/(1+torch.exp(-1+5.*(r-0.2))))

        # divide x and y by the distance to get the values as a function of the unit circle
        rel_state[:,:,:,0:2] = rel_state[:,:,:,0:2].clone()/r
        
        # add transformed distance vector to the state vector
        rel_state = torch.cat((rel_state,r_trans),dim=-1)
    
        b, t,_, k = rel_state.size()

        if self.sorting_strategy == 'random':
            pass
        elif self.sorting_strategy == 'ascending':
            column_values = rel_state[..., k-1]
            _, sort_indices = torch.sort(column_values, descending=True, dim=-1)
            sort_indices = sort_indices.view(b,t,t-1,1)
            sort_indices = sort_indices.expand(-1, -1, -1, k)
            rel_state = torch.gather(rel_state, dim=2, index=sort_indices)
        elif self.sorting_strategy == 'descending':
            column_values = rel_state[..., k-1]
            _, sort_indices = torch.sort(column_values, descending=False, dim=-1)
            sort_indices = sort_indices.view(b,t,t-1,1)
            sort_indices = sort_indices.expand(-1, -1, -1, k)
            rel_state = torch.gather(rel_state, dim=2, index=sort_indices)
        else:
            raise('invalid sorting strategy: choose from random, ascending or descending')


        # Fold aircraft into batch dimension
        x = rel_state.view(b*t,t-1,k)

        # Transpose input for efficient vectorized calculation. After transposing
        # the input will have (T-1, B*T, K).
        x = x.transpose(0, 1)

        # Unpack dimensions
        T, N, H = x.shape[0], x.shape[1], self.hidden_size

        # Initialize hidden and cell states to zero. There will be one hidden
        # and cell state for each input, so they will have shape of (N, H)
        h0 = torch.zeros(N, H, device=x.device)
        c0 = torch.zeros(N, H, device=x.device)

        ht_1 = h0
        ct_1 = c0
        for ac in range(T):
            # LSTM update rule
            xh = torch.addmm(self.bias_xh, x[ac], self.weight_xh) 
            hh = torch.addmm(self.bias_hh, ht_1, self.weight_hh)
            it = torch.sigmoid(xh[:, 0:H] + hh[:, 0:H])
            ft = torch.sigmoid(xh[:, H:2*H] + hh[:, H:2*H])
            gt = torch.tanh(xh[:, 2*H:3*H] + hh[:, 2*H:3*H])
            ot = torch.sigmoid(xh[:, 3*H:4*H] + hh[:, 3*H:4*H])
            ct = ft * ct_1 + it * gt
            ht = ot * torch.tanh(ct)

            # For the next iteration c(t-1) and h(t-1) will be current ct and ht
            ct_1 = ct
            ht_1 = ht

        # ht should be of shape (B*T, hidden_dim)
        # Retrieve aircraft from batch dimension
        
        x = ht.view(b,t,self.hidden_size)

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)

        mu =  self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        if self.test:
            return mu, log_prob

        return action, log_prob
