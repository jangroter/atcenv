import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from typing import Tuple

from abc import ABC, abstractmethod

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
        self.tokeys    = nn.Linear(in_dim, in_dim*num_heads, bias=False)
        self.toqueries = nn.Linear(in_dim, in_dim*num_heads, bias=False)
        self.tovalues  = nn.Linear(in_dim, in_dim*num_heads, bias=False)

        self.layers = nn.ModuleList()
        in_dim = in_dim * num_heads + in_dim
        for layer in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim

        log_std_layer = nn.Linear(hidden_dim, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(hidden_dim, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:  

        rel_state_init = state

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





