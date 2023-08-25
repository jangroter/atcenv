import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from abc import ABC, abstractmethod

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

class Critic_Q(ABC, nn.Module):

    def __init__(self):
        super(Critic_Q, self).__init__()
    
    @abstractmethod
    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

class FeedForward_Q(Critic_Q):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()

        in_dim = state_dim + action_dim
        for layer in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.out = nn.Linear(hidden_dim,1)
        self.out = init_layer_uniform(self.out)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((observation, action), dim=-1)
        for layer in self.layers:
            x = layer(x)
        value = self.out(x)
        return value

class RelTransformerQ(Critic_Q):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_heads: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2):
        super().__init__()

        self.num_heads = num_heads

        # Generate the transformer matrices
        self.tokeys    = nn.Linear(state_dim, state_dim*num_heads, bias=False)
        self.toqueries = nn.Linear(state_dim + action_dim, state_dim*num_heads, bias=False)
        self.tovalues  = nn.Linear(state_dim, state_dim*num_heads, bias=False)

        self.layers = nn.ModuleList()

        in_dim = state_dim * num_heads + state_dim + action_dim
        for layer in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.out = nn.Linear(hidden_dim, 1)
        self.out = init_layer_uniform(self.out)

    def forward(
        self, 
        state:torch.Tensor, 
        action: torch.Tensor) -> torch.Tensor:

        q_state = torch.cat((state, action), dim=-1)

        rel_state_init = state

        h = self.num_heads
        b, t, k = rel_state_init.size()
        
        # Subtracts this vector from itself to generate a relative state matrix 
        rel_state = rel_state_init.view(b,1,t,k) - rel_state_init.view(b,1,t,k).transpose(1,2)

        # Create a boolean mask to exclude the diagonal elements
        mask = ~torch.eye(t, dtype=bool).unsqueeze(0).repeat(b,1,1)
        rel_state = rel_state[mask].view(b, t, t-1, k)

        # update values of the state vector size
        b, t,_, k = rel_state.size()

        # transformer operations
        queries =  self.toqueries(q_state).view(b, t, h, k)
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

        x = torch.cat((x,q_state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)

        value = self.out(x)

        return value





