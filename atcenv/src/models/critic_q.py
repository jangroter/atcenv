import torch.nn as nn
import torch
import torch.nn.functional as f
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
                 in_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()

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

        




