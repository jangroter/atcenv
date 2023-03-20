import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int=1,
        log_std_min: float= -20,
        log_std_max: float=2,
        hidden_dim1: int=256,
        hidden_dim2: int=256):
        super(Actor, self).__init__()

        self.num_heads = num_heads

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Generate the transformer matrices
        self.tokeys    = nn.Linear(in_dim, in_dim*num_heads, bias=False)
        self.toqueries = nn.Linear(in_dim, in_dim*num_heads, bias=False)
        self.tovalues  = nn.Linear(in_dim, in_dim*num_heads, bias=False)

        # Feedforward network
        self.hidden1 = nn.Linear(in_dim*num_heads, hidden_dim1)
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)

        log_std_layer = nn.Linear(hidden_dim2, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(hidden_dim2, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:

        b, t, k = state.size()
        h = self.num_heads

        # Project state to queries, keys and values
        queries = self.toqueries(state).view(b, t, h, k)
        keys    = self.tokeys(state).view(b, t, h, k)
        values  = self.tovalues(state).view(b, t, h, k)

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

        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))

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

        return action, log_prob


class CriticQ(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim1: int=256,
        hidden_dim2: int=256):
        super().__init__()

        self.hidden1 = nn.Linear(in_dim, hidden_dim1)
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim1, 1)
        self.out = init_layer_uniform(self.out)

    def forward(
        self, 
        state:torch.Tensor, 
        action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value

class CriticV(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim1: int=256,
        hidden_dim2: int=256):
        super().__init__()

        self.hidden1 = nn.Linear(in_dim, hidden_dim1)
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, 1)
        self.out = init_layer_uniform(self.out)

    def forward(
        self, 
        state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value