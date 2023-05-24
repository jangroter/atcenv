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
        self.tokeys    = nn.Linear(5, 5*num_heads, bias=False)
        self.toqueries = nn.Linear(in_dim, 5*num_heads, bias=False)
        self.tovalues  = nn.Linear(5, 5*num_heads, bias=False)

        # Feedforward network
        self.hidden1 = nn.Linear(5*num_heads+in_dim, hidden_dim1)
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)

        log_std_layer = nn.Linear(hidden_dim2, out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        self.print = False
        self.count = 0

        mu_layer = nn.Linear(hidden_dim2, out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:  
        # First takes for all aircraft the first 4 entries of the state vector (x,y,vx,vy).
        rel_state_init = state[:,:,0:4]

        h = self.num_heads
        b, t, k = rel_state_init.size()
        
        # Subtracts this vector from itself to generate a relative state matrix 
        rel_state = rel_state_init.view(b,1,t,k) - rel_state_init.view(b,1,t,k).transpose(1,2)

        # Create a boolean mask to exclude the diagonal elements
        mask = ~torch.eye(t, dtype=bool).unsqueeze(0).repeat(b,1,1)

        # Apply the mask to the relative states tensor
        rel_state = rel_state[mask].view(b, t, t-1, k)

        # Calculate absolute distance matrix of size (batch,n_agents,n_agents-1,1)
        r = torch.sqrt(rel_state[:,:,:,0]**2 + rel_state[:,:,:,1]**2).view(b,t,t-1,1)
        
        # Apply transformation to distance to have a distance closer to zero have a higher value, assymptotically going to zero.
        r_trans = (1/(1+torch.exp(-1+5.*(r-0.2))))

        # divide x and y by the distance to get the values as a function of the unit circle
        rel_state[:,:,:,0:2] = rel_state[:,:,:,0:2]/r
        
        # add transformed distance vector to the state vector
        rel_state = torch.cat((rel_state,r_trans),dim=-1)

        # update values of the state vector size
        b, t,_, k = rel_state.size()
        # rel_state.size -> [b,t,t,k]

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
        
        if self.print:
            self.count += 1
            if self.count == 10:
                print(state[0,0,:])
                print(state[0,1,:])
                print(rel_state[0,0,:,:])
                print(keys[0,0,:,:])
                print(values[0,0,:,:])
                print(w[0,0,:])
                print(x[0,0,:])
                self.print = False
                self.count = 0

        x = torch.cat((x,state),dim=2) # add absolute state information also before passing through the FFN

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