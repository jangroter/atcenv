import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from abc import ABC, abstractmethod

import atcenv.src.models.transformer as transformer

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
        self.tokeys    = nn.Linear(4, 4*num_heads, bias=False)
        self.toqueries = nn.Linear(state_dim+action_dim, 4*num_heads, bias=False)
        self.tovalues  = nn.Linear(4, 4*num_heads, bias=False)

        self.layers = nn.ModuleList()

        in_dim = 4 * num_heads + state_dim + action_dim
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

        rel_state_init = state[:,:,0:4]

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

class RelDistTransformerQ(Critic_Q):

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
        self.tokeys    = nn.Linear(5, 5*num_heads, bias=False)
        self.toqueries = nn.Linear(state_dim+action_dim, 5*num_heads, bias=False)
        self.tovalues  = nn.Linear(5, 5*num_heads, bias=False)

        self.layers = nn.ModuleList()

        in_dim = 5 * num_heads + state_dim + action_dim
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
    
class OwnRefTransformerQ(Critic_Q):

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
        self.tokeys    = nn.Linear(5, 5*num_heads, bias=False)
        self.toqueries = nn.Linear(state_dim+action_dim, 5*num_heads, bias=False)
        self.tovalues  = nn.Linear(5, 5*num_heads, bias=False)

        self.layers = nn.ModuleList()

        in_dim = 5 * num_heads + state_dim + action_dim
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

class OwnRefTransformerSkipQ(Critic_Q):

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
        self.tokeys    = nn.Linear(5, 5*num_heads, bias=False)
        self.toqueries = nn.Linear(state_dim+action_dim, 5*num_heads, bias=False)
        self.tovalues  = nn.Linear(5, 5*num_heads, bias=False)

        self.layers = nn.ModuleList()

        input = 5 * num_heads + state_dim + action_dim
        for layer in range(num_layers):
            self.layers.append(nn.Linear(input, hidden_dim))
            input = hidden_dim+state_dim+action_dim+5*num_heads

        self.out = nn.Linear(input, 1)
        self.out = init_layer_uniform(self.out)

    def forward(
        self, 
        state:torch.Tensor, 
        action: torch.Tensor) -> torch.Tensor:

        q_state = torch.cat((state, action), dim=-1)

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
        skip = x

        # Forward pass
        for layer in self.layers:
            x = F.relu(layer(x))
            x = torch.cat((x,skip),dim=2)

        value = self.out(x)

        return value

class AbsTransformerQ(Critic_Q):
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
        self.tokeys    = nn.Linear(4, 4*num_heads, bias=False)
        self.toqueries = nn.Linear(state_dim + action_dim, 4*num_heads, bias=False)
        self.tovalues  = nn.Linear(4, 4*num_heads, bias=False)

        self.layers = nn.ModuleList()

        in_dim = 4 * num_heads + state_dim + action_dim
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

        t_state = state[:,:,0:4]

        b, t, k = t_state.size()
        h = self.num_heads

        # Project state to queries, keys and values
        queries = self.toqueries(q_state).view(b, t, h, k)
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

        x = torch.cat((x,q_state),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)

        value = self.out(x)

        return value

class TransformerTestQ(Critic_Q):
    def __init__(self,
                 q_dim: int,
                 kv_dim: int,
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
        in_dim = kv_dim * num_heads + q_dim + 2
        for layer in range(2):
            self.layers.append(nn.Linear(in_dim, 256))
            self.layers.append(nn.ReLU())
            in_dim = 256

        self.out = nn.Linear(in_dim, 1)
        self.out = init_layer_uniform(self.out)
    
    def forward(self, state, action):

        x = self.block1(state)

        x = torch.cat((x,state,action),dim=2) # add absolute state information also before passing through the FFN

        # Forward pass
        for layer in self.layers:
            x = layer(x)

        value = self.out(x)
        
        return value









