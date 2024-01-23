import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from typing import Tuple

class RelativeMultiHeadAttention(nn.Module):
    """
    Wide mult-head self-attention layer.

    Args:
        k: embedding dimension
        heads: number of heads (k mod heads must be 0)

    """
    def __init__(self, q_dim, kv_dim, num_heads=3):
        super(RelativeMultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        # These compute the queries, keys and values for all 
        # heads (as a single concatenated vector)
        # Generate the transformer matrices
        self.tokeys    = nn.Linear(kv_dim, kv_dim*num_heads, bias=False)
        self.toqueries = nn.Linear(q_dim, kv_dim*num_heads, bias=False)
        self.tovalues  = nn.Linear(kv_dim, kv_dim*num_heads, bias=False)

        # This unifies the outputs of the different heads into 
        # a single k-vector
        self.unifyheads = nn.Linear(kv_dim * num_heads, q_dim)
        
    def forward(self, q_x, kv_x):

        """ State should be [batch,n_agents,[x,y,vx,vy,v,cos(drift),sin(drift),track]]"""
    
        b, t,_, k = kv_x.size()
        h = self.num_heads

        # transformer operations
        queries =  self.toqueries(q_x).view(b, t, h, k)
        keys =  self.tokeys(kv_x.view(b,t*(t-1),k)).view(b,t*(t-1),h,k)
        values =  self.tovalues(kv_x.view(b,t*(t-1),k)).view(b,t*(t-1),h,k)

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

        return x
    
class MultiHeadAttention(nn.Module):
    """
    Wide mult-head self-attention layer.

    Args:
        k: embedding dimension
        heads: number of heads (k mod heads must be 0)

    """
    def __init__(self, in_dim, num_heads=3):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        # These compute the queries, keys and values for all 
        # heads (as a single concatenated vector)
        self.tokeys    = nn.Linear(in_dim, in_dim * num_heads, bias=False)
        self.toqueries = nn.Linear(in_dim, in_dim * num_heads, bias=False)
        self.tovalues  = nn.Linear(in_dim, in_dim * num_heads, bias=False)

        # This unifies the outputs of the different heads into 
        # a single in_dim-vector
        self.unifyheads = nn.Linear(in_dim * num_heads, in_dim)
        
    def forward(self, x):

        b, t, k = x.size()
        h = self.num_heads

        ########################################################################
        #     TODO: Perform wide multi-head self-attention operation with      #
        #   learnable query, key and value mappings. Calculate w_prime, apply  #
        #  scaling, softmax and compute and concatenate the output tensors y,  #
        #        and transform back to the original embedding dimension.       #
        ########################################################################

        # Project input to queries, keys and values
        queries = self.toqueries(x).view(b, t, h, k)
        keys    = self.tokeys(x).view(b, t, h, k)
        values  = self.tovalues(x).view(b, t, h, k)

        # Fold heads into the batch dimension
        keys = keys.transpose(1, 2).reshape(b * h, t, k)
        queries = queries.transpose(1, 2).reshape(b * h, t, k)
        values = values.transpose(1, 2).reshape(b * h, t, k)
        
        # Compute attention weights
        w_prime = torch.bmm(queries, keys.transpose(1, 2))
        w_prime = w_prime / (k ** (1 / 2))
        w = F.softmax(w_prime, dim=2)

        # Apply the self-attention to the values
        y = torch.bmm(w, values).view(b, h, t, k)

        # Swap h, t back, unify heads
        y = y.transpose(1, 2).reshape(b, t, h * k)

        y = self.unifyheads(y)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return y

class OwnRefTransformerBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(OwnRefTransformerBlock, self).__init__()

        self.att = RelativeMultiHeadAttention(q_dim, kv_dim, num_heads=num_heads)

        self.norm1 = nn.LayerNorm(kv_dim * num_heads)

        self.ff = nn.Sequential(
            nn.Linear(kv_dim * num_heads, 4 * kv_dim * num_heads),
            nn.ReLU(),
            nn.Linear(4 * kv_dim * num_heads, kv_dim * num_heads))
        
        self.norm2 = nn.LayerNorm(kv_dim * num_heads)

    def forward(self, state):
        """
        Forward pass of trasformer block.

        Args:
            x: input with shape of (b, k)
        
        Returns:
            y: output with shape of (b, k)
        """

        q_x, kv_x = self.get_input_tensors(state)

        # Self-attend
        y = self.att(q_x,kv_x)

        # First residual connection
        # x = q_x + y
        x = y

        # Normalize
        x = self.norm1(x)

        # Pass through feed-forward network
        y = self.ff(x)

        # Second residual connection
        x = x + y

        # Again normalize
        y = self.norm2(x)

        return y
    
    def get_input_tensors(self, state):
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

        q_x = state
        kv_x = rel_state

        return q_x, kv_x
    
class TransformerBlock(nn.Module):
    def __init__(self, in_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(TransformerBlock, self).__init__()

        self.att = MultiHeadAttention(in_dim, num_heads=num_heads)

        self.norm1 = nn.LayerNorm(in_dim)

        self.ff = nn.Sequential(
            nn.Linear(in_dim, 4 * in_dim),
            nn.ReLU(),
            nn.Linear(4 * in_dim, in_dim))
        
        self.norm2 = nn.LayerNorm(in_dim)

    def forward(self, x):
        """
        Forward pass of trasformer block.

        Args:
            x: input with shape of (b, k)
        
        Returns:
            y: output with shape of (b, k)
        """
        ########################################################################
        #        TODO: Perform the forward pass of a transformer block         #
        #                       as depicted in the image.                      #
        ########################################################################

        # Self-attend
        y = self.att(x)

        # First residual connection
        x = x + y

        # Normalize
        x = self.norm1(x)

        # Pass through feed-forward network
        y = self.ff(x)

        # Second residual connection
        x = x + y

        # Again normalize
        y = self.norm2(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return y
    

