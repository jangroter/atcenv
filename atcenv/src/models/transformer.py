import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from typing import Tuple

"""
for NN attention (not implemented) maybe I can do:
keys_unfold = keys.view(b,t,t-1,h,in_dim)
queries_unfold = queries.view(b,t,h,in_dim)
queries_unfold = queries.view(b,t,1,h,in_dim)

kq = keys_unfold + queries_unfold

# kq.size() = (b,t,t-1,h,in_dim)

custom_weight_matrix = torch.nn.Parameter(torch.rand(h,in_dim,out_dim).uniform_(-0.1,0.1))


# value / context vector as calculated from each head
v_kq = torch.matmul(kq.transpose(2,3),custom_weight_matrix.transpose(1,2))

# v_kq.size() = (b,t,h,t-1,out_dim)

# then rearrange to get shape (b,t,t-1,h,out_dim)

"""

### ATTENTION MODULES

class RelativeNeuralNetworkAttentionMultiHead(nn.Module):
    """
    """

    def __init__(self, q_dim, kv_dim, num_heads=3):
        super(RelativeNeuralNetworkAttentionMultiHead, self).__init__()

        # These compute the queries, keys and values for all 
        # heads (as a single concatenated vector)
        # Generate the transformer matrices
        self.num_heads = num_heads

        self.tokeys    = nn.Linear(kv_dim, kv_dim, bias=False)
        self.toqueries = nn.Linear(q_dim, kv_dim, bias=False)
        self.tovalues  = nn.Linear(kv_dim*2, (1+kv_dim)*num_heads, bias=True)

        self.bias = nn.Parameter(torch.rand(kv_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(kv_dim, 1)
    
    def forward(self, q_x, kv_x):
        b, t,_, k = kv_x.size()
        h = self.num_heads
        
        keys = self.tokeys(kv_x).view(b,t,(t-1),k)
        queries =  self.toqueries(q_x).view(b,t,k)
        queries = queries.view(b,t,1,k)
        queries = queries.expand(*(-1,-1,t-1,-1))

        value_input = torch.cat((keys,queries),dim=-1)
        values = self.tovalues(value_input).view(b,t,t-1,h,k+1)

        score = values[:,:,:,:,0]
        values = values[:,:,:,:,1:]

        # score = self.score_proj(torch.tanh(keys + queries + self.bias)).squeeze(-1)
        
        # values = self.tovalues(kv_x)
        # values = values.view(b,t,t-1,h,k)
        
        w = F.softmax(score, dim=2)

        w = w.transpose(2,3).view(b,t,h,1,t-1)
        v = values.transpose(2,3)

        x = torch.matmul(w,v)
        x = x.view(b,t,k*h)
        return x

class RelativeNeuralNetworkAttentionMultiHeadScore(nn.Module):
    """
    """

    def __init__(self, q_dim, kv_dim, num_heads=3):
        super(RelativeNeuralNetworkAttentionMultiHeadScore, self).__init__()

        # These compute the queries, keys and values for all 
        # heads (as a single concatenated vector)
        # Generate the transformer matrices
        self.num_heads = num_heads

        self.tokeys    = nn.Linear(kv_dim, kv_dim*num_heads, bias=False)
        self.toqueries = nn.Linear(q_dim, kv_dim*num_heads, bias=False)
        self.tovalues  = nn.Linear(kv_dim+q_dim, kv_dim*num_heads, bias=True)

        self.bias = nn.Parameter(torch.rand(kv_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(kv_dim, 1)
    
    def forward(self, q_x, kv_x):
        b, t,_, k = kv_x.size()
        _,_,q = q_x.size()

        h = self.num_heads
        
        keys = self.tokeys(kv_x).view(b,t,(t-1),h,k)
        queries =  self.toqueries(q_x).view(b,t,h,k)
        queries = queries.view(b,t,1,h,k)

        score = self.score_proj(torch.tanh(keys + queries + self.bias)).squeeze(-1)

        q_x = q_x.view(b,t,1,q).expand(*(-1,-1,t-1,-1))
        value_input = torch.cat((kv_x,q_x),dim=-1)
        
        values = self.tovalues(value_input)
        values = values.view(b,t,t-1,h,k)
        
        w = F.softmax(score, dim=2)

        w = w.transpose(2,3).view(b,t,h,1,t-1)
        v = values.transpose(2,3)

        x = torch.matmul(w,v)
        x = x.view(b,t,k*h)
        return x

class RelativeAdditiveAttention(nn.Module):

    def __init__(self, q_dim, kv_dim):
        super(RelativeAdditiveAttention, self).__init__()

        # These compute the queries, keys and values for all 
        # heads (as a single concatenated vector)
        # Generate the transformer matrices
        self.tokeys    = nn.Linear(kv_dim, kv_dim, bias=False)
        self.toqueries = nn.Linear(q_dim, kv_dim, bias=False)
        self.tovalues  = nn.Linear(kv_dim, kv_dim, bias=False)

        self.bias = nn.Parameter(torch.rand(kv_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(kv_dim, 1)
    
    def forward(self, q_x, kv_x):
        b, t,_, k = kv_x.size()
        
        keys = self.tokeys(kv_x).view(b,t,(t-1),k)
        queries =  self.toqueries(q_x).view(b,t,1,k)

        score = self.score_proj(torch.tanh(keys + queries + self.bias)).squeeze(-1)
        
        values = self.tovalues(kv_x)
        
        w = F.softmax(score, dim=-1)

        x = torch.matmul(w.view(b,t,1,t-1),values)
        x = x.view(b,t,k)
        return x

class RelativeAdditiveAttentionMultiHead(nn.Module):
    """
    for how multihead additive
    **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
    """

    def __init__(self, q_dim, kv_dim, num_heads=3):
        super(RelativeAdditiveAttentionMultiHead, self).__init__()

        # These compute the queries, keys and values for all 
        # heads (as a single concatenated vector)
        # Generate the transformer matrices
        self.num_heads = num_heads

        self.tokeys    = nn.Linear(kv_dim, kv_dim*num_heads, bias=False)
        self.toqueries = nn.Linear(q_dim, kv_dim*num_heads, bias=False)
        self.tovalues  = nn.Linear(kv_dim, kv_dim*num_heads, bias=False)

        self.bias = nn.Parameter(torch.rand(kv_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(kv_dim, 1)
    
    def forward(self, q_x, kv_x):
        b, t,_, k = kv_x.size()
        h = self.num_heads
        
        keys = self.tokeys(kv_x).view(b,t,(t-1),h,k)
        queries =  self.toqueries(q_x).view(b,t,h,k)
        queries = queries.view(b,t,1,h,k)

        score = self.score_proj(torch.tanh(keys + queries + self.bias)).squeeze(-1)
        
        values = self.tovalues(kv_x)
        values = values.view(b,t,t-1,h,k)
        
        w = F.softmax(score, dim=2)

        w = w.transpose(2,3).view(b,t,h,1,t-1)
        v = values.transpose(2,3)

        x = torch.matmul(w,v)
        x = x.view(b,t,k*h)
        return x

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

### COMPOUND BLOCKS

## DOT PRODUCT

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
            nn.Linear(kv_dim * num_heads, 5 * kv_dim * num_heads),
            nn.ReLU(),
            nn.Linear(5 * kv_dim * num_heads, kv_dim * num_heads))
        
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

## ADDITIVE

class OwnRefTransformerBlockAdditive(nn.Module):
    def __init__(self, q_dim, kv_dim):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(OwnRefTransformerBlockAdditive, self).__init__()

        self.att = RelativeAdditiveAttention(q_dim, kv_dim)

        self.norm1 = nn.LayerNorm(kv_dim)

        self.ff = nn.Sequential(
            nn.Linear(kv_dim, 5 * kv_dim),
            nn.ReLU(),
            nn.Linear(5 * kv_dim, kv_dim))
        
        self.norm2 = nn.LayerNorm(kv_dim)

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

class MultiHeadAdditiveAttentionBlockQ(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(MultiHeadAdditiveAttentionBlockQ, self).__init__()

        self.att = RelativeAdditiveAttentionMultiHead(q_dim, kv_dim, num_heads)

        self.norm1 = nn.LayerNorm(kv_dim * num_heads)

        self.ff = nn.Sequential(
            nn.Linear(kv_dim * num_heads, 5 * kv_dim * num_heads),
            nn.ReLU(),
            nn.Linear(5 * kv_dim * num_heads, kv_dim * num_heads))
        
        self.norm2 = nn.LayerNorm(kv_dim * num_heads)

    def forward(self, state, action):
        """
        Forward pass of trasformer block.

        Args:
            x: input with shape of (b, k)
        
        Returns:
            y: output with shape of (b, k)
        """

        q_x, kv_x = self.get_input_tensors(state)
        q_x = torch.cat((q_x, action), dim=-1)

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

class MultiHeadAdditiveAttentionBlockQBasic(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(MultiHeadAdditiveAttentionBlockQBasic, self).__init__()

        self.att = RelativeAdditiveAttentionMultiHead(q_dim, kv_dim, num_heads)

    def forward(self, state, action):
        """
        Forward pass of trasformer block.

        Args:
            x: input with shape of (b, k)
        
        Returns:
            y: output with shape of (b, k)
        """

        q_x, kv_x = self.get_input_tensors(state)
        q_x = torch.cat((q_x, action), dim=-1)

        # Self-attend
        y = self.att(q_x,kv_x)

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

class MultiHeadAdditiveAttentionBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(MultiHeadAdditiveAttentionBlock, self).__init__()

        self.att = RelativeAdditiveAttentionMultiHead(q_dim, kv_dim, num_heads)

        self.norm1 = nn.LayerNorm(kv_dim * num_heads)

        self.ff = nn.Sequential(
            nn.Linear(kv_dim * num_heads, 5 * kv_dim * num_heads),
            nn.ReLU(),
            nn.Linear(5 * kv_dim * num_heads, kv_dim * num_heads))
        
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

class MultiHeadAdditiveAttentionBlockBasic(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(MultiHeadAdditiveAttentionBlockBasic, self).__init__()

        self.att = RelativeAdditiveAttentionMultiHead(q_dim, kv_dim, num_heads)

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

## NEURAL NETWORK

class MultiHeadNeuralNetworkBlockBasic(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(MultiHeadNeuralNetworkBlockBasic, self).__init__()

        self.att = RelativeNeuralNetworkAttentionMultiHead(q_dim, kv_dim, num_heads)

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

class MultiHeadNeuralNetworkBlockQBasic(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(MultiHeadNeuralNetworkBlockQBasic, self).__init__()

        self.att = RelativeNeuralNetworkAttentionMultiHead(q_dim, kv_dim, num_heads)

    def forward(self, state, action):
        """
        Forward pass of trasformer block.

        Args:
            x: input with shape of (b, k)
        
        Returns:
            y: output with shape of (b, k)
        """

        q_x, kv_x = self.get_input_tensors(state)
        q_x = torch.cat((q_x, action), dim=-1)

        # Self-attend
        y = self.att(q_x,kv_x)

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

class MultiHeadNeuralNetworkBlockBasicScore(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(MultiHeadNeuralNetworkBlockBasicScore, self).__init__()

        self.att = RelativeNeuralNetworkAttentionMultiHeadScore(q_dim, kv_dim, num_heads)

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

class MultiHeadNeuralNetworkBlockQBasicScore(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        """
        Basic transformer block.

        Args:
            k: embedding dimension
            heads: number of heads (k mod heads must be 0)

        """
        super(MultiHeadNeuralNetworkBlockQBasicScore, self).__init__()

        self.att = RelativeNeuralNetworkAttentionMultiHeadScore(q_dim, kv_dim, num_heads)

    def forward(self, state, action):
        """
        Forward pass of trasformer block.

        Args:
            x: input with shape of (b, k)
        
        Returns:
            y: output with shape of (b, k)
        """

        q_x, kv_x = self.get_input_tensors(state)
        q_x = torch.cat((q_x, action), dim=-1)

        # Self-attend
        y = self.att(q_x,kv_x)

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