import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

# http://peterbloem.nl/blog/transformers
# This code was written by Peter Bloem for the article Transformers from Scratch
# implements multi-headed attention and a simple "transformer" block
class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k, self.heads = k, heads
        self.tokeys    = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues  = nn.Linear(k, k * heads, bias=False)

        # This unifies the outputs of the different heads into 
        # a single k-vector
        self.unifyheads = nn.Linear(heads * k, k)
    def forward(self, x):
        b, t, k = x.size()
        
        h = self.heads
        queries = self.toqueries(x).view(b, t, h, k)
        keys    = self.tokeys(x)   .view(b, t, h, k)
        values  = self.tovalues(x) .view(b, t, h, k)

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)
        
        queries = queries / (k ** (1/4))
        keys    = keys / (k ** (1/4))
        
        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # - dot has size (b*h, t, t) containing raw weights

        dot = F.softmax(dot, dim=2) 
        # - dot now contains row-wise normalized weights
        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, k)
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unifyheads(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
          nn.Linear(k, 4 * k),
          nn.ReLU(),
          nn.Linear(4 * k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        return self.norm2(fedforward + x)

# also from Peter Bloem, modified to output regression output instead of
# class probabilities
class TransformerModel(nn.Module): 
    def __init__(self, k, heads, depth):
        super().__init__()

        # The sequence of transformer blocks that does all the 
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to a linear output
        self.toscore = nn.Linear(k, 1)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing 
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the 
                 classes (where c is the nr. of classes).
        """
        x = self.tblocks(x)
        x = self.toscore(x)
        return torch.sum(x, dim = 1)



# #https://tomekkorbak.com/2020/06/26/implementing-attention-in-pytorch/
# class Attention(nn.Module):

#     def __init__(self, encoder_dim: int, decoder_dim: int):
#         super().__init__()
#         self.encoder_dim = encoder_dim
#         self.decoder_dim = decoder_dim

#     def forward(self, 
#         query: torch.Tensor,  # [decoder_dim]
#         values: torch.Tensor, # [seq_length, encoder_dim]
#         ):
#         weights = self._get_weights(query, values) # [seq_length]
#         weights = torch.nn.functional.softmax(weights, dim=0)
#         return weights @ values  # [encoder_dim]
    
# class MultiplicativeAttention(Attention):
#     # aka scaled dot product attention
#     def __init__(self, encoder_dim: int, decoder_dim: int):
#         super().__init__(encoder_dim, decoder_dim)
#         self.W = torch.nn.Parameter(torch.FloatTensor(
#             self.decoder_dim, self.encoder_dim).uniform_(-0.1, 0.1))

#     def _get_weights(self,
#         query: torch.Tensor,  # [decoder_dim]
#         values: torch.Tensor, # [seq_length, encoder_dim]
#     ):
#         weights = query @ self.W @ values.T  # [seq_length]
#         return weights/np.sqrt(self.decoder_dim)  # [seq_length]

# class Attn_Stack(nn.Module):
#     def __init__(self, input_dim, H):
#         super(Attn_Stack, self).__init__()
#         self.attn1 = MultiplicativeAttention(input_dim, input_dim)
#         self.bn1 = nn.BatchNorm1d(input_dim)
#         self.linear1 = nn.Linear(input_dim, H)
#         #self.drop_layer = nn.Dropout(p = 0.5)
#         self.bn2 = nn.BatchNorm1d(H)
        

#     def forward(self, x):
#         x = self.attn1(x, x)
#         x = self.bn1(x)
#         x = F.relu(self.linear1(x))
#         #x = self.drop_layer(x)
#         x = self.bn2(x)
    
#         return x

# class Attn_Net(nn.Module):
#     def __init__(self, input_dim):
#         super(Attn_Net, self).__init__()

#         self.l1 = Attn_Stack(input_dim, input_dim//2)
#         self.l4 = Attn_Stack(input_dim//2, input_dim//2)
#         # we add a final regression layer
#         self.lastlinear = nn.Linear(input_dim//2, 1)
    
#     def forward(self, x):
#         x = self.l1(x)
#         x = self.l4(x)
#         x = self.lastlinear(x)
#         return x