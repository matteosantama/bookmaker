import pkbar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

import attn_dl

# http://peterbloem.nl/blog/transformers
# This code was written by Peter Bloem for the article Transformers from Scratch
# implements multi-headed attention and a simple "transformer" block
weights = []
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
        weights.append(dot)
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
        # have to sum over the first dimension to get the scores for each game
        return torch.sum(x, dim = 1)


def train_model(num_epochs, batch_size, learning_rate, heads, depth, loss_func):
    """
    Trains the model with the specified parameters and returns the model
    as well as the point estimate of losses on the validation set over each
    epoch.
    """
    global weights
    weights = []
    pbar = pkbar.Pbar(name='Training Model', target = num_epochs)
    
    # Load data as torch.tensors
    _, x_train, y_train = attn_dl.load_vectorized_data('train')
    _, x_validate, y_validate = attn_dl.load_vectorized_data('dev')

    model = TransformerModel(x_train.shape[2], heads, depth).double()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_ds = TensorDataset(x_train, y_train)
    
    # Notice we shuffle our training data so the seasons are mixed!
    train_dl = DataLoader(train_ds, batch_size=batch_size, 
                          shuffle=True)

    validate_ds = TensorDataset(x_validate, y_validate)
    validate_dl = DataLoader(validate_ds, 
                             batch_size=batch_size * 2)

    # L1 loss is more robust to outliers
    losses = []
    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred.double(), yb.double())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            epoch_loss = sum(loss_func(model(xb), yb) for xb, yb in validate_dl)
            if epoch % 10 == 0:
                print(epoch_loss/len(xb))
            losses.append( epoch_loss / len(xb) )
        pbar.update(epoch)
    return model, losses

def get_weights():
    return weights

