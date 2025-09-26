import torch.nn as nn
import torch

class SVDModel(nn.Module):
    def __init__(self, n_users, n_items, ratings_mean, n_factors=50):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        
        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        
        # Global mean
        self.global_bias = nn.Parameter(torch.tensor([ratings_mean]))

    def forward(self, u, i):
        dot = (self.user_emb(u) * self.item_emb(i)).sum(1)
        return dot + self.user_bias(u).squeeze() + self.item_bias(i).squeeze() + self.global_bias

#prediction = user_vector · item_vector + biases