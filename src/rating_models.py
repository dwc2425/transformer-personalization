"""Week 4 models: per-user rating prediction from the movie's 19 genre flags.
 
LogisticRegression5:  z = W g + b,                       W in R^{5x19}, b in R^5   (100 params)
GenreTransformer:     t_j = e_j + g_j v_j  (all 19 tokens kept, fixed order)
                      A = softmax_row((T Wq)(T Wk)^T / sqrt(d)),  U = T + A (T Wv)
                      z = R flatten(U) + b                                         (2,901 params)
Both return RAW logits over the 5 rating classes (k = rating - 1).
"""
import math
import torch
import torch.nn as nn
 
G, D, C = 19, 16, 5
 
 
class LogisticRegression5(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(G, C)
 
    def forward(self, g):  # g [B,19] -> [B,5]
        return self.linear(g)
 
 
class GenreTransformer(nn.Module):
    def __init__(self, d=D, e_std=0.5, v_std=0.5):
        super().__init__()
        self.d = d
        self.e = nn.Parameter(torch.randn(G, d) * e_std)
        self.v = nn.Parameter(torch.randn(G, d) * v_std)
        self.Wq = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.Wk = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.Wv = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.readout = nn.Linear(G * d, C)  # R in R^{5 x 304}, b in R^5
 
    def tokens(self, g):  # [B,19] -> [B,19,d]
        return self.e[None] + g[..., None] * self.v[None]
 
    def forward(self, g):
        T = self.tokens(g)
        A = torch.softmax((T @ self.Wq) @ (T @ self.Wk).transpose(1, 2) / math.sqrt(self.d), dim=-1)
        U = T + A @ (T @ self.Wv)
        return self.readout(U.flatten(1))
