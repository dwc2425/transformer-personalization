"""Multinomial logistic regression on fixed history features x_j(H) = 1[j in H] / |H|."""
import torch
import torch.nn as nn
 
 
class LogisticRegression(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.linear = nn.Linear(n, n)  # W in R^{n x n}, b in R^n -> z = W x + b
 
    def features(self, hist, mask):
        """hist [B,L] movie indices (pad = n), mask [B,L] bool -> x [B,n], rows sum to 1."""
        B = hist.shape[0]
        x = torch.zeros(B, self.n + 1, device=hist.device)
        x.scatter_add_(1, hist, mask.float())  # pad lands in the extra column, then dropped
        x = x[:, : self.n]
        return x / mask.sum(1, keepdim=True).clamp(min=1)
 
    def forward(self, hist, mask):
        return self.linear(self.features(hist, mask))  # RAW logits (no softmax) for CrossEntropyLoss
