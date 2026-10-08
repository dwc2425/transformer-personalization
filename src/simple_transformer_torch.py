"""One-attention-layer predictor (PyTorch port of Week 2).

Q, K in R^{n x d_kq}, V in R^{n x d_v} are learned per-movie rows. For a history H:
    E_H = softmax(Q_H K_H^T) V_H      (unscaled; keys masked so padding is excluded)
    h   = mean_{i in H} (E_H)_i       (masked mean)
    z   = R h + c                     (raw logits over all n movies)
Set-based: no positional encodings.
"""
import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    def __init__(self, n, d_kq=16, d_v=16, qk_std=0.5, v_std=0.1):
        super().__init__()
        self.n = n
        self.Q = nn.Parameter(torch.randn(n, d_kq) * qk_std)
        self.K = nn.Parameter(torch.randn(n, d_kq) * qk_std)
        self.V = nn.Parameter(torch.randn(n, d_v) * v_std)
        self.readout = nn.Linear(d_v, n)  # R in R^{n x d_v}, c in R^n

    def attend(self, hist, mask):
        """Returns E_H [B,L,d_v]; padded key slots get zero attention weight."""
        idx = hist.clamp(max=self.n - 1)  # pad index n -> any valid row; masked out below
        q, k, v = self.Q[idx], self.K[idx], self.V[idx]
        scores = q @ k.transpose(1, 2)  # unscaled
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        return torch.softmax(scores, dim=-1) @ v

    def represent(self, hist, mask):
        E = self.attend(hist, mask)
        m = mask.unsqueeze(-1).float()
        return (E * m).sum(1) / m.sum(1).clamp(min=1)  # padded query rows excluded from pooling

    def forward(self, hist, mask):
        return self.readout(self.represent(hist, mask))
