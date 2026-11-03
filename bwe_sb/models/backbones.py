from __future__ import annotations
"""Selective state space trunk and the quantum-inspired amplitude modules: angle embedding, MPS head and MPS projection."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMLayer(nn.Module):
    """State-space layer whose transition depends on the INPUT."""
    def __init__(self, dim, n_state=8):
        super().__init__()
        self.dim, self.n = dim, n_state
        self.in_proj = nn.Linear(dim, 2 * dim)
        self.dt_proj = nn.Linear(dim, dim)
        self.bc_proj = nn.Linear(dim, 2 * n_state)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n_state + 1).float()
                                            .repeat(dim, 1)))
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        self.conv = nn.Conv1d(dim, dim, 4, padding=3, groups=dim)

    def forward(self, x):
        B_, C_, T = x.shape
        u = x.transpose(1, 2)
        u, gate = self.in_proj(u).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :T].transpose(1, 2)
        u = F.silu(u)
        dt = F.softplus(self.dt_proj(u))
        Bm, Cm = self.bc_proj(u).chunk(2, dim=-1)
        A = -self.A_log.exp()
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dBu = (dt * u).unsqueeze(-1) * Bm.unsqueeze(2)
        h = u.new_zeros(B_, C_, self.n)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dBu[:, t]
            ys.append(h)
        y = (torch.stack(ys, dim=1) * Cm.unsqueeze(2)).sum(-1)
        y = y + u * self.D.view(1, 1, -1)
        return self.out_proj(y * F.silu(gate)).transpose(1, 2)

class SelectiveSSMStack(nn.Module):
    def __init__(self, dim, depth=4, n_state=8, **_):
        super().__init__()
        self.layers = nn.ModuleList([SelectiveSSMLayer(dim, n_state) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in range(depth)])
        self.out = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        for ssm, nrm in zip(self.layers, self.norms):
            h = nrm(x.transpose(1, 2)).transpose(1, 2)
            x = x + ssm(h)
        return self.out(x.transpose(1, 2)).transpose(1, 2)


class AngleEmbedding(nn.Module):
    """Angle/amplitude embedding of the amplitude, computed classically."""
    def __init__(self, n_bins, n_feat=16, amplitude_norm=True):
        super().__init__()
        self.proj = nn.Linear(n_bins, n_feat)
        self.scale = nn.Parameter(torch.ones(n_feat))
        self.amplitude_norm = amplitude_norm
        self.out_dim = 2 * n_feat

    def forward(self, mag):
        z = self.proj(mag.transpose(1, 2))
        if self.amplitude_norm:
            z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        th = z * self.scale
        return torch.cat([th.cos(), th.sin()], dim=-1).transpose(1, 2)

class MPSHead(nn.Module):
    """Matrix-product-state (tensor-train) factorisation of the amplitude output head."""
    def __init__(self, dim, n_bins, rank=None, cores=8):
        super().__init__()
        if rank is None:
            rank = max(4, min(16, int(math.sqrt(dim) / 2)))
        self.n_bins, self.cores, self.rank = n_bins, cores, rank
        self.seg = math.ceil(n_bins / cores)
        self.inp = nn.Linear(dim, rank)
        self.core = nn.Parameter(torch.randn(cores, rank, rank, self.seg) * (rank ** -0.75))
        self.bias = nn.Parameter(torch.zeros(n_bins))
        self.n_params = cores * rank * rank * self.seg + dim * rank + n_bins
        self.dense_params = dim * n_bins
        if self.n_params >= self.dense_params:
            print(f"  [MPSHead] rank={rank} at dim={dim} costs {self.n_params/1e3:.1f}K vs "
                  f"{self.dense_params/1e3:.1f}K dense - not a compression; lower the rank",
                  flush=True)

    def forward(self, h):
        B, T, _ = h.shape
        v = self.inp(h)
        outs = []
        for c in range(self.cores):
            o = torch.einsum("btr,rqs->btqs", v, self.core[c])
            outs.append(o.sum(2))
            v = o.mean(-1)
            v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.cat(outs, dim=-1)[..., :self.n_bins] + self.bias

class MPSProj(nn.Module):
    """Matrix-product-state factorisation of the amplitude INPUT projection."""
    def __init__(self, n_in, dim, rank=None, cores=8):
        super().__init__()
        if rank is None:
            rank = max(4, min(16, int(math.sqrt(dim) / 2)))
        self.n_in, self.dim, self.cores, self.rank = n_in, dim, cores, rank
        self.seg = math.ceil(n_in / cores)
        self.pad = self.cores * self.seg - n_in
        self.core = nn.Parameter(torch.randn(cores, rank, rank, self.seg) * (rank ** -0.75))
        self.out = nn.Linear(rank + 1, dim)
        self.n_params = cores * rank * rank * self.seg + (rank + 1) * dim + dim
        self.dense_params = n_in * dim
        if self.n_params >= self.dense_params:
            print(f"  [MPSProj] rank={rank} at dim={dim} costs {self.n_params/1e3:.1f}K vs "
                  f"{self.dense_params/1e3:.1f}K dense - not a compression; lower the rank",
                  flush=True)

    def forward(self, x):
        B, C, T = x.shape
        if self.pad:
            x = F.pad(x, (0, 0, 0, self.pad))
        h = x.transpose(1, 2)
        lvl = h.abs().mean(-1, keepdim=True).clamp(min=1e-6).log()
        v = h.new_ones(B, T, self.rank) / math.sqrt(self.rank)
        for c in range(self.cores):
            seg = h[..., c * self.seg:(c + 1) * self.seg]
            o = torch.einsum("bts,qrs->btqr", seg, self.core[c])
            v = torch.einsum("btqr,btr->btq", o, v)
            v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return self.out(torch.cat([v, lvl], -1)).transpose(1, 2)
