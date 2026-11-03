from __future__ import annotations
"""Schrodinger bridge schedule, sampling and EDM2-style preconditioning."""

import math

import torch

def t_embed(t, dim=64):
    """Sinusoidal embedding of t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(torch.linspace(0, math.log(1000.0), half, device=t.device))
    a = t.view(-1, 1) * freqs.view(1, -1)
    return torch.cat([a.sin(), a.cos()], dim=-1)

def bridge_linear(x0, x1, t, sigma=0.0, noise=True):
    """Brownian bridge on a linear quantity (log-magnitude)."""
    if not torch.is_tensor(t):
        t = torch.full((x0.shape[0],), float(t), device=x0.device, dtype=x0.dtype)
    tt = t.view(-1, *([1] * (x0.dim() - 1)))
    x = (1 - tt) * x0 + tt * x1
    if sigma > 0 and noise:
        x = x + sigma * (tt * (1 - tt)).clamp(min=0).sqrt() * torch.randn_like(x)
    return x

def bridge_phase(p0, p1, t, sigma=0.0, noise=True):
    """Bridge on the UNIT CIRCLE, not on the wrapped value."""
    c = bridge_linear(p0.cos(), p1.cos(), t, sigma, noise)
    s = bridge_linear(p0.sin(), p1.sin(), t, sigma, noise)
    return torch.atan2(s, c)

class SBSchedule:
    """Timesteps and noise level for training and inference."""
    def __init__(self, steps=1, sigma=0.05, t_min=0.0, t_max=1.0):
        self.steps = max(1, int(steps))
        self.sigma = float(sigma)
        self.t_min, self.t_max = float(t_min), float(t_max)

    def sample_t(self, batch, device):
        """Training timesteps."""
        return torch.rand(batch, device=device) * (self.t_max - self.t_min) + self.t_min

    def infer_ts(self):
        """Descending-information schedule: start at t=0 (only the degraded endpoint is known) and walk toward t=1."""
        if self.steps == 1:
            return [0.0]
        return [i / self.steps for i in range(self.steps)]

@torch.no_grad()
def sb_infer(predict, mag0, pha0, sched: SBSchedule):
    """Iteratively restore from the degraded endpoint."""
    mag_t, pha_t = mag0, pha0
    out = None
    ts = sched.infer_ts()
    for i, t in enumerate(ts):
        tv = torch.full((mag0.shape[0],), t, device=mag0.device)
        out = predict(mag_t, pha_t, tv)
        if i + 1 < len(ts):
            t_next = ts[i + 1]
            mag_t = bridge_linear(mag0, out[0], t_next, sched.sigma, noise=False)
            pha_t = bridge_phase(pha0, out[1], t_next, sched.sigma, noise=False)
    return out

def precond(t, sigma=0.05, sigma_d=1.0, lam_clip=(0.1, 10.0)):
    """EDM2-style preconditioning for the bridge (Richter et al., ICASSP 2026)."""
    if not torch.is_tensor(t):
        t = torch.tensor(float(t))
    t = t.clamp(0.0, 1.0)
    var_t = sigma ** 2 * t * (1 - t)
    c_out = (((1 - t) * sigma_d) ** 2 + var_t).clamp(min=1e-8).sqrt()
    c_in = 1.0 / (sigma_d ** 2 + var_t).clamp(min=1e-8).sqrt()
    lam = (1.0 / c_out.pow(2)).clamp(*lam_clip)
    v = lambda a: a.view(-1, *([1] * 2))
    return v(c_in), v(c_out), v(lam)
