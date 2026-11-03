from __future__ import annotations
"""Degradation recipes and banks for training and evaluation; the sinc bank reproduces the single operator AP-BWE, CIS-BWE and UDSS-BWE train on."""

import math
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF

DESC = ["cutoff", "mulaw", "bits", "reverb", "clip", "snr", "tele", "ploss"]
N_DESC = len(DESC)

BASE = 2000

def cutoffs_seen(base=None):
    b = base or BASE
    return [int(b * f) for f in (0.5, 1.0, 1.5, 2.0)]

def cutoffs_unseen(base=None):
    b = base or BASE
    return [int(b * f) for f in (0.75, 1.25)]

CUTOFFS_SEEN = cutoffs_seen(2000)
CUTOFFS_UNSEEN = cutoffs_unseen(2000)

SNRS_SEEN = [0, 5, 10, 15, 20]
SNR_UNSEEN = -5

SNRS_AURORA = [-10, -5, 0, 5, 10]

def _mulaw(x, mu=255.0):
    """G.711 mu-law companding: a real telephony codec, not an approximation of one."""
    y = torch.sign(x) * torch.log1p(mu * x.abs()) / math.log1p(mu)
    q = torch.round((y + 1) * 127.5) / 127.5 - 1
    return torch.sign(q) * ((1 + mu) ** q.abs() - 1) / mu

def _bitcrush(x, bits):
    s = 2 ** (bits - 1)
    return torch.round(x.clamp(-1, 1) * s) / s

def _reverb(x, sr, t60, mix=0.3):
    """Synthetic RIR: exponentially decaying noise."""
    n = int(0.25 * sr)
    t = torch.arange(n, device=x.device) / sr
    rir = torch.randn(n, device=x.device) * torch.exp(-6.9 * t / max(t60, 1e-3))
    rir[0] = 1.0
    rir = rir / rir.norm().clamp(min=1e-8)
    wet = F.conv1d(x.unsqueeze(1), rir.flip(0).view(1, 1, -1), padding=n - 1)[..., :x.shape[-1]]
    return (1 - mix) * x + mix * wet.squeeze(1)

def _softclip(x, drive):
    return torch.tanh(x * drive) / math.tanh(drive) if drive > 1 else x

def _mix_snr(x, snr_db, coloured=False):
    """Additive noise at a given SNR, computed PER UTTERANCE."""
    n = torch.randn_like(x)
    if coloured:
        n = torch.cumsum(n, -1)
        n = n - n.mean(-1, keepdim=True)
    rs = x.pow(2).mean(-1, keepdim=True).clamp(min=1e-12).sqrt()
    rn = n.pow(2).mean(-1, keepdim=True).clamp(min=1e-12).sqrt()
    return x + n * (rs / (10 ** (snr_db / 20.0) * rn))

def _ssn(x):
    """Speech-shaped noise: white noise given the long-term spectrum of this utterance."""
    n = torch.randn_like(x)
    X = torch.fft.rfft(x, dim=-1)
    N = torch.fft.rfft(n, dim=-1)
    mag = X.abs().mean(0, keepdim=True)
    mag = mag / mag.max().clamp(min=1e-12)
    return torch.fft.irfft(N * mag, n=x.shape[-1], dim=-1)

def _telephone(x, sr):
    """ITU-T narrowband telephone band, 300-3400 Hz."""
    x = aF.highpass_biquad(x, sr, 300.0)
    return aF.lowpass_biquad(x, sr, 3400.0)

def _packet_loss(x, sr, rate, ms=20.0):
    """Drop whole packets, as a VoIP channel does."""
    n = max(1, int(sr * ms / 1000.0))
    nb = math.ceil(x.shape[-1] / n)
    keep = (torch.rand(x.shape[0], nb, device=x.device) > rate).float()
    keep = keep.repeat_interleave(n, dim=-1)[..., :x.shape[-1]]
    return x * keep

def degrade(hr, sr, recipe: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply one degradation recipe to a clean waveform."""
    x = hr
    cut = int(recipe.get("cutoff", BASE))
    x = aF.resample(aF.resample(x, sr, cut), cut, sr)[..., :hr.shape[-1]]
    if x.shape[-1] < hr.shape[-1]:
        x = F.pad(x, (0, hr.shape[-1] - x.shape[-1]))

    if recipe.get("mulaw"):
        x = _mulaw(x)
    bits = int(recipe.get("bits", 0) or 0)
    if bits:
        x = _bitcrush(x, bits)
    t60 = float(recipe.get("reverb", 0.0) or 0.0)
    if t60:
        x = _reverb(x, sr, t60)
    drive = float(recipe.get("clip", 0.0) or 0.0)
    if drive:
        x = _softclip(x, drive)
    if recipe.get("tele"):
        x = _telephone(x, sr)
    ploss = float(recipe.get("ploss", 0.0) or 0.0)
    if ploss:
        x = _packet_loss(x, sr, ploss)
    snr_db = recipe.get("snr", None)
    if snr_db is not None:
        noise = str(recipe.get("noise", "awgn"))
        if noise == "ssn":
            n = _ssn(x)
            rs = x.pow(2).mean(-1, keepdim=True).clamp(min=1e-12).sqrt()
            rn = n.pow(2).mean(-1, keepdim=True).clamp(min=1e-12).sqrt()
            x = x + n * (rs / (10 ** (float(snr_db) / 20.0) * rn))
        else:
            x = _mix_snr(x, float(snr_db), coloured=bool(recipe.get("coloured", False)))

    d = torch.tensor([
        cut / (sr / 2.0),
        1.0 if recipe.get("mulaw") else 0.0,
        (bits / 16.0) if bits else 0.0,
        min(t60, 1.0),
        min(drive / 10.0, 1.0) if drive else 0.0,
        ((float(snr_db) + 10) / 30.0) if snr_db is not None else 1.0,
        1.0 if recipe.get("tele") else 0.0,
        min(ploss / 0.2, 1.0) if ploss else 0.0,
    ], device=hr.device, dtype=torch.float32)
    return x, d.unsqueeze(0).expand(hr.shape[0], -1).contiguous()

def bank(kind: str, base: int = None) -> List[Dict]:
    base = int(base or BASE)
    seen, unseen = cutoffs_seen(base), cutoffs_unseen(base)
    """Named degradation sets. 'sinc' is the single idealised operator every prior paper
    uses; 'mixed' is the training distribution; 'unseen' holds out whole degradation TYPES
    and cutoffs, which is what cross-domain generalisation has to be measured on."""
    if kind == "sinc":
        return [dict(cutoff=base)]
    if kind == "noise":
        return [dict(cutoff=base, snr=s) for s in SNRS_SEEN]
    if kind == "mixed":
        r = []
        for c in seen:
            r += [dict(cutoff=c),
                  dict(cutoff=c, mulaw=True),
                  dict(cutoff=c, bits=8),
                  dict(cutoff=c, snr=random.choice(SNRS_SEEN)),
                  dict(cutoff=c, snr=5, coloured=True),
                  dict(cutoff=c, mulaw=True, snr=10)]
        return r
    if kind == "aurora":
        return [dict(cutoff=base, snr=s, noise=nz)
                for nz in ("awgn", "ssn") for s in SNRS_AURORA]
    if kind.startswith("R") and kind[1:].isdigit():
        tier = int(kind[1:])
        r = [dict(cutoff=base)]
        if tier >= 1:
            r += [dict(cutoff=base, bits=8), dict(cutoff=base, bits=12)]
        if tier >= 2:
            r += [dict(cutoff=base, mulaw=True)]
        if tier >= 3:
            r += [dict(cutoff=base, mulaw=True, bits=8),
                  dict(cutoff=base, bits=4),
                  dict(cutoff=base, mulaw=True, clip=4.0)]
        if tier >= 4:
            r += [dict(cutoff=base, tele=True),
                  dict(cutoff=base, tele=True, mulaw=True),
                  dict(cutoff=base, ploss=0.05),
                  dict(cutoff=base, tele=True, mulaw=True, ploss=0.1)]
        return r
    if kind == "channel":
        out = []
        for t in range(5):
            for rr in bank(f"R{t}", base):
                d = dict(rr)
                d["tier"] = f"R{t}"
                out.append(d)
        return out
    if kind == "unseen":
        r = []
        for c in unseen:
            r += [dict(cutoff=c), dict(cutoff=c, mulaw=True)]
        for c in seen[:2]:
            r += [dict(cutoff=c, reverb=0.3),
                  dict(cutoff=c, clip=4.0),
                  dict(cutoff=c, bits=4),
                  dict(cutoff=c, snr=SNR_UNSEEN)]
        return r
    raise ValueError(kind)

def sample_recipe(recipes: List[Dict], rng=random) -> Dict:
    return dict(rng.choice(recipes))

class DegradationEncoder(nn.Module):
    """BLIND degradation encoder: infers the degradation from the input alone."""
    def __init__(self, n_bins, emb=64, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_bins, hidden, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2), nn.GELU(),
        )
        self.head_emb = nn.Linear(hidden, emb)
        self.head_desc = nn.Linear(hidden, N_DESC)
        self.emb_dim = emb

    def forward(self, mag_nb):
        h = self.net(mag_nb).mean(-1)
        return self.head_emb(h), torch.sigmoid(self.head_desc(h))
