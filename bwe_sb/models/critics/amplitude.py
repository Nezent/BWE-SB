from __future__ import annotations
"""Amplitude critics: universal harmonic, multi-tier magnitude and sub-band magnitude discriminators."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def _stft_mag(x, n_fft, hop, win):
    w = torch.hann_window(win, device=x.device)
    X = torch.stft(x.squeeze(1), n_fft, hop, win, window=w, return_complex=True)
    return X.abs()

class _Conv2dTower(nn.Module):
    """Shared discriminator body over a (frequency, time) magnitude image."""
    def __init__(self, ch=32, n_in=1):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Conv2d(n_in, ch, (3, 9), padding=(1, 4)),
            nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(ch, ch, (3, 3), padding=(1, 1)),
        ])
        self.out = nn.Conv2d(ch, 1, (3, 3), padding=(1, 1))
        self.act = nn.LeakyReLU(0.1)

    def forward(self, m):
        f = []
        for layer in self.net:
            m = self.act(layer(m))
            f.append(m)
        return self.out(m), f

class UniversalHarmonicDiscriminator(nn.Module):
    """Magnitude discriminator on a LOG-frequency (pitch-aligned) axis."""
    call_kind = "wav"

    def __init__(self, sr=16000, n_fft=1024, hop=256, win=1024, n_log=128,
                 f_min=50.0, ch=32):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        n_bins = n_fft // 2 + 1
        f_lin = torch.linspace(0, sr / 2, n_bins)
        f_log = torch.logspace(math.log10(f_min), math.log10(sr / 2), n_log)
        W = torch.zeros(n_log, n_bins)
        for i, f in enumerate(f_log):
            j = torch.searchsorted(f_lin, f).clamp(1, n_bins - 1)
            lo, hi = f_lin[j - 1], f_lin[j]
            a = ((f - lo) / (hi - lo).clamp(min=1e-9)).clamp(0, 1)
            W[i, j - 1], W[i, j] = 1 - a, a
        self.register_buffer("warp", W)
        self.tower = _Conv2dTower(ch, n_in=1)

    def _one(self, y):
        m = _stft_mag(y, self.n_fft, self.hop, self.win)
        m = torch.log(m.clamp(min=1e-5))
        m = torch.einsum("lf,bft->blt", self.warp, m).unsqueeze(1)
        return self.tower(m)

    def forward(self, y, y_hat):
        dr, fr = self._one(y)
        dg, fg = self._one(y_hat)
        return dr, dg, [fr], [fg]


class SubBandMagnitudeDiscriminator(nn.Module):
    """Local, per-band magnitude verdicts (FA-GAN's multi-band local idea)."""
    call_kind = "wav"

    def __init__(self, n_fft=1024, hop=256, win=1024, n_bands=4, ch=20, f_cut_bin=None):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.n_bands = n_bands
        self.f_cut_bin = f_cut_bin
        self.towers = nn.ModuleList([_Conv2dTower(ch, n_in=1) for _ in range(n_bands)])

    def _one(self, y):
        m = torch.log(_stft_mag(y, self.n_fft, self.hop, self.win).clamp(min=1e-5))
        Fb = m.shape[1]
        lo = self.f_cut_bin or 0
        edges = torch.linspace(lo, Fb, self.n_bands + 1).long()
        ds, fs = [], []
        for b, tower in enumerate(self.towers):
            seg = m[:, edges[b]:edges[b + 1], :].unsqueeze(1)
            if seg.shape[2] < 4:
                continue
            d, f = tower(seg)
            ds.append(d); fs.append(f)
        return ds, fs

    def forward(self, y, y_hat):
        dr, fr = self._one(y)
        dg, fg = self._one(y_hat)
        return torch.cat([d.flatten(1) for d in dr], 1), \
               torch.cat([d.flatten(1) for d in dg], 1), fr, fg
