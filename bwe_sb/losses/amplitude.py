"""Amplitude losses: spectral convergence, band energy in dB and multi-resolution log-magnitude."""
import math

import torch
import torch.nn.functional as F

MRSTFT_CFGS = ((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048))

BAND_EDGES = (0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000)

def _stft_mag(x, n_fft, hop, win):
    """(B, T) -> (B, F, frames) magnitude."""
    w = torch.hann_window(win, device=x.device, dtype=x.dtype)
    return torch.stft(x, n_fft, hop, win, window=w, return_complex=True).abs()

def _soft_cap(x, cap):
    return cap * torch.tanh(x / cap)

def mrstft_loss(est, ref, cfgs=MRSTFT_CFGS, eps=1e-7, rms_floor=1e-3,
                sc_max=10.0, lm_max=5.0):
    """Multi-resolution STFT loss, returned as its two terms separately."""
    est = est.reshape(-1, est.shape[-1])
    ref = ref.reshape(-1, ref.shape[-1])
    k = min(est.shape[-1], ref.shape[-1])
    est, ref = est[..., :k], ref[..., :k]
    sc = est.new_zeros(())
    lm = est.new_zeros(())
    for n_fft, hop, win in cfgs:
        if k < win:
            continue
        A = _stft_mag(est, n_fft, hop, win)
        B = _stft_mag(ref, n_fft, hop, win)
        den = torch.linalg.norm(B).clamp(min=rms_floor * math.sqrt(B.numel()))
        sc = sc + _soft_cap(torch.linalg.norm(B - A) / den, sc_max)
        lm = lm + _soft_cap((A.clamp(min=eps).log()
                             - B.clamp(min=eps).log()).abs(), lm_max).mean()
    n = max(1, len(cfgs))
    return sc / n, lm / n

def band_energy_loss(est, ref, sr, n_fft=2048, hop=512, edges=BAND_EDGES,
                     f_lo=None, eps=1e-10, db_max=30.0):
    """L1 between per-band energies, in dB."""
    est = est.reshape(-1, est.shape[-1])
    ref = ref.reshape(-1, ref.shape[-1])
    k = min(est.shape[-1], ref.shape[-1])
    if k < n_fft:
        return est.new_zeros(())
    A = _stft_mag(est[..., :k], n_fft, hop, n_fft).pow(2)
    B = _stft_mag(ref[..., :k], n_fft, hop, n_fft).pow(2)
    nb = A.shape[-2]
    loss = est.new_zeros(())
    used = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > sr / 2:
            break
        if f_lo is not None and hi <= f_lo:
            continue
        i0 = int(lo / (sr / 2) * nb)
        i1 = max(i0 + 1, int(hi / (sr / 2) * nb))
        ea = A[..., i0:i1, :].mean(-2).clamp(min=eps)
        eb = B[..., i0:i1, :].mean(-2).clamp(min=eps)
        loss = loss + _soft_cap(
            (10 * ea.log10() - 10 * eb.log10()).abs(), db_max).mean()
        used += 1
    return loss / max(1, used)
