"""Phase losses; the anti-wrapping function is AP-BWE's (Lu et al., 2024), the harmonic phase coherence loss is ours."""
import torch
import torch.nn.functional as F

def antiwrap(x):
    return x - torch.round(x / (2 * torch.pi)) * 2 * torch.pi

@torch.no_grad()
def frame_f0(wav, sr, hop, win, fmin=80.0, fmax=330.0, vthr=0.3):
    """Per-frame f0 (Hz) and voicing strength from the narrowband waveform."""
    B = wav.shape[0]
    frames = wav.unfold(-1, win, hop)
    frames = frames - frames.mean(-1, keepdim=True)
    n = frames.shape[-1]
    spec = torch.fft.rfft(frames, n=2 * n)
    ac = torch.fft.irfft(spec * spec.conj(), n=2 * n)[..., :n]
    ac0 = ac[..., :1].clamp(min=1e-8)
    lo, hi = int(sr / fmax), min(int(sr / fmin), n - 1)
    if hi <= lo + 1:
        z = torch.zeros(ac.shape[:2], device=wav.device)
        return z, z
    seg = ac[..., lo:hi] / ac0
    peak, idx = seg.max(-1)
    f0 = sr / (idx + lo).clamp(min=1).float()
    voiced = (peak.clamp(0, 1) > vthr).float() * peak.clamp(0, 1)
    return f0, voiced

def hpc_loss(pha, mag, f0, voiced, sr, n_fft, hop, f_cut, max_h=40, eps=1e-8):
    """Harmonic phase coherence loss on a predicted phase spectrogram."""
    B, Fb, T = pha.shape
    T = min(T, f0.shape[-1])
    if T < 2:
        return pha.sum() * 0.0
    pha, mag = pha[..., :T], mag[..., :T]
    f0, voiced = f0[..., :T], voiced[..., :T]

    df = sr / n_fft
    dphi = antiwrap(pha[..., 1:] - pha[..., :-1])
    f0_ = f0[..., :-1]; v = voiced[..., :-1]

    k1 = (f0_ / df).round().long().clamp(1, Fb - 1)
    d1 = torch.gather(dphi, 1, k1.unsqueeze(1)).squeeze(1)

    h = torch.arange(2, max_h + 1, device=pha.device).view(1, -1, 1).float()
    kh = (h * f0_.unsqueeze(1) / df).round().long()
    valid = (kh < Fb) & (kh * df > f_cut)
    kh = kh.clamp(0, Fb - 1)

    dh = torch.gather(dphi, 1, kh)
    resid = antiwrap(dh - h * d1.unsqueeze(1)).abs()

    w = torch.gather(mag, 1, kh).exp()
    w = w * valid.float() * v.unsqueeze(1)
    return (resid * w).sum() / (w.sum() + eps)

def hpc_map(pha, mag, f0, voiced, sr, n_fft, hop, f_cut, max_h=40):
    """Same quantity as a (B, 1, H, T-1) map, for a discriminator to score."""
    B, Fb, T = pha.shape
    T = min(T, f0.shape[-1])
    pha = pha[..., :T]; f0_ = f0[..., :T][..., :-1]; v = voiced[..., :T][..., :-1]
    df = sr / n_fft
    dphi = antiwrap(pha[..., 1:] - pha[..., :-1])
    k1 = (f0_ / df).round().long().clamp(1, Fb - 1)
    d1 = torch.gather(dphi, 1, k1.unsqueeze(1)).squeeze(1)
    h = torch.arange(2, max_h + 1, device=pha.device).view(1, -1, 1).float()
    kh = (h * f0_.unsqueeze(1) / df).round().long()
    valid = ((kh < Fb) & (kh * df > f_cut)).float()
    dh = torch.gather(dphi, 1, kh.clamp(0, Fb - 1))
    resid = antiwrap(dh - h * d1.unsqueeze(1))
    return (resid * valid * v.unsqueeze(1)).unsqueeze(1)
