from __future__ import annotations
"""Evaluation metrics; LSD and the anti-wrapped phase distances follow AP-BWE's cal_metrics.py (Lu et al., 2024) so the numbers are comparable to its tables."""

import warnings

import numpy as np
import torch

_WARNED = set()

def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"  [metrics] {msg}", flush=True)

def si_sdr(est, ref, eps=1e-8):
    """Scale-invariant SDR, plus the optimal scale alpha."""
    k = min(est.shape[-1], ref.shape[-1])
    est, ref = est[..., :k], ref[..., :k]
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    p = a * ref
    sdr = 10 * torch.log10((p.pow(2).sum(-1) + eps) / ((est - p).pow(2).sum(-1) + eps))
    return float(sdr.mean()), float(a.mean())

def snr(est, ref, eps=1e-8):
    """Scale-DEPENDENT SNR."""
    k = min(est.shape[-1], ref.shape[-1])
    e, r = est[..., :k], ref[..., :k]
    return float(10 * torch.log10((r.pow(2).sum() + eps) / ((r - e).pow(2).sum() + eps)))

def lsd(est, ref, n_fft=2048, hop=512, sr=None, f_lo=None, f_hi=None):
    """Log-spectral distance, optionally restricted to a band."""
    k = min(est.shape[-1], ref.shape[-1])
    w = torch.hann_window(n_fft, device=est.device)
    A = torch.stft(est[..., :k].reshape(-1, k), n_fft, hop, window=w, return_complex=True).abs()
    B = torch.stft(ref[..., :k].reshape(-1, k), n_fft, hop, window=w, return_complex=True).abs()
    if f_lo or f_hi:
        nb = A.shape[-2]
        lo = int((f_lo or 0) / (sr / 2) * nb)
        hi = int((f_hi or sr / 2) / (sr / 2) * nb)
        A, B = A[..., lo:hi, :], B[..., lo:hi, :]
    la = torch.log10(A.pow(2).clamp(min=1e-10))
    lb = torch.log10(B.pow(2).clamp(min=1e-10))
    return float((la - lb).pow(2).mean(-2).sqrt().mean())

def _antiwrap(x):
    """Anti-wrapping function from AP-BWE: maps a phase difference into (-pi, pi]."""
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def apd(est, ref, n_fft=1024, hop=80, win=320):
    """Anti-wrapped phase distances: APD-IP, APD-GD, APD-IAF."""
    k = min(est.shape[-1], ref.shape[-1])
    w = torch.hann_window(win, device=est.device)
    def ph(x):
        X = torch.stft(x[..., :k].reshape(-1, k), n_fft, hop, win, window=w,
                       return_complex=True)
        return torch.angle(X)
    pe, pr = ph(est), ph(ref)
    ip = _antiwrap(pe - pr).mean()
    gd = _antiwrap(torch.diff(pe, dim=-2) - torch.diff(pr, dim=-2)).mean()
    iaf = _antiwrap(torch.diff(pe, dim=-1) - torch.diff(pr, dim=-1)).mean()
    return float(ip), float(gd), float(iaf)

def awpd(est, ref, n_fft=1024, hop=80, win=320):
    """AP-BWE's cal_apd reduction exactly: RMS over frequency per frame, then mean."""
    k = min(est.shape[-1], ref.shape[-1])
    w = torch.hann_window(win, device=est.device)
    def ph(x):
        X = torch.stft(x[..., :k].reshape(-1, k), n_fft, hop, win, window=w,
                       return_complex=True)
        return torch.angle(X)
    pe, pr = ph(est), ph(ref)
    def rms_f(d):
        return d.square().mean(dim=-2).sqrt().mean()
    ip = rms_f(_antiwrap(pe - pr))
    gd = rms_f(_antiwrap(torch.diff(pe, dim=-2) - torch.diff(pr, dim=-2)))
    iaf = rms_f(_antiwrap(torch.diff(pe, dim=-1) - torch.diff(pr, dim=-1)))
    return float(ip), float(gd), float(iaf)

LSD_BAND_EDGES = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

def lsd_bands(est, ref, sr=16000, edges=None):
    """LSD within each frequency band."""
    edges = edges or LSD_BAND_EDGES
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > sr / 2:
            break
        out[f"lsd_b{lo}_{hi}"] = lsd(est, ref, sr=sr, f_lo=lo, f_hi=hi)
    return out

def _band_mag(x, sr, f_lo, f_hi, n_fft=1024, hop=256):
    w = torch.hann_window(n_fft, device=x.device)
    X = torch.stft(x.reshape(1, -1), n_fft, hop, window=w, return_complex=True).abs()[0]
    nb = X.shape[0]
    lo = int((f_lo or 0) / (sr / 2) * nb)
    hi = min(nb, int((f_hi or sr / 2) / (sr / 2) * nb))
    return X[lo:hi, :]

def d_spectral_flux(est, ref, sr, f_lo, f_hi=None):
    """BS.1387-inspired: how different is frame-to-frame spectral change in the HF band."""
    A, B = _band_mag(est, sr, f_lo, f_hi), _band_mag(ref, sr, f_lo, f_hi)
    t = min(A.shape[-1], B.shape[-1])
    fa = torch.diff(A[..., :t], dim=-1).abs().mean(0)
    fb = torch.diff(B[..., :t], dim=-1).abs().mean(0)
    den = fb.abs().mean().clamp(min=1e-8)
    return float((fa - fb).abs().mean() / den)

def _k_weight(x, sr):
    """BS.1770 K-weighting: a high-shelf followed by a high-pass."""
    import torchaudio.functional as aF
    y = aF.treble_biquad(x, sr, gain=4.0, central_freq=1500.0, Q=0.707)
    return aF.highpass_biquad(y, sr, 38.0, Q=0.5)

def d_loudness(est, ref, sr, block_ms=400.0):
    """BS.1770-inspired: error in SHORT-TERM K-weighted loudness."""
    try:
        e, r = _k_weight(est, sr), _k_weight(ref, sr)
    except Exception:
        e, r = est, ref
    n = max(1, int(sr * block_ms / 1000.0))
    def loud(x):
        k = (x.shape[-1] // n) * n
        if k == 0:
            return torch.zeros(1, device=x.device)
        b = x[..., :k].reshape(-1, n)
        return -0.691 + 10 * torch.log10(b.pow(2).mean(-1).clamp(min=1e-12))
    le, lr = loud(e.reshape(1, -1)), loud(r.reshape(1, -1))
    t = min(le.shape[0], lr.shape[0])
    return float((le[:t] - lr[:t]).abs().mean())

def d_missing(est, ref, sr, f_lo, f_hi=None):
    """P.862-inspired asymmetric disturbance: only HF energy present in the reference and ABSENT from the estimate..."""
    A, B = _band_mag(est, sr, f_lo, f_hi), _band_mag(ref, sr, f_lo, f_hi)
    t = min(A.shape[-1], B.shape[-1])
    la = torch.log10(A[..., :t].pow(2).clamp(min=1e-10))
    lb = torch.log10(B[..., :t].pow(2).clamp(min=1e-10))
    return float((lb - la).clamp(min=0).mean())

BASE_KEYS = ("lsd", "lsd_hf", "si_sdr", "si_snr", "snr", "alpha",
             "apd_ip", "apd_gd", "apd_iaf", "awpd_ip", "awpd_gd", "awpd_iaf")
ITU_KEYS = ("d_sf", "d_loud", "d_miss", "d_sf_top", "d_miss_top")
PERCEPTUAL_KEYS = ("pesq", "stoi", "estoi", "mos", "nisqa_mos")

def metric_keys(sr=16000, want_perceptual=True, edges=None):
    """The float keys evaluate() returns, computed WITHOUT running it."""
    edges = edges or LSD_BAND_EDGES
    bands = [f"lsd_b{lo}_{hi}" for lo, hi in zip(edges[:-1], edges[1:]) if hi <= sr / 2]
    return list(BASE_KEYS) + bands + list(ITU_KEYS) + \
        (list(PERCEPTUAL_KEYS) if want_perceptual else [])

def itu_metrics(est, ref, sr, f_cut):
    """The three diagnostics, in the generated band and in the top octave separately."""
    e, r = est.reshape(-1), ref.reshape(-1)
    top = max(f_cut, sr / 4)
    return dict(d_sf=d_spectral_flux(e, r, sr, f_cut),
                d_loud=d_loudness(e, r, sr),
                d_miss=d_missing(e, r, sr, f_cut),
                d_sf_top=d_spectral_flux(e, r, sr, top),
                d_miss_top=d_missing(e, r, sr, top))

def pesq_wb(est, ref, sr=16000):
    """ITU-T P.862.2 wideband PESQ."""
    try:
        from pesq import pesq as _pesq
    except Exception:
        _warn_once("pesq", "pesq not installed -> pesq=NaN (pip install pesq)")
        return float("nan")
    try:
        k = min(est.shape[-1], ref.shape[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(_pesq(sr, ref[..., :k].squeeze().cpu().numpy(),
                               est[..., :k].squeeze().cpu().numpy(), "wb"))
    except Exception as e:
        _warn_once("pesq_run", f"pesq failed on a file ({type(e).__name__}) -> skipped")
        return float("nan")

def stoi_(est, ref, sr=16000, extended=False):
    try:
        from pystoi import stoi as _stoi
    except Exception:
        _warn_once("stoi", "pystoi not installed -> stoi=NaN (pip install pystoi)")
        return float("nan")
    try:
        k = min(est.shape[-1], ref.shape[-1])
        return float(_stoi(ref[..., :k].squeeze().cpu().numpy(),
                           est[..., :k].squeeze().cpu().numpy(), sr, extended=extended))
    except Exception as e:
        _warn_once("stoi_run", f"pystoi failed on a file ({type(e).__name__}) -> skipped")
        return float("nan")

_MOS = {"kind": None, "model": None}

def _init_mos(dev):
    if _MOS["kind"] is not None:
        return
    try:
        from nisqa.NISQA_model import nisqaModel
        _MOS.update(kind="nisqa", model=nisqaModel)
        print("  [metrics] MOS backend: NISQA", flush=True)
        return
    except Exception:
        pass
    try:
        from torchaudio.pipelines import SQUIM_SUBJECTIVE
        m = SQUIM_SUBJECTIVE.get_model().to(dev).eval()
        _MOS.update(kind="squim", model=m)
        print("  [metrics] MOS backend: torchaudio SQUIM_SUBJECTIVE "
              "(NISQA unavailable; different scale from NISQA-MOS)", flush=True)
        return
    except Exception as e:
        _MOS.update(kind="none", model=None)
        _warn_once("mos", f"no MOS backend ({type(e).__name__}) -> mos=NaN")

def mos(est, ref, sr=16000, dev="cpu"):
    """Non-intrusive MOS."""
    _init_mos(dev)
    if _MOS["kind"] in (None, "none"):
        return float("nan"), "none"
    if _MOS["kind"] == "squim":
        try:
            k = min(est.shape[-1], ref.shape[-1])
            e = est[..., :k].reshape(1, -1).to(dev)
            r = ref[..., :k].reshape(1, -1).to(dev)
            with torch.no_grad():
                return float(_MOS["model"](e, r).mean()), "squim"
        except Exception as e:
            _warn_once("squim_run", f"SQUIM failed ({type(e).__name__}) -> mos=NaN")
            return float("nan"), "squim"
    return float("nan"), _MOS["kind"]

def evaluate(est, ref, sr, f_cut, dev="cpu", want_perceptual=True):
    """Every metric for one utterance."""
    est = est.reshape(1, -1).float()
    ref = ref.reshape(1, -1).float()
    sd, alpha = si_sdr(est, ref)
    ip, gd, iaf = apd(est, ref)
    wip, wgd, wiaf = awpd(est, ref)
    out = dict(lsd=lsd(est, ref), lsd_hf=lsd(est, ref, sr=sr, f_lo=f_cut),
               si_sdr=sd, si_snr=sd, snr=snr(est, ref), alpha=alpha,
               apd_ip=ip, apd_gd=gd, apd_iaf=iaf,
               awpd_ip=wip, awpd_gd=wgd, awpd_iaf=wiaf)
    out.update(lsd_bands(est, ref, sr))
    out.update(itu_metrics(est, ref, sr, f_cut))
    if want_perceptual:
        out["pesq"] = pesq_wb(est, ref, sr)
        out["stoi"] = stoi_(est, ref, sr)
        out["estoi"] = stoi_(est, ref, sr, extended=True)
        m, kind = mos(est, ref, sr, dev)
        out["mos"] = m
        out["mos_backend"] = kind
        out["nisqa_mos"] = float("nan")
    got = {k for k, v in out.items() if isinstance(v, float)}
    want = set(metric_keys(sr, want_perceptual))
    if got != want:
        raise AssertionError(
            f"metric schema drift: missing {sorted(want - got)}, "
            f"unexpected {sorted(got - want)} - update metric_keys()")
    return out
