"""Critic registry: MRAD and MRPD from AP-BWE (Lu et al., 2024), UHD, SBMD and BiCoD from this work."""
import torch.nn as nn

import bwe_sb.models.critics.highband as CN
import bwe_sb.models.critics.amplitude as CAMP

def build(name, h):
    sr, nf, hop, win = h.hr_sampling_rate, h.n_fft, h.hop_size, h.win_size
    f_cut = h.lr_sampling_rate / 2
    import importlib
    from bwe_sb.models.critics import apbwe as M
    table = {
        "MPD":   (lambda: M.MultiPeriodDiscriminator(), "wav"),
        "MRAD":  (lambda: M.MultiResolutionAmplitudeDiscriminator(), "wav"),
        "MRPD":  (lambda: M.MultiResolutionPhaseDiscriminator(), "wav"),
        "BiCoD": (lambda: CN.BicoherenceDiscriminator(), "wav"),
        "UHD":   (lambda: CAMP.UniversalHarmonicDiscriminator(sr=sr), "wav"),
        "SBMD":  (lambda: CAMP.SubBandMagnitudeDiscriminator(
                      f_cut_bin=int(f_cut / (sr / 2) * (nf // 2))), "wav"),
    }
    if name not in table:
        raise ValueError(f"unknown critic {name!r}; have {sorted(table)}")
    mk, kind = table[name]
    d = mk()
    d.call_kind = kind
    return d

ALL = ["MPD", "MRAD", "MRPD", "BiCoD", "UHD", "SBMD"]
OURS = ["BiCoD", "UHD", "SBMD"]
AMP = ["UHD", "SBMD"]
HF = ["BiCoD"]
