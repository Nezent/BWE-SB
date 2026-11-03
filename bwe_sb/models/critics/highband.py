"""Bicoherence critic: is the synthesised band phase-locked to the given one; the separable convolutions are from the CIS-BWE lab code (Tamiti et al., ACL 2026)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_ch, bias=bias
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_ch, bias=bias
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

from bwe_sb.losses.phase import hpc_map, frame_f0

def _head2d(ch=(1, 16, 32, 32, 1), k=5):
    layers, feats = [], []
    for i in range(len(ch) - 1):
        s = 2 if i < len(ch) - 2 else 1
        layers.append(SeparableConv2d(ch[i], ch[i + 1], k if i < len(ch) - 2 else 3,
                                      stride=s, padding=(k if i < len(ch) - 2 else 3) // 2))
    return nn.ModuleList(layers)

class _Scorer2d(nn.Module):
    """Shared 2D critic head: scores a (B,1,H,W) feature map, returning logits + fmaps."""
    def __init__(self, ch=(1, 16, 32, 32, 1)):
        super().__init__()
        self.layers = _head2d(ch)

    def forward(self, x):
        fmap = []
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class BicoherenceDiscriminator(nn.Module):
    """BiCoD -- quadratic phase coupling between low and high frequency."""
    def __init__(self, n_fft=256, hop=128, nbands=16):
        super().__init__()
        self.n_fft, self.hop, self.nbands = n_fft, hop, nbands
        self.net = _Scorer2d()

    def _bicoh(self, x, eps=1e-8):
        w = torch.hann_window(self.n_fft, device=x.device)
        X = torch.stft(x.squeeze(1), self.n_fft, self.hop, self.n_fft, w,
                       center=True, return_complex=True)
        B, Fb, T = X.shape
        nb = min(self.nbands, Fb // 2)
        idx = torch.linspace(1, Fb // 2 - 1, nb, device=x.device).long()
        X1 = X[:, idx, :]
        s = idx.view(-1, 1) + idx.view(1, -1)
        s = s.clamp(max=Fb - 1)
        Xs = X[:, s.reshape(-1), :].reshape(B, nb, nb, T)
        tri = X1.unsqueeze(2) * X1.unsqueeze(1) * Xs.conj()
        num = tri.mean(-1).abs()
        den = ((X1.unsqueeze(2) * X1.unsqueeze(1)).abs().pow(2).mean(-1).sqrt()
               * Xs.abs().pow(2).mean(-1).sqrt() + eps)
        return (num / den).unsqueeze(1)

    def forward(self, y, y_hat):
        r, fr = self.net(self._bicoh(y))
        g, fg = self.net(self._bicoh(y_hat))
        return [r], [g], [fr], [fg]
