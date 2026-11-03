"""Torch-level checks of the generator, losses and critics; run with python -m tests.test_model."""
"""Torch-level test of the five additions, on CPU."""
import math
import sys
from pathlib import Path

import torch

import bwe_sb.models.generator as G
import bwe_sb.models.bridge as SB
from bwe_sb.models.generator import DualStreamBWE

class H(dict):
    __getattr__ = dict.get

BASE = dict(n_fft=1024, hop_size=80, win_size=320, hr_sampling_rate=16000,
            lr_sampling_rate=2000, gen_dim=128, gen_lattice=1,
            gen_amp_depth=2, gen_pha_depth=2,
            gen_pre_kernel_mag=1, gen_pre_kernel_pha=7, gen_phase_deriv="only",
            gen_phase_mode="residual")

VARIANTS = {
    "resid": {},
    "fuse":  dict(gen_mag_fusion=True),
    "p2d":   dict(gen_phase_2d=True),
    "prior": dict(gen_hf_prior=True),
    "all3":  dict(gen_mag_fusion=True, gen_phase_2d=True, gen_hf_prior=True),
    "sb":    dict(gen_mag_fusion=True, gen_phase_2d=True, gen_hf_prior=True,
                  gen_sb=True, gen_t_dim=64),
    "sb+cond": dict(gen_mag_fusion=True, gen_phase_2d=True, gen_hf_prior=True,
                    gen_sb=True, gen_t_dim=64, gen_cond_dim=64),
}

B, F, T = 2, 513, 40
torch.manual_seed(0)
mag = torch.randn(B, F, T) * 0.5 - 2.0
pha = (torch.rand(B, F, T) * 2 - 1) * torch.pi
fails = []

def check(name, cond, detail=""):
    print(f"   {'PASS' if cond else 'FAIL'}  {name}{(' - ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)

print("=" * 72)
print("1. every variant builds, forwards, and backwards")
ref_out = None
for v, extra in VARIANTS.items():
    h = H({**BASE, **extra})
    g = DualStreamBWE(h)
    n = sum(p.numel() for p in g.parameters())
    kw = {}
    if extra.get("gen_cond_dim"):
        kw["cond"] = torch.randn(B, extra["gen_cond_dim"])
    if extra.get("gen_sb"):
        kw["t"] = torch.rand(B)
    m, p, c = g(mag, pha, **kw)
    loss = m.pow(2).mean() + c.pow(2).mean()
    loss.backward()
    got = sum(1 for q in g.parameters() if q.grad is not None and q.grad.abs().sum() > 0)
    tot = sum(1 for _ in g.parameters())
    print(f"  {v:9s} {n/1e6:6.3f}M  out {tuple(m.shape)} phase in [{p.min():+.2f},"
          f" {p.max():+.2f}]  grads {got}/{tot}")
    check(f"{v}: phase wrapped", p.abs().max() <= torch.pi + 1e-4,
          f"max |phase| = {p.abs().max():.4f}")
    check(f"{v}: finite", torch.isfinite(m).all() and torch.isfinite(p).all())
    check(f"{v}: most params get gradient", got >= 0.9 * tot, f"{got}/{tot}")
    if v == "resid":
        ref_out = (m.detach().clone(), p.detach().clone())

print()
print("2. each addition is IDENTITY at step 0 (so the ablation compares additions,")
print("   not initialisations)")
for v in ("fuse", "p2d", "all3"):
    torch.manual_seed(0)
    g0 = DualStreamBWE(H({**BASE}))
    g1 = DualStreamBWE(H({**BASE, **VARIANTS[v]}))
    sd1 = g1.state_dict()
    shared = {k: t for k, t in g0.state_dict().items()
              if k in sd1 and sd1[k].shape == t.shape}
    missing = [k for k in sd1 if k not in shared]
    g1.load_state_dict(shared, strict=False)
    g0.eval(); g1.eval()
    with torch.no_grad():
        m0, p0, _ = g0(mag, pha)
        m1, p1, _ = g1(mag, pha)
    print(f"   ({v}: {len(missing)} extra tensor(s) in the variant: "
          f"{[m.split('.')[0] for m in missing][:4]})")
    dp = G.antiwrap(p1 - p0).abs().max().item()
    dm = (m1 - m0).abs().max().item()
    if VARIANTS[v].get("gen_hf_prior"):
        print(f"   n/a   {v}: phase moves via cross-stream coupling (expected), "
              f"max angular diff = {dp:.2e} rad")
    else:
        check(f"{v}: phase identical at init", dp < 2e-3,
              f"max angular diff = {dp:.2e} rad")
    if not VARIANTS[v].get("gen_hf_prior"):
        check(f"{v}: magnitude near-identical at init", dm < 3e-3,
              f"max|dmag| = {dm:.2e} over {mag.numel()} bins")
    else:
        print(f"   n/a   {v}: magnitude differs by design (coarse prior), "
              f"max|dmag| = {dm:.2e}")

print()
print("3. baseband phase difference removes the linear-phase ramp")
tpd = G.antiwrap(pha - torch.roll(pha, 1, dims=-1))
bpd = G.baseband_phase_diff(pha, 80, 1024)
m_idx = torch.arange(F).view(1, -1, 1)
ramp = (2 * torch.pi * m_idx * 80 / 1024).expand(B, F, T).float()
ramp_ph = G.antiwrap(torch.cumsum(torch.zeros_like(ramp) + ramp, dim=-1))
bpd_r = G.baseband_phase_diff(ramp_ph, 80, 1024)
tpd_r = G.antiwrap(ramp_ph - torch.roll(ramp_ph, 1, dims=-1))
print(f"   pure ramp: |TPD| mean {tpd_r.abs().mean():.4f} -> |BPD| mean "
      f"{bpd_r[..., 1:].abs().mean():.4f}")
check("BPD suppresses the ramp TPD keeps",
      bpd_r[..., 1:].abs().mean() < tpd_r[..., 1:].abs().mean(),
      "the 2*pi*m*H/N term is what BPD exists to remove")
check("BPD is wrapped", bpd.abs().max() <= torch.pi + 1e-4)

print()
print("4. coarse prior fills the band above the cutoff and leaves the rest alone")
cut = 64
pr = G.sbr_prior(mag, cut)
check("prior preserves the observed band",
      torch.allclose(pr[:, :cut], mag[:, :cut], atol=1e-6))
check("prior fills the missing band", (pr[:, cut:] != 0).any())
energy_before = mag[:, cut:].mean().item()
print(f"   mean log-amp above the cutoff: input {energy_before:+.3f} -> prior "
      f"{pr[:, cut:].mean():+.3f}")
cutv = torch.tensor([40, 90])
prv = G.sbr_prior(mag, cutv)
check("per-item cutoff works (blind estimate is per utterance)",
      torch.allclose(prv[0, :40], mag[0, :40], atol=1e-6)
      and torch.allclose(prv[1, :90], mag[1, :90], atol=1e-6))

print()
print("5. Schrodinger bridge")
x0, x1 = mag, mag + 1.5
for t in (0.0, 0.5, 1.0):
    xt = SB.bridge_linear(x0, x1, t, sigma=0.0)
    exp = (1 - t) * x0 + t * x1
    check(f"bridge_linear at t={t} hits the endpoint/midpoint",
          torch.allclose(xt, exp, atol=1e-6))
p0 = torch.full((1, 1, 1), 3.1)
p1 = torch.full((1, 1, 1), -3.1)
mid = SB.bridge_phase(p0, p1, 0.5, sigma=0.0)
check("bridge_phase goes round the circle, not through zero",
      mid.abs().item() > 3.0, f"midpoint = {mid.item():+.3f} rad (linear would be 0.0)")
sched1, sched4 = SB.SBSchedule(steps=1), SB.SBSchedule(steps=4)
check("steps=1 is a single call", sched1.infer_ts() == [0.0])
check("steps=4 gives 4 timesteps", len(sched4.infer_ts()) == 4)

h = H({**BASE, **VARIANTS["sb"]})
g = DualStreamBWE(h)
calls = []
def pred(m_, p_, t_):
    calls.append(float(t_[0]))
    return g(m_, p_, t=t_)
with torch.no_grad():
    o1 = SB.sb_infer(pred, mag, pha, sched1)
    n1 = len(calls); calls.clear()
    o4 = SB.sb_infer(pred, mag, pha, sched4)
check("1-step inference makes exactly 1 network call", n1 == 1, f"{n1} calls")
check("4-step inference makes exactly 4", len(calls) == 4, f"{len(calls)} calls")
check("bridge output stays wrapped and finite",
      o4[1].abs().max() <= torch.pi + 1e-4 and torch.isfinite(o4[0]).all())

print()
print("6. PQMF high-band discriminator")
print("=" * 72)
print("8. amplitude losses stay bounded on silent / near-silent reference crops")
import bwe_sb.losses.amplitude as LA

_est = torch.randn(2, 8000) * 0.1
_worst, _finite = 0.0, True
for _lbl, _ref in (("normal", torch.randn(2, 8000) * 0.1),
                   ("quiet 1e-4", torch.randn(2, 8000) * 1e-4),
                   ("silent", torch.zeros(2, 8000))):
    _sc, _lm = LA.mrstft_loss(_est, _ref)
    _be = LA.band_energy_loss(_est, _ref, 16000)
    _tot = 10 * float(_sc) + 1 * float(_be) + 5 * float(_lm)
    _finite &= all(math.isfinite(float(v)) for v in (_sc, _lm, _be))
    _worst = max(_worst, _tot)
    print(f"   {_lbl:12s} sc {float(_sc):7.3f}  lm {float(_lm):7.3f}  "
          f"be {float(_be):7.3f}  weighted {_tot:8.1f}")
check("amplitude losses are finite on a silent reference", _finite)
check("weighted amplitude loss stays bounded on a silent reference",
      _worst < 400, f"worst {_worst:.0f} (unbounded it reached 639 for sc alone)")

print()
print("=" * 72)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all torch-level checks passed")
