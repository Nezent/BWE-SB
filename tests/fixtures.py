from __future__ import annotations
"""Synthetic run directories and paired corpora so the reporting code can be exercised without a GPU."""
import re

import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

import bwe_sb.data.degrade as DG
import bwe_sb.metrics.metrics as MT

try:
    import torch
    if not hasattr(torch, "Generator"):
        torch = None
except Exception:
    torch = None

def metric_schema(sr=16000, f_cut=1000.0, want_perceptual=True) -> List[str]:
    """The exact float keys MT.evaluate() returns."""
    keys = MT.metric_keys(sr, want_perceptual)
    if torch is not None:
        g = torch.Generator().manual_seed(0)
        ref = torch.randn(1, sr, generator=g) * 0.1
        est = ref * 0.9 + torch.randn(1, sr, generator=g) * 0.02
        m = MT.evaluate(est, ref, sr, f_cut, dev="cpu", want_perceptual=want_perceptual)
        live = [k for k, v in m.items() if isinstance(v, float)]
        assert set(live) == set(keys), \
            f"schema mismatch: {sorted(set(live) ^ set(keys))}"
    return keys

def _plausible(key: str, rng: random.Random, quality: float) -> float:
    """A number in the right range for that metric, so plots and sorts behave as they would on real data."""
    q = quality
    if key.startswith("lsd"):
        return round(3.2 - 1.8 * q + rng.uniform(-.1, .1), 4)
    if key in ("si_sdr", "si_snr", "snr"):
        return round(-6 + 16 * q + rng.uniform(-.8, .8), 4)
    if key == "alpha":
        return round(0.15 + 0.8 * q + rng.uniform(-.03, .03), 4)
    if key.startswith("apd"):
        return round(1.6 - 0.9 * q + rng.uniform(-.05, .05), 4)
    if key == "pesq":
        return round(1.1 + 1.4 * q + rng.uniform(-.05, .05), 4)
    if key in ("stoi", "estoi"):
        return round(0.45 + 0.45 * q + rng.uniform(-.02, .02), 4)
    if key in ("mos", "nisqa_mos"):
        return round(2.1 + 1.6 * q + rng.uniform(-.08, .08), 4)
    if key.startswith("d_"):
        return round(0.9 - 0.6 * q + rng.uniform(-.04, .04), 4)
    return round(rng.uniform(0, 1), 4)

def _recipes(bank: str, cap: int) -> List[Dict]:
    """Subsample a bank exactly as run_val() does, so recipe keys and counts match."""
    r = DG.bank(bank)
    if cap and len(r) > cap and bank != "aurora" and not re.fullmatch(r"R\d", bank):
        keep = [x for x in r if set(x) == {"cutoff"}]
        rest = [x for x in r if set(x) != {"cutoff"}]
        n_rest = max(0, cap - len(keep))
        if rest and n_rest:
            idx = np.linspace(0, len(rest) - 1, n_rest).round().astype(int)
            rest = [rest[i] for i in sorted(set(idx.tolist()))]
        else:
            rest = []
        r = keep + rest
    return r

def _recipe_key(bank: str, recipe: Dict) -> str:
    return f"{bank}|" + "_".join(f"{a}={b}" for a, b in sorted(recipe.items()))

def write_run(out: Path, name: str, *, banks: List[str], cfg: Dict, quality=0.6,
              backbone="?", sb_x0="prior",
              train_bank="sinc", conditioned=False, hpc_weight=0.0, generator="cnx",
              critics=("MRAD", "MRPD"), steps=6250, epochs=50, seed=1234,
              eval_max_recipes=6, val_files=6, stats_files=20, sr=16000,
              with_samples=True, rng=None) -> Path:
    """One complete synthetic run directory."""
    out.mkdir(parents=True, exist_ok=True)
    rng = rng or random.Random(abs(hash(name)) % (2 ** 31))
    keys = metric_schema(sr=sr)

    hardness = {"sinc": 0.0, "R0": 0.0, "R1": .12, "R2": .18, "R3": .35, "R4": .45,
                "mixed": .3, "unseen": .4, "aurora": .55}

    by_recipe, by_bank, utt_rows = {}, {}, []
    for b in banks:
        recs = _recipes(b, eval_max_recipes)
        pen = hardness.get(b, .3) * (0.35 if train_bank != "sinc" else 1.0)
        rows = []
        for ri, rec in enumerate(recs):
            q = max(0.02, min(0.98, quality - pen + rng.uniform(-.03, .03)))
            vals = {k: _plausible(k, rng, q) for k in keys}
            vals.update(in_lsd_hf=vals.get("lsd_hf", 2.0) + 0.9, in_si_sdr=vals.get("si_sdr", 5.0) - 4.0)
            by_recipe[_recipe_key(b, rec)] = vals
            rows.append(vals)
            n_files = stats_files if b == banks[0] else val_files
            for i in range(n_files):
                row = dict(bank=b, recipe_idx=ri, utt=i,
                           rec_tier=str(rec.get("tier", "")),
                           rec_cutoff=rec.get("cutoff", ""),
                           rec_snr=rec.get("snr", ""),
                           rec_noise=rec.get("noise", ""))
                row.update({k: round(v + rng.gauss(0, abs(v) * .08 + 1e-3), 5)
                            for k, v in vals.items()})
                utt_rows.append(row)
        by_bank[b] = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys} \
            if rows else {}

    res = dict(name=name, generator=generator, backbone=backbone, sb_x0=sb_x0,
               critics="+".join(critics) or "none",
               hpc_weight=hpc_weight, conditioned=conditioned, train_bank=train_bank,
               gen_params=31_810_571 + rng.randint(0, 300_000),
               enc_params=345_478 if conditioned else 0,
               disc_params=1_203_655,
               disc_breakdown={c: 600_198 if c.startswith("MR") else 3_259
                               for c in critics},
               steps=steps, epochs=epochs, seed=seed,
               minutes=round(rng.uniform(44, 52), 3),
               macs_g=5.5998, flops_g=11.1997,
               infer_ms=round(rng.uniform(6.2, 6.7), 4),
               rtf=round(rng.uniform(.0062, .0067), 6), device="cuda",
               by_bank=by_bank, by_recipe=by_recipe)
    res.update(by_bank.get(banks[0], {}))
    (out / "result.json").write_text(json.dumps(res, indent=2))
    (out / "exp.json").write_text(json.dumps(
        dict(name=name, generator=generator, critics=list(critics), cfg=cfg,
             train_bank=train_bank, eval_banks=banks), indent=2))

    with (out / "per_utt.csv").open("w", newline="") as f:
        cols = list(dict.fromkeys(k for r in utt_rows for k in r))
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader(); w.writerows(utt_rows)

    with (out / "curves.csv").open("w") as f:
        f.write("step,loss_g,loss_d,mag,pha,com,stft,fm,adv,hpc,desc,bpd,sc,be,lm\n")
        for st in range(0, steps + 1, 25):
            d = math.exp(-st / 2000.0)
            f.write(",".join(f"{v:.4f}" for v in (
                st, 350 + 900 * d + rng.gauss(0, 12), 2.5 + 11 * d + rng.gauss(0, .5),
                40 + 120 * d, 95 + 60 * d, 60 + 90 * d, 28 + 40 * d,
                9 + 5 * d, 2.4 + 1.5 * d,
                (hpc_weight * (1.5 + d)) if hpc_weight else 0.0,
                (0.5 + 2.4 * d) if conditioned else 0.0,
                1.3, 3 + 7 * d, 12 + 20 * d, 8 + 4 * d)) + "\n")

    (out / "train.log").write_text(
        f"[{name}] step {steps}/{steps} G 350.00 D 2.50 hpc 0.000 desc 0.000\n"
        "  [metrics] MOS backend: torchaudio SQUIM_SUBJECTIVE "
        "(NISQA unavailable; different scale from NISQA-MOS)\n"
        f"RESULT {json.dumps({k: v for k, v in res.items() if k not in ('by_bank','by_recipe')})}\n")

    if with_samples and torch is not None:
        t = torch.arange(sr) / sr
        ref = (0.3 * torch.sin(2 * math.pi * 190 * t)
               + 0.15 * torch.sin(2 * math.pi * 1300 * t)
               + 0.05 * torch.randn(sr))
        nb = ref * 0.8
        est = ref + 0.03 * torch.randn(sr)
        items = [dict(nb=nb, est=est, ref=ref) for _ in range(3)]
        blob = {b: dict(recipe=DG.bank(b)[0], items=items) for b in banks[:3]}
        torch.save(dict(sr=sr, banks=blob, recipe=DG.bank(banks[0])[0], items=items),
                   out / "samples.pt")
    return out

def build(runs_dir: Path, grid: List[Dict], banks: List[str], cfg: Dict, **kw) -> List[Path]:
    """Fixtures for a whole grid."""
    made = []
    for gi, g in enumerate(grid):
        q = 0.45 if g.get("generator") == "apbwe" else 0.62 + 0.04 * gi
        made.append(write_run(
            runs_dir / g["name"], g["name"], banks=banks, cfg=cfg, quality=min(q, 0.9),
            train_bank=g.get("bank", "sinc"), conditioned=bool(g.get("cond")),
            hpc_weight=g.get("hpc", 0.0) or 0.0,
            generator=g.get("generator", "cnx"), backbone=g.get("backbone", "?"),
            sb_x0=("raw" if g.get("x0_prior") is False else "prior"),
            critics=tuple(g.get("critics") or ("MRAD", "MRPD")), **kw))
    return made

def write_paired(lr_dir, hr_dir, list_path, n=12, sr=16000, seed=0):
    """A synthetic in-ear / headset pair: aligned, correct polarity, and band-limited by a smooth transfer functio..."""
    import soundfile as sf
    rng = np.random.default_rng(seed)
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for i in range(n):
        t = np.arange(sr) / sr
        f0 = 100 + 60 * rng.random()
        hr = sum((0.5 / h) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 40))
        hr = (hr / np.abs(hr).max() * 0.6).astype(np.float32)
        F = np.fft.rfft(hr)
        f = np.fft.rfftfreq(len(hr), 1 / sr)
        g = 1.0 / (1.0 + (f / 900.0) ** 2) + 0.4 * np.exp(-((f - 180) / 120) ** 2)
        lr = np.fft.irfft(F * g, n=len(hr)).astype(np.float32)
        lr += (0.002 * rng.standard_normal(len(lr))).astype(np.float32)
        fid = f"spk{i % 5:02d}_{i:03d}"
        sf.write(hr_dir / f"{fid}.wav", hr, sr, "PCM_16")
        sf.write(lr_dir / f"{fid}.wav", lr, sr, "PCM_16")
        ids.append(fid)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(ids) + "\n")
    return dict(split="test", files=len(ids), speakers=5, lag_median=0, lag_min=0,
                lag_max=0, corr_median=0.82, polarity_flips=0, dropped_weak_corr=0,
                scanned=n, synthetic=True)

def write_paired_scores(out_csv, model, ids, bank="vibravox", quality=0.35, seed=0):
    """Per-pair metric rows in the shape eval_ext.py writes in paired mode."""
    import csv as _csv
    rng = random.Random(seed)
    keys = metric_schema()
    rows = []
    for fid in ids:
        row = dict(model=model, dataset="synthetic", file=fid, bank=bank, recipe_idx=0)
        row.update({k: _plausible(k, rng, max(.02, quality + rng.uniform(-.04, .04)))
                    for k in keys})
        rows.append(row)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader(); w.writerows(rows)
    return rows
