from __future__ import annotations
"""Evaluate a trained run on a wav directory or a paired corpus such as Vibravox."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio
import torchaudio.functional as aF

def stft_pair():
    """AP-BWE's amp_pha_stft / amp_pha_istft, from the repo if it is importable and from a local copy otherwise."""
    try:
        from bwe_sb.data.dataset import amp_pha_istft, amp_pha_stft
        return amp_pha_stft, amp_pha_istft
    except Exception:
        pass

    def amp_pha_stft(audio, n_fft, hop_size, win_size, center=True):
        hann = torch.hann_window(win_size).to(audio.device)
        spec = torch.stft(audio, n_fft, hop_length=hop_size, win_length=win_size,
                          window=hann, center=center, pad_mode="reflect",
                          normalized=False, return_complex=True)
        log_amp = torch.log(torch.abs(spec) + 1e-4)
        pha = torch.angle(spec)
        com = torch.stack((torch.exp(log_amp) * torch.cos(pha),
                           torch.exp(log_amp) * torch.sin(pha)), dim=-1)
        return log_amp, pha, com

    def amp_pha_istft(log_amp, pha, n_fft, hop_size, win_size, center=True):
        amp = torch.exp(log_amp)
        com = torch.complex(amp * torch.cos(pha), amp * torch.sin(pha))
        hann = torch.hann_window(win_size).to(com.device)
        return torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size,
                           window=hann, center=center)

    print("  [eval_ext] AP-BWE repo not importable; using the bundled STFT pair")
    return amp_pha_stft, amp_pha_istft

def load_model(path, dev):
    """Rebuild the generator."""
    path = Path(path)
    if path.is_dir():
        ck = torch.load(path / "ckpt.pt", map_location=dev, weights_only=False)
        exp = json.loads((path / "exp.json").read_text())
        cfg = exp["cfg"]
        blob = dict(format="bwe-export-1", name=exp["name"],
                    generator=exp.get("generator", "cnx"), cfg=cfg,
                    G=ck["G"], enc=ck.get("enc"),
                    enc_emb=int(cfg.get("gen_cond_dim") or 0),
                    n_bins=cfg["n_fft"] // 2 + 1,
                    sr=cfg["hr_sampling_rate"], lr_sr=cfg["lr_sampling_rate"],
                    n_fft=cfg["n_fft"], hop_size=cfg["hop_size"],
                    win_size=cfg["win_size"],
                    critics=exp.get("critics", []),
                    train_bank=exp.get("train_bank", "sinc"),
                    steps=int(ck.get("step", 0)), gen_params=0, enc_params=0)
    else:
        blob = torch.load(path, map_location=dev, weights_only=False)
    assert blob.get("format") == "bwe-export-1", \
        f"{path} is neither a bwe export nor a run directory with ckpt.pt + exp.json"

    class H(dict):
        __getattr__ = dict.get
    h = H(blob["cfg"])

    from bwe_sb.models.generator import DualStreamBWE
    G = DualStreamBWE(h).to(dev)
    G.load_state_dict(blob["G"])
    G.eval()

    enc = None
    if blob.get("enc") is not None:
        import bwe_sb.data.degrade as DG
        enc = DG.DegradationEncoder(blob["n_bins"], emb=blob["enc_emb"] or 64).to(dev)
        enc.load_state_dict(blob["enc"])
        enc.eval()

    @torch.no_grad()
    def fwd(mag_i, pha_i):
        if enc is None:
            return G(mag_i, pha_i)
        emb, _ = enc(mag_i)
        return G(mag_i, pha_i, cond=emb)

    return fwd, blob

def find_wavs(root, exts=(".wav", ".flac", ".mp3", ".ogg")):
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)

def _write(a, rows, banks):
    """CSV plus a per-bank mean table; shared by the synthetic and paired modes."""
    import numpy as np
    assert rows, "nothing was evaluated"
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {a.out}")
    num = [c for c in cols if c not in ("model", "dataset", "file", "bank", "recipe_idx")]
    print(f"\n{'bank':10s}" + "".join(f"{c:>11s}" for c in num[:9]))
    for bname in banks:
        sub = [r for r in rows if r["bank"] == bname]
        if not sub:
            continue
        vals = [np.nanmean([r.get(c, np.nan) for r in sub]) for c in num[:9]]
        print(f"{bname:10s}" + "".join(f"{v:11.3f}" for v in vals))

@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="model_final.pt, or a run directory with ckpt.pt + exp.json")
    ap.add_argument("--wavs", default=None,
                    help="dataset directory (searched recursively); not needed in paired mode")
    ap.add_argument("--banks", nargs="*", default=["sinc"],
                    help="degradation banks to evaluate under (sinc mixed unseen aurora R0..R4)")
    ap.add_argument("--paired-lr", default=None,
                    help="PAIRED mode: directory of real degraded inputs (e.g. Vibravox "
                         "in-ear). No synthetic degradation is applied.")
    ap.add_argument("--paired-hr", default=None,
                    help="PAIRED mode: directory of matching references, same filenames")
    ap.add_argument("--paired-list", default=None,
                    help="PAIRED mode: optional file of ids, one per line")
    ap.add_argument("--bank-name", default="paired",
                    help="PAIRED mode: label for the bank column")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--max-recipes", type=int, default=0, help="0 = every recipe")
    ap.add_argument("--max-seconds", type=float, default=10.0, help="trim long files")
    ap.add_argument("--out", default="ext_eval.csv")
    ap.add_argument("--write-wavs", default=None, help="directory to save extended audio")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-perceptual", action="store_true",
                    help="skip PESQ/STOI/MOS (much faster)")
    a = ap.parse_args(argv)

    dev = torch.device(a.device)
    import bwe_sb.data.degrade as DG
    import bwe_sb.metrics.metrics as MT
    amp_pha_stft, amp_pha_istft = stft_pair()

    fwd, blob = load_model(a.model, dev)
    DG.BASE = int(blob["lr_sr"])
    sr, n_fft, hop, win = blob["sr"], blob["n_fft"], blob["hop_size"], blob["win_size"]
    _p = blob.get("gen_params") or 0
    print(f"model: {blob['name']}  ("
          + (f"{_p/1e6:.2f}M params, " if _p else "")
          + f"trained on '{blob['train_bank']}' for {blob['steps']} steps)")

    if a.paired_lr:
        assert a.paired_hr, "--paired-lr requires --paired-hr"
        lrd, hrd = Path(a.paired_lr), Path(a.paired_hr)
        if a.paired_list and Path(a.paired_list).exists():
            ids = [l.strip() for l in Path(a.paired_list).read_text().split() if l.strip()]
        else:
            ids = sorted(p.stem for p in lrd.glob("*.wav"))
        ids = [i for i in ids if (hrd / f"{i}.wav").exists()][: a.max_files]
        assert ids, f"no paired files between {lrd} and {hrd}"
        print(f"PAIRED: {len(ids)} pairs | input {lrd} | reference {hrd}")

        rows = []
        for i, fid in enumerate(ids):
            lr, s1 = torchaudio.load(str(lrd / f"{fid}.wav"))
            hr, s2 = torchaudio.load(str(hrd / f"{fid}.wav"))
            lr, hr = lr.mean(0, keepdim=True), hr.mean(0, keepdim=True)
            if s1 != sr:
                lr = aF.resample(lr, s1, sr)
            if s2 != sr:
                hr = aF.resample(hr, s2, sr)
            k = min(lr.shape[-1], hr.shape[-1])
            if a.max_seconds:
                k = min(k, int(a.max_seconds * sr))
            lr, hr = lr[..., :k].to(dev), hr[..., :k].to(dev)
            mi, pi, _ = amp_pha_stft(lr, n_fft, hop, win)
            mg, pg, _ = fwd(mi, pi)
            est = amp_pha_istft(mg, pg, n_fft, hop, win)
            k2 = min(est.shape[-1], hr.shape[-1])
            m = MT.evaluate(est[..., :k2], hr[..., :k2], sr, blob["lr_sr"] / 2,
                            dev=dev, want_perceptual=not a.no_perceptual)
            row = dict(model=blob["name"], dataset=str(lrd), file=fid,
                       bank=a.bank_name, recipe_idx=0)
            row.update({k3: v for k3, v in m.items() if isinstance(v, float)})
            rows.append(row)
            if a.write_wavs:
                d = Path(a.write_wavs) / blob["name"] / a.bank_name
                d.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(d / f"{fid}.wav"), est[..., :k2].cpu().clamp(-1, 1), sr)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(ids)}", flush=True)
        _write(a, rows, [a.bank_name])
        return 0

    files = find_wavs(a.wavs)[: a.max_files]
    assert files, f"no audio found under {a.wavs}"
    print(f"dataset: {a.wavs}  ({len(files)} files)")

    rows: List[Dict] = []
    for bname in a.banks:
        recipes = DG.bank(bname)
        if a.max_recipes and len(recipes) > a.max_recipes:
            recipes = recipes[: a.max_recipes]
        for ri, recipe in enumerate(recipes):
            for fi, f in enumerate(files):
                try:
                    x, osr = torchaudio.load(str(f))
                except Exception as e:
                    print(f"  skip {f.name}: {type(e).__name__}")
                    continue
                x = x.mean(0, keepdim=True)
                if osr != sr:
                    x = aF.resample(x, osr, sr)
                if a.max_seconds:
                    x = x[..., : int(a.max_seconds * sr)]
                if x.shape[-1] < win * 2:
                    continue
                hr = x.to(dev)
                lr, _ = DG.degrade(hr, sr, recipe)
                mi, pi, _ = amp_pha_stft(lr, n_fft, hop, win)
                mg, pg, _ = fwd(mi, pi)
                est = amp_pha_istft(mg, pg, n_fft, hop, win)
                k = min(est.shape[-1], hr.shape[-1])
                m = MT.evaluate(est[..., :k], hr[..., :k], sr,
                                recipe.get("cutoff", blob["lr_sr"]) / 2, dev=dev,
                                want_perceptual=not a.no_perceptual)
                row = dict(model=blob["name"], dataset=str(a.wavs), file=f.name,
                           bank=bname, recipe_idx=ri)
                row.update({k2: v for k2, v in m.items() if isinstance(v, float)})
                rows.append(row)
                if a.write_wavs:
                    d = Path(a.write_wavs) / blob["name"] / bname
                    d.mkdir(parents=True, exist_ok=True)
                    torchaudio.save(str(d / f"{f.stem}.wav"),
                                    est[..., :k].cpu().clamp(-1, 1), sr)
            print(f"  {bname} recipe {ri+1}/{len(recipes)}: {len(rows)} rows", flush=True)

    _write(a, rows, a.banks)
    return 0

if __name__ == "__main__":
    sys.exit(main())
