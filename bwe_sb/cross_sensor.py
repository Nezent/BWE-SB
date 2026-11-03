"""Vibravox cross-sensor test: download and align the pairs, score every run, plot the transfer."""
from pathlib import Path
import importlib, vibravox
import json
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shutil
import soundfile as sf
import subprocess
import sys
import time


def download_and_pair(ns):
    ns.VIB = ns.WORK / 'vibravox'
    ns.VIB_LR, ns.VIB_HR = (ns.VIB / 'in_ear', ns.VIB / 'headset')
    ns.VIB_LIST = ns.VIB / 'test.txt'
    if ns.VIBRAVOX and (not ns.VIB_LIST.exists()):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'], check=False)
        importlib.reload(vibravox)
        _saved = [p for p in sys.path if p == str(ns.REPO)]
        for p in _saved:
            sys.path.remove(p)
        for m in [k for k in list(sys.modules) if k.split('.')[0] == 'datasets']:
            del sys.modules[m]
        try:
            ns.vib_report = vibravox.export('test', ns.VIB_PER_SPEAKER, ns.VIB_LR, ns.VIB_HR, ns.VIB_LIST, max_scan=ns.VIB_MAX_SCAN, token=ns.HF_TOKEN)
        finally:
            for m in [k for k in list(sys.modules) if k.split('.')[0] == 'datasets']:
                del sys.modules[m]
            for p in _saved:
                sys.path.insert(0, p)
        (ns.VIB / 'report.json').write_text(json.dumps(ns.vib_report, indent=2))
    elif ns.VIB_LIST.exists():
        ns.vib_report = json.loads((ns.VIB / 'report.json').read_text())
        print(f"reusing {ns.vib_report['files']} aligned pairs from {ns.vib_report['speakers']} speakers")
    else:
        ns.vib_report = None
        print('Vibravox disabled (VIBRAVOX=False)')


def alignment_check(ns):
    if ns.vib_report:
        importlib.reload(vibravox)
        ids = ns.VIB_LIST.read_text().split()
        lags = vibravox.residual_lags(ns.VIB_LR, ns.VIB_HR, ids, n=60)
        med = int(np.median(lags))
        print(f'residual lag: median {med}, range [{min(lags)}, {max(lags)}] over {len(lags)} pairs')
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
        ax[0].hist(lags, bins=range(min(lags) - 1, max(lags) + 2), color='#4878d0')
        ax[0].axvline(0, c='k', ls='--')
        ax[0].set(xlabel='samples', ylabel='pairs', title=f'residual lag after correction (median {med})')
        a_, _ = sf.read(ns.VIB_HR / f'{ids[0]}.wav', dtype='float32')
        b_, _ = sf.read(ns.VIB_LR / f'{ids[0]}.wav', dtype='float32')
        k_ = min(len(a_), len(b_), 4 * ns.SR)
        for sig, lab in ((a_[:k_], 'headset (reference)'), (b_[:k_], 'in-ear (input)')):
            f_ = np.fft.rfftfreq(k_, 1 / ns.SR)
            P = np.abs(np.fft.rfft(sig * np.hanning(k_)))
            ax[1].semilogx(f_[1:], 20 * np.log10(P[1:] / P[1:].max() + 1e-12), label=lab)
        ax[1].set(xlabel='Hz', ylabel='dB', ylim=(-80, 5), title='why this is not a resampling task')
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'vibravox_alignment.png', dpi=140)
        plt.show()
        assert abs(med) <= 1, f'pairs still misaligned (median {med}). Phase metrics compare bin by bin, so this would score every model as noise. Delete {ns.VIB} and re-export.'
        print('aligned and polarity-corrected: safe to evaluate')


def score_runs(ns):
    Path(SRC / 'eval_ext.py').write_text('\nfrom __future__ import annotations\n\nimport argparse\nimport csv\nimport json\nimport sys\nfrom pathlib import Path\nfrom typing import Dict, List\n\nimport torch\nimport torchaudio\nimport torchaudio.functional as aF\n\ndef stft_pair():\n    """AP-BWE\'s amp_pha_stft / amp_pha_istft, from the repo if it is importable and from a\n    local copy otherwise. The local copy is what makes this file usable on its own: a\n    checkpoint is worth little if testing it needs the original repo checked out."""\n    try:\n        from bwe_sb.data.dataset import amp_pha_istft, amp_pha_stft\n        return amp_pha_stft, amp_pha_istft\n    except Exception:\n        pass\n\n    def amp_pha_stft(audio, n_fft, hop_size, win_size, center=True):\n        hann = torch.hann_window(win_size).to(audio.device)\n        spec = torch.stft(audio, n_fft, hop_length=hop_size, win_length=win_size,\n                          window=hann, center=center, pad_mode="reflect",\n                          normalized=False, return_complex=True)\n        log_amp = torch.log(torch.abs(spec) + 1e-4)\n        pha = torch.angle(spec)\n        com = torch.stack((torch.exp(log_amp) * torch.cos(pha),\n                           torch.exp(log_amp) * torch.sin(pha)), dim=-1)\n        return log_amp, pha, com\n\n    def amp_pha_istft(log_amp, pha, n_fft, hop_size, win_size, center=True):\n        amp = torch.exp(log_amp)\n        com = torch.complex(amp * torch.cos(pha), amp * torch.sin(pha))\n        hann = torch.hann_window(win_size).to(com.device)\n        return torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size,\n                           window=hann, center=center)\n\n    print("  [eval_ext] AP-BWE repo not importable; using the bundled STFT pair")\n    return amp_pha_stft, amp_pha_istft\n\ndef load_model(path, dev):\n    """Rebuild the generator. Accepts either a model_final.pt export or a RUN DIRECTORY\n    holding ckpt.pt + exp.json.\n\n    The directory form matters when training and testing happen in the same notebook:\n    nothing has to be exported or kept, the checkpoint the trainer already wrote is\n    enough.\n    """\n    path = Path(path)\n    if path.is_dir():\n        ck = torch.load(path / "ckpt.pt", map_location=dev, weights_only=False)\n        exp = json.loads((path / "exp.json").read_text())\n        cfg = exp["cfg"]\n        blob = dict(format="bwe-export-1", name=exp["name"],\n                    generator=exp.get("generator", "cnx"), cfg=cfg,\n                    G=ck["G"], enc=ck.get("enc"),\n                    enc_emb=int(cfg.get("gen_cond_dim") or 0),\n                    n_bins=cfg["n_fft"] // 2 + 1,\n                    sr=cfg["hr_sampling_rate"], lr_sr=cfg["lr_sampling_rate"],\n                    n_fft=cfg["n_fft"], hop_size=cfg["hop_size"],\n                    win_size=cfg["win_size"],\n                    critics=exp.get("critics", []),\n                    train_bank=exp.get("train_bank", "sinc"),\n                    steps=int(ck.get("step", 0)), gen_params=0, enc_params=0)\n    else:\n        blob = torch.load(path, map_location=dev, weights_only=False)\n    assert blob.get("format") == "bwe-export-1", \\\n        f"{path} is neither a bwe export nor a run directory with ckpt.pt + exp.json"\n\n    class H(dict):\n        __getattr__ = dict.get\n    h = H(blob["cfg"])\n\n    if blob["generator"] == "apbwe":\n        from bwe_sb.models.critics.apbwe import APNet_BWE_Model\n        G = APNet_BWE_Model(h).to(dev)\n    else:\n        from bwe_sb.models.generator import DualStreamBWE\n        G = DualStreamBWE(h).to(dev)\n    G.load_state_dict(blob["G"])\n    G.eval()\n\n    enc = None\n    if blob.get("enc") is not None:\n        import bwe_sb.data.degrade as DG\n        enc = DG.DegradationEncoder(blob["n_bins"], emb=blob["enc_emb"] or 64).to(dev)\n        enc.load_state_dict(blob["enc"])\n        enc.eval()\n\n    @torch.no_grad()\n    def fwd(mag_i, pha_i):\n        if enc is None:\n            return G(mag_i, pha_i)\n        emb, _ = enc(mag_i)\n        return G(mag_i, pha_i, cond=emb)\n\n    return fwd, blob\n\ndef find_wavs(root, exts=(".wav", ".flac", ".mp3", ".ogg")):\n    root = Path(root)\n    if root.is_file():\n        return [root]\n    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)\n\ndef _write(a, rows, banks):\n    """CSV plus a per-bank mean table; shared by the synthetic and paired modes."""\n    import numpy as np\n    assert rows, "nothing was evaluated"\n    cols = list(dict.fromkeys(k for r in rows for k in r))\n    with open(a.out, "w", newline="") as fh:\n        w = csv.DictWriter(fh, fieldnames=cols, restval="")\n        w.writeheader()\n        w.writerows(rows)\n    print(f"\\nwrote {len(rows)} rows to {a.out}")\n    num = [c for c in cols if c not in ("model", "dataset", "file", "bank", "recipe_idx")]\n    print(f"\\n{\'bank\':10s}" + "".join(f"{c:>11s}" for c in num[:9]))\n    for bname in banks:\n        sub = [r for r in rows if r["bank"] == bname]\n        if not sub:\n            continue\n        vals = [np.nanmean([r.get(c, np.nan) for r in sub]) for c in num[:9]]\n        print(f"{bname:10s}" + "".join(f"{v:11.3f}" for v in vals))\n\n@torch.no_grad()\ndef main(argv=None):\n    ap = argparse.ArgumentParser(description=__doc__,\n                                 formatter_class=argparse.RawDescriptionHelpFormatter)\n    ap.add_argument("--model", required=True,\n                    help="model_final.pt, or a run directory with ckpt.pt + exp.json")\n    ap.add_argument("--wavs", default=None,\n                    help="dataset directory (searched recursively); not needed in paired mode")\n    ap.add_argument("--banks", nargs="*", default=["sinc"],\n                    help="degradation banks to evaluate under (sinc mixed unseen aurora R0..R4)")\n    ap.add_argument("--paired-lr", default=None,\n                    help="PAIRED mode: directory of real degraded inputs (e.g. Vibravox "\n                         "in-ear). No synthetic degradation is applied.")\n    ap.add_argument("--paired-hr", default=None,\n                    help="PAIRED mode: directory of matching references, same filenames")\n    ap.add_argument("--paired-list", default=None,\n                    help="PAIRED mode: optional file of ids, one per line")\n    ap.add_argument("--bank-name", default="paired",\n                    help="PAIRED mode: label for the bank column")\n    ap.add_argument("--max-files", type=int, default=200)\n    ap.add_argument("--max-recipes", type=int, default=0, help="0 = every recipe")\n    ap.add_argument("--max-seconds", type=float, default=10.0, help="trim long files")\n    ap.add_argument("--out", default="ext_eval.csv")\n    ap.add_argument("--write-wavs", default=None, help="directory to save extended audio")\n    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")\n    ap.add_argument("--no-perceptual", action="store_true",\n                    help="skip PESQ/STOI/MOS (much faster)")\n    a = ap.parse_args(argv)\n\n    dev = torch.device(a.device)\n    import bwe_sb.data.degrade as DG\n    import bwe_sb.metrics.metrics as MT\n    amp_pha_stft, amp_pha_istft = stft_pair()\n\n    fwd, blob = load_model(a.model, dev)\n    sr, n_fft, hop, win = blob["sr"], blob["n_fft"], blob["hop_size"], blob["win_size"]\n    _p = blob.get("gen_params") or 0\n    print(f"model: {blob[\'name\']}  ("\n          + (f"{_p/1e6:.2f}M params, " if _p else "")\n          + f"trained on \'{blob[\'train_bank\']}\' for {blob[\'steps\']} steps)")\n\n    if a.paired_lr:\n        assert a.paired_hr, "--paired-lr requires --paired-hr"\n        lrd, hrd = Path(a.paired_lr), Path(a.paired_hr)\n        if a.paired_list and Path(a.paired_list).exists():\n            ids = [l.strip() for l in Path(a.paired_list).read_text().split() if l.strip()]\n        else:\n            ids = sorted(p.stem for p in lrd.glob("*.wav"))\n        ids = [i for i in ids if (hrd / f"{i}.wav").exists()][: a.max_files]\n        assert ids, f"no paired files between {lrd} and {hrd}"\n        print(f"PAIRED: {len(ids)} pairs | input {lrd} | reference {hrd}")\n\n        rows = []\n        for i, fid in enumerate(ids):\n            lr, s1 = torchaudio.load(str(lrd / f"{fid}.wav"))\n            hr, s2 = torchaudio.load(str(hrd / f"{fid}.wav"))\n            lr, hr = lr.mean(0, keepdim=True), hr.mean(0, keepdim=True)\n            if s1 != sr:\n                lr = aF.resample(lr, s1, sr)\n            if s2 != sr:\n                hr = aF.resample(hr, s2, sr)\n            k = min(lr.shape[-1], hr.shape[-1])\n            if a.max_seconds:\n                k = min(k, int(a.max_seconds * sr))\n            lr, hr = lr[..., :k].to(dev), hr[..., :k].to(dev)\n            mi, pi, _ = amp_pha_stft(lr, n_fft, hop, win)\n            mg, pg, _ = fwd(mi, pi)\n            est = amp_pha_istft(mg, pg, n_fft, hop, win)\n            k2 = min(est.shape[-1], hr.shape[-1])\n            m = MT.evaluate(est[..., :k2], hr[..., :k2], sr, blob["lr_sr"] / 2,\n                            dev=dev, want_perceptual=not a.no_perceptual)\n            row = dict(model=blob["name"], dataset=str(lrd), file=fid,\n                       bank=a.bank_name, recipe_idx=0)\n            row.update({k3: v for k3, v in m.items() if isinstance(v, float)})\n            rows.append(row)\n            if a.write_wavs:\n                d = Path(a.write_wavs) / blob["name"] / a.bank_name\n                d.mkdir(parents=True, exist_ok=True)\n                torchaudio.save(str(d / f"{fid}.wav"), est[..., :k2].cpu().clamp(-1, 1), sr)\n            if (i + 1) % 25 == 0:\n                print(f"  {i+1}/{len(ids)}", flush=True)\n        _write(a, rows, [a.bank_name])\n        return 0\n\n    files = find_wavs(a.wavs)[: a.max_files]\n    assert files, f"no audio found under {a.wavs}"\n    print(f"dataset: {a.wavs}  ({len(files)} files)")\n\n    rows: List[Dict] = []\n    for bname in a.banks:\n        recipes = DG.bank(bname)\n        if a.max_recipes and len(recipes) > a.max_recipes:\n            recipes = recipes[: a.max_recipes]\n        for ri, recipe in enumerate(recipes):\n            for fi, f in enumerate(files):\n                try:\n                    x, osr = torchaudio.load(str(f))\n                except Exception as e:\n                    print(f"  skip {f.name}: {type(e).__name__}")\n                    continue\n                x = x.mean(0, keepdim=True)\n                if osr != sr:\n                    x = aF.resample(x, osr, sr)\n                if a.max_seconds:\n                    x = x[..., : int(a.max_seconds * sr)]\n                if x.shape[-1] < win * 2:\n                    continue\n                hr = x.to(dev)\n                lr, _ = DG.degrade(hr, sr, recipe)\n                mi, pi, _ = amp_pha_stft(lr, n_fft, hop, win)\n                mg, pg, _ = fwd(mi, pi)\n                est = amp_pha_istft(mg, pg, n_fft, hop, win)\n                k = min(est.shape[-1], hr.shape[-1])\n                m = MT.evaluate(est[..., :k], hr[..., :k], sr,\n                                recipe.get("cutoff", blob["lr_sr"]) / 2, dev=dev,\n                                want_perceptual=not a.no_perceptual)\n                row = dict(model=blob["name"], dataset=str(a.wavs), file=f.name,\n                           bank=bname, recipe_idx=ri)\n                row.update({k2: v for k2, v in m.items() if isinstance(v, float)})\n                rows.append(row)\n                if a.write_wavs:\n                    d = Path(a.write_wavs) / blob["name"] / bname\n                    d.mkdir(parents=True, exist_ok=True)\n                    torchaudio.save(str(d / f"{f.stem}.wav"),\n                                    est[..., :k].cpu().clamp(-1, 1), sr)\n            print(f"  {bname} recipe {ri+1}/{len(recipes)}: {len(rows)} rows", flush=True)\n\n    _write(a, rows, a.banks)\n    return 0\n\nif __name__ == "__main__":\n    sys.exit(main())\n')
    print('eval_ext.py', (SRC / 'eval_ext.py').stat().st_size, 'bytes')


def transfer_figure(ns):
    if len(vdf):
        VMETS2 = [m for m in ('lsd_hf', 'si_sdr', 'pesq', 'stoi', 'nisqa_mos') if m in vdf.columns]
        vctk = ns.df.set_index('name')
        xs_rows = []
        for mn in vsum.index:
            for m in VMETS2:
                iv = float(vctk.loc[mn, m]) if mn in vctk.index and m in vctk else np.nan
                xs_rows.append(dict(model=mn, metric=m, vctk=iv, vibravox=float(vsum.loc[mn, m])))
        xsdf = pd.DataFrame(xs_rows)
        xsdf['drop'] = np.where(xsdf.metric.isin(ns.LOWER), xsdf.vibravox - xsdf.vctk, xsdf.vctk - xsdf.vibravox)
        ns.display(xsdf.pivot_table(index='model', columns='metric', values=['vctk', 'vibravox', 'drop']).round(3))
        xsdf.to_csv(ns.WORK / 'cross_sensor.csv', index=False)
        fig, axs = plt.subplots(2, len(VMETS2), figsize=(3.8 * len(VMETS2), 8), squeeze=False)
        for ci, m in enumerate(VMETS2):
            g = xsdf[xsdf.metric == m].set_index('model')
            yy = np.arange(len(g))
            w = 0.38
            a = axs[0][ci]
            a.barh(yy - w / 2, g.vctk.values, w, label='VCTK (in-domain)', color='#999999')
            a.barh(yy + w / 2, g.vibravox.values, w, label='Vibravox (unseen sensor)', color='#d65f5f')
            a.set(yticks=yy, title=f"{m}\n({('lower' if m in ns.LOWER else 'higher')} better)")
            a.set_yticklabels(g.index, fontsize=6)
            if ci == 0:
                a.legend(fontsize=7)
            b = axs[1][ci]
            ns.order = g['drop'].sort_values()
            _bb = b.barh(np.arange(len(ns.order)), ns.order.values, color=['#6acc64' if v <= 0 else '#d65f5f' for v in ns.order.values])
            b.axvline(0, c='k', lw=0.8)
            b.bar_label(_bb, fmt='%+.3f', fontsize=7, padding=3)
            b.margins(x=0.2)
            if len(ns.order) == 1:
                b.set_ylim(-1.2, 1.2)
            b.set(yticks=np.arange(len(ns.order)), title=f'{m} degradation\n(shorter = travels better)')
            b.set_yticklabels(ns.order.index, fontsize=6)
        fig.suptitle('VCTK-trained, Vibravox-tested. Top: absolute. Bottom: how much was lost.', y=1.0, fontsize=10)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'cross_sensor.png', dpi=140)
        plt.show()


STEPS = [download_and_pair, alignment_check, score_runs, transfer_figure]
