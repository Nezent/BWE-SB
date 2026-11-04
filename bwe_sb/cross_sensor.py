"""Vibravox cross-sensor test: download and align the pairs, score every run, plot the transfer."""
from pathlib import Path
from bwe_sb.data import vibravox
from bwe_sb import evaluate as EV
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
        plt.close('all')
        assert abs(med) <= 1, f'pairs still misaligned (median {med}). Phase metrics compare bin by bin, so this would score every model as noise. Delete {ns.VIB} and re-export.'
        print('aligned and polarity-corrected: safe to evaluate')


def score_runs(ns):
    vib_rows = []
    if ns.vib_report:
        VEX = ns.WORK / 'vib_eval'
        VEX.mkdir(exist_ok=True)
        for d in sorted(ns.RUNS.iterdir()):
            if not d.is_dir():
                continue
            outf = VEX / f'{d.name}.csv'
            if not outf.exists():
                if not ((d / 'ckpt.pt').exists() and (d / 'exp.json').exists()):
                    continue
                wav_root = VEX / 'wavs'
                print('=' * 70)
                print(d.name, flush=True)
                rc = EV.main(['--model', str(d), '--paired-lr', str(ns.VIB_LR), '--paired-hr', str(ns.VIB_HR), '--paired-list', str(ns.VIB_LIST), '--bank-name', 'vibravox', '--max-files', str(ns.VIB_EVAL_FILES), '--out', str(outf), '--write-wavs', str(wav_root)])
                if rc:
                    raise RuntimeError(f'{d.name} failed on Vibravox')
                wdir = wav_root / d.name / 'vibravox'
                if ns.NISQA_DIR and Path(ns.NISQA_DIR).exists() and wdir.exists():
                    r = subprocess.run([sys.executable, 'run_predict.py', '--mode', 'predict_dir', '--pretrained_model', str(Path(ns.NISQA_DIR) / 'weights' / 'nisqa.tar'), '--data_dir', str(wdir), '--num_workers', '0', '--bs', '10', '--output_dir', str(wdir)], cwd=str(ns.NISQA_DIR), capture_output=True, text=True)
                    res_csv = wdir / 'NISQA_results.csv'
                    if r.returncode != 0 or not res_csv.exists():
                        raise RuntimeError(f'NISQA failed on Vibravox for {d.name}: {(r.stderr or r.stdout)[-600:]}')
                    nq = pd.read_csv(res_csv)
                    nq['file'] = nq['deg'].map(lambda p: Path(p).stem)
                    t = pd.read_csv(outf)
                    t = t.drop(columns=[c for c in ('nisqa_mos',) if c in t.columns])
                    t = t.merge(nq[['file', 'mos_pred']].rename(columns={'mos_pred': 'nisqa_mos'}), on='file', how='left')
                    t.to_csv(outf, index=False)
                    print(f'  NISQA-MOS (Vibravox): {t.nisqa_mos.mean():.3f} over {t.nisqa_mos.notna().sum()} pairs', flush=True)
            t = pd.read_csv(outf)
            t['model'] = d.name
            vib_rows.append(t)
    if vib_rows:
        ns.vdf = pd.concat(vib_rows, ignore_index=True)
        ns.vdf.to_csv(ns.WORK / 'vibravox_results.csv', index=False)
        VMETS = [m for m in ('lsd_hf', 'si_sdr', 'pesq', 'stoi', 'estoi', 'mos', 'nisqa_mos', 'apd_ip', 'alpha') if m in ns.vdf.columns]
        ns.vsum = ns.vdf.groupby('model')[VMETS].mean().round(4)
        print('\n--- Vibravox (cross-sensor, zero-shot from VCTK) ---')
        ns.display(ns.vsum)
        print(f'\n{len(ns.vdf)} pairs scored per model: {len(ns.vdf) // max(1, ns.vdf.model.nunique())}')
    else:
        ns.vdf = pd.DataFrame()
        print('no Vibravox results')


def transfer_figure(ns):
    if len(ns.vdf):
        VMETS2 = [m for m in ('lsd_hf', 'si_sdr', 'pesq', 'stoi', 'nisqa_mos') if m in ns.vdf.columns]
        vctk = ns.df.set_index('name')
        xs_rows = []
        for mn in ns.vsum.index:
            for m in VMETS2:
                iv = float(vctk.loc[mn, m]) if mn in vctk.index and m in vctk else np.nan
                xs_rows.append(dict(model=mn, metric=m, vctk=iv, vibravox=float(ns.vsum.loc[mn, m])))
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
            order = g['drop'].sort_values()
            _bb = b.barh(np.arange(len(order)), order.values, color=['#6acc64' if v <= 0 else '#d65f5f' for v in order.values])
            b.axvline(0, c='k', lw=0.8)
            b.bar_label(_bb, fmt='%+.3f', fontsize=7, padding=3)
            b.margins(x=0.2)
            if len(order) == 1:
                b.set_ylim(-1.2, 1.2)
            b.set(yticks=np.arange(len(order)), title=f'{m} degradation\n(shorter = travels better)')
            b.set_yticklabels(order.index, fontsize=6)
        fig.suptitle('VCTK-trained, Vibravox-tested. Top: absolute. Bottom: how much was lost.', y=1.0, fontsize=10)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'cross_sensor.png', dpi=140)
        plt.close('all')


STEPS = [download_and_pair, alignment_check, score_runs, transfer_figure]
