"""Post-training diagnostics: loss-curve health, magnitude/phase oracle, bridge step sweep."""
from bwe_sb.data.dataset import Dataset, amp_pha_stft, amp_pha_istft
from bwe_sb.models.generator import DualStreamBWE
from bwe_sb import train as T


class _H(dict):
    __getattr__ = dict.get
from pathlib import Path
import json
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas as pd, numpy as np
import re
import shutil
import subprocess
import sys
import time
import torch, numpy as np, importlib


def training_health(ns):
    HEALTH_WINDOW = 100
    SC_LIMIT, BE_LIMIT, D_COLLAPSE = (15.0, 40.0, 0.35)
    problems = []
    for d in sorted(ns.RUNS.iterdir()):
        f = d / 'curves.csv'
        if not f.exists():
            continue
        c = pd.read_csv(f)
        if not len(c):
            continue
        num = c.select_dtypes('number')
        bad = num.columns[~np.isfinite(num).all()].tolist()
        if bad:
            problems.append(f'{d.name}: non-finite values in {bad}')
        head, tail = (c.head(HEALTH_WINDOW), c.tail(HEALTH_WINDOW))
        rows = []
        for col in ('loss_g', 'loss_d', 'mag', 'sc', 'be', 'lm', 'bpd', 'hpc'):
            if col in c:
                rows.append((col, float(head[col].mean()), float(tail[col].mean()), float(tail[col].max())))
        print(f'\n{d.name}  ({len(c)} logged steps)')
        print(f"  {'term':7s} {'first':>9s} {'last':>9s} {'last max':>9s}")
        for col, a, b, m in rows:
            print(f'  {col:7s} {a:9.2f} {b:9.2f} {m:9.2f}')
        t = dict(((r[0], r) for r in rows))
        if 'sc' in t and t['sc'][3] > SC_LIMIT:
            problems.append(f"{d.name}: sc reached {t['sc'][3]:.1f} in the last {HEALTH_WINDOW} rows (limit {SC_LIMIT})")
        if 'be' in t and t['be'][2] > t['be'][1] * 1.5 and (t['be'][2] > BE_LIMIT):
            problems.append(f"{d.name}: be rising {t['be'][1]:.1f} -> {t['be'][2]:.1f}")
        if 'loss_d' in t and t['loss_d'][1] > 0 and (t['loss_d'][2] < D_COLLAPSE * t['loss_d'][1]):
            print(f"  WARNING {d.name}: discriminator loss fell {t['loss_d'][1]:.1f} -> {t['loss_d'][2]:.1f}; watch be/lm for a rise")
    if problems:
        print('\nHEALTH CHECK FAILED')
        for q in problems:
            print('  -', q)
        raise RuntimeError('training is not healthy; fix before a longer tier')
    print('\ntraining health OK')


def oracle_decomposition(ns):

    def _sisdr_a(e, r, eps=1e-08):
        k = min(e.shape[-1], r.shape[-1])
        e, r = (e[..., :k], r[..., :k])
        e = e - e.mean(-1, keepdim=True)
        r = r - r.mean(-1, keepdim=True)
        a = (e * r).sum(-1, keepdim=True) / (r.pow(2).sum(-1, keepdim=True) + eps)
        p = a * r
        sd = 10 * torch.log10((p.pow(2).sum(-1) + eps) / ((e - p).pow(2).sum(-1) + eps))
        return (float(sd.mean()), float(a.mean()))

    def oracle_ladder(run_name, n=8):
        d = ns.RUNS / run_name
        exp = json.loads((d / 'exp.json').read_text())
        h = _H(exp['cfg'])
        G = DualStreamBWE(h)
        G.load_state_dict(torch.load(d / 'ckpt.pt', map_location='cpu')['G'])
        G.eval()
        ds = Dataset(Path(exp['test_list']).read_text().split(), exp['wavs_dir'], h.segment_size, h.hr_sampling_rate, h.lr_sampling_rate, split=False, shuffle=False, n_cache_reuse=0)
        rows = {k: [] for k in ('stft_roundtrip', 'input_anchor', 'pred_mag + true_pha', 'true_mag + pred_pha', 'model')}
        alphas, rms = ([], [])
        with torch.no_grad():
            for i in range(min(n, len(ds))):
                hr, lr = ds[i]
                hr = hr.unsqueeze(0)
                lr = lr.unsqueeze(0)
                mt, pt, _ = amp_pha_stft(hr, h.n_fft, h.hop_size, h.win_size)
                mi, pi, _ = amp_pha_stft(lr, h.n_fft, h.hop_size, h.win_size)
                mg, pg, _ = G(mi, pi)
                I = lambda m, p: amp_pha_istft(m, p, h.n_fft, h.hop_size, h.win_size)
                rows['stft_roundtrip'].append(_sisdr_a(I(mt, pt), hr)[0])
                rows['input_anchor'].append(_sisdr_a(lr, hr)[0])
                rows['pred_mag + true_pha'].append(_sisdr_a(I(mg, pt), hr)[0])
                rows['true_mag + pred_pha'].append(_sisdr_a(I(mt, pg), hr)[0])
                s, a = _sisdr_a(I(mg, pg), hr)
                rows['model'].append(s)
                alphas.append(a)
                est = I(mg, pg)
                rms.append(float(est.pow(2).mean().sqrt() / hr.pow(2).mean().sqrt().clamp(min=1e-09)))
        print(f'\n=== {run_name} ===')
        print(f"{'condition':22s} {'SI-SDR dB':>10s}")
        for k2, v in rows.items():
            print(f'{k2:22s} {np.mean(v):10.2f}')
        print(f'\nmean alpha {np.mean(alphas):.4f}   RMS(est)/RMS(ref) {np.mean(rms):.4f}')
        m, ip, tp = (np.mean(rows['model']), np.mean(rows['pred_mag + true_pha']), np.mean(rows['true_mag + pred_pha']))
        if ip - m > 6 and ip > tp:
            print('>> PHASE is the deficit')
        elif tp - m > 6:
            print('>> MAGNITUDE is the deficit')
        else:
            print('>> both streams are weak; not yet converged')
        if np.mean(rms) < 0.3:
            print('>> output energy is far too low (under-predicted magnitude)')
        return rows
    done = [d.name for d in sorted(ns.RUNS.iterdir()) if (d / 'ckpt.pt').exists()]
    print('checkpoints available:', done)
    pick = [n for n in done if n == ns.REF_RUN]
    pick += [n for n in done if n not in pick][:3]
    for r in pick:
        oracle_ladder(r)


def bridge_step_sweep(ns):
    STEP_GRID = [1, 2, 4, 8]
    DIAG = ns.WORK / 'sb_diag'
    DIAG.mkdir(exist_ok=True)
    diag_rows = []
    for d in sorted(ns.RUNS.iterdir()):
        if not (d / 'ckpt.pt').exists() or not (d / 'exp.json').exists():
            continue
        base = json.loads((d / 'exp.json').read_text())
        for k in STEP_GRID:
            out = DIAG / f'{d.name}_sb{k}'
            out.mkdir(parents=True, exist_ok=True)
            ck = out / 'ckpt.pt'
            if not ck.exists():
                os.symlink(d / 'ckpt.pt', ck)
            exp = dict(base, sb_steps=k, out_dir=str(out), resume=True, minutes_budget=0, val_files=8, stats_files=0, eval_banks=['sinc', 'mixed', 'unseen'], eval_max_recipes=1)
            (out / 'exp.json').write_text(json.dumps(exp, indent=2))
            if (out / 'result.json').exists():
                r = json.loads((out / 'result.json').read_text())
            else:
                T.main(exp)
                if not (out / 'result.json').exists():
                    raise RuntimeError(f'{d.name} sb={k} produced no result')
                r = json.loads((out / 'result.json').read_text())
            diag_rows.append(dict(model=d.name, sb_steps=k, si_sdr=r.get('si_sdr'), lsd_hf=r.get('lsd_hf'), lsd=r.get('lsd'), alpha=r.get('alpha'), rtf=r.get('rtf')))
            print(f"  {d.name:12s} sb={k:<2d} SI-SDR {r.get('si_sdr'):7.2f}  LSD-HF {r.get('lsd_hf'):6.3f}  alpha {r.get('alpha'):5.3f}  RTF {r.get('rtf'):.3f}", flush=True)
    sdf = pd.DataFrame(diag_rows)
    if len(sdf):
        ns.display(sdf.pivot(index='sb_steps', columns='model', values=['si_sdr', 'lsd_hf', 'alpha']).round(3))
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for mm, (met, lbl) in enumerate([('si_sdr', 'SI-SDR dB (higher better)'), ('lsd_hf', 'LSD-HF (lower better)'), ('alpha', 'alpha (1.0 = correct gain)')]):
            for nm, g in sdf.groupby('model'):
                ax[mm].plot(g.sb_steps, g[met], 'o-', label=nm)
            ax[mm].set(xlabel='bridge steps at inference', ylabel=lbl, xscale='log')
            ax[mm].set_xticks(STEP_GRID)
            ax[mm].set_xticklabels(STEP_GRID)
            ax[mm].xaxis.set_minor_locator(plt.NullLocator())
            ax[mm].grid(alpha=0.3)
            ax[mm].legend(fontsize=7)
        ax[2].axhline(1.0, c='k', ls=':', lw=1)
        fig.suptitle('Does iterating the bridge help, or compound a gain bias?', y=1.02)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'bridge_steps_diag.png', dpi=150, bbox_inches='tight')
        plt.close('all')
        _ok = sdf.dropna(subset=['si_sdr'])
        if len(_ok):
            best = _ok.loc[_ok.groupby('model').lsd_hf.idxmin(), ['model', 'sb_steps', 'lsd_hf', 'si_sdr']]
            print('\nbest sb_steps by LSD-HF:')
            print(best.to_string(index=False))
        else:
            print('\nno finite SI-SDR in the sweep - the checkpoint is NaN; the training run did not survive.')
        print('if steps=1 wins, set SB_STEPS = 1 in the grid cell before the next run')
    else:
        print('no checkpoints to diagnose')


STEPS = [training_health, oracle_decomposition, bridge_step_sweep]
