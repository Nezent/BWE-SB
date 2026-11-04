"""Result tables and figures, one function per figure; run them in order with a shared namespace R."""
from pathlib import Path
from scipy.stats import wilcoxon
import json
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas as pd, numpy as np, json
import re
import shutil
import subprocess
import sys
import time
import torch


def results(ns):
    ns.rows, failed = ([], [])
    for d in sorted(ns.RUNS.iterdir()):
        if (d / 'result.json').exists():
            r = json.loads((d / 'result.json').read_text())
            r['stage'] = r.get('backbone') or r['name']
            ns.rows.append(r)
        elif (d / 'exp.json').exists():
            failed.append(d.name)
    if failed:
        print(f'!! {len(failed)} configuration(s) did not finish:')
        for n in failed[:3]:
            log = ns.RUNS / n / 'train.log'
            if log.exists():
                err = [l for l in log.read_text().strip().split('\n') if l.strip() and (not l.startswith(('  File', '    ', 'Traceback')))]
                print(f"   {n}\n     {(err[-1][:150] if err else '(empty log)')}")
        print()
    if not ns.rows:
        print('No configuration produced result.json.')
        for d in sorted(ns.RUNS.iterdir()):
            if (d / 'ckpt.pt').exists():
                st = torch.load(d / 'ckpt.pt', map_location='cpu', weights_only=False).get('step', '?')
                print(f'  {d.name}: checkpoint at step {st} - re-run the runner cell to finish from there')
        raise SystemExit('re-run section 7, then this cell')
    ns.df = pd.DataFrame(ns.rows)
    ns.df['gen_M'] = (ns.df.gen_params / 1000000.0).round(2)
    ns.df['enc_M'] = (ns.df.get('enc_params', 0) / 1000000.0).round(3)
    ns.df['disc_M'] = (ns.df.disc_params / 1000000.0).round(3)
    ns.METRICS = ['lsd', 'lsd_hf', 'pesq', 'stoi', 'estoi', 'si_sdr', 'snr', 'mos', 'nisqa_mos', 'apd_ip', 'apd_gd', 'apd_iaf', 'alpha']
    ns.METRICS = [m for m in ns.METRICS if m in ns.df.columns]
    ns.LOWER = {'lsd', 'lsd_hf', 'apd_ip', 'apd_gd', 'apd_iaf', 'awpd_ip', 'awpd_gd', 'awpd_iaf'}
    ns.ALPHA_MIN = 0.3
    ns.df['valid'] = ns.df.alpha > ns.ALPHA_MIN if 'alpha' in ns.df else True
    ns.df = ns.df.sort_values(['valid', 'si_sdr'], ascending=[False, False])
    bad = ns.df[~ns.df.valid]
    if len(bad):
        print(f'!! {len(bad)} configuration(s) produced an output uncorrelated with the target (alpha < {ns.ALPHA_MIN}); their LSD is not meaningful:')
        for _, r in bad.iterrows():
            print(f"   {r['name']:42s} alpha={r.alpha:+.4f}  si_sdr={r.si_sdr:+.2f}  lsd_hf={r.lsd_hf:.3f}")
        print()
    show = ['stage', 'name', 'valid', 'gen_M', 'disc_M', 'train_bank', 'conditioned'] + ns.METRICS + ['minutes']
    ns.display(ns.df[[c for c in show if c in ns.df.columns]].reset_index(drop=True).round(4))
    ns.df.to_csv(ns.WORK / 'results.csv', index=False)
    _mb = 'n/a'
    for _d in sorted(ns.RUNS.iterdir()):
        _lg = _d / 'train.log'
        if _lg.exists():
            for _l in _lg.read_text().split('\n'):
                if 'MOS backend:' in _l:
                    _mb = _l.split('MOS backend:')[1].strip()
                    break
        if _mb != 'n/a':
            break
    print('MOS backend:', _mb)
    if 'SQUIM' in _mb:
        print('  (mos = SQUIM proxy on its own scale; nisqa_mos = real NISQA, the column comparable to published tables)')
    bank_rows = []
    for r in ns.rows:
        for bank, vals in (r.get('by_bank') or {}).items():
            _row = dict(name=r['name'], stage=r['stage'], bank=bank, train_bank=r.get('train_bank'))
            _row.update(vals)
            bank_rows.append(_row)
    ns.bdf = pd.DataFrame(bank_rows)
    if len(ns.bdf):
        print('\n--- per evaluation bank (sinc = prior-work setting, unseen = held-out types) ---')
        ns.display(ns.bdf.pivot_table(index='name', columns='bank', values=[m for m in ('lsd_hf', 'pesq', 'stoi', 'si_sdr', 'mos', 'nisqa_mos') if m in ns.bdf], aggfunc='mean').round(3))
        ns.bdf.to_csv(ns.WORK / 'results_by_bank.csv', index=False)


def leaderboard(ns):
    pass
    plt.rcParams.update({'figure.dpi': 110, 'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 9})
    ns.colors = dict(zip(ns.TRUNKS, ['#4878d0', '#ee854a', '#6acc64', '#d65f5f']))
    d = ns.df.sort_values(['valid', 'si_sdr'], ascending=[False, False])
    y = np.arange(len(d))
    ncol = len(ns.METRICS)
    fig, axes = plt.subplots(1, ncol, figsize=(2.9 * ncol, max(5, 0.36 * len(d))), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, ns.METRICS):
        v = d[m].astype(float).values
        cc = d.stage.map(ns.colors).fillna('#bbbbbb').where(d.valid, '#dddddd')
        _bc = ax.barh(y, v, color=cc)
        ax.bar_label(_bc, fmt='%.3f', fontsize=6, padding=2)
        ax.margins(x=0.18)
        if len(d) == 1:
            ax.set_ylim(-1.2, 1.2)
        ax.set(title=f"{m}\n({('lower' if m in ns.LOWER else 'higher')} better)", yticks=y)
        ax.grid(alpha=0.3, axis='x')
        ax.tick_params(labelsize=7)
    axes[0].set_yticklabels(d.name, fontsize=6)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in ns.colors.values()]
    axes[-1].legend(handles, list(ns.colors), fontsize=7, loc='lower right')
    fig.suptitle('Every configuration, every metric', y=1.01)
    fig.tight_layout()
    fig.savefig(ns.FIGS / 'leaderboard_all_metrics.png', dpi=140, bbox_inches='tight')
    plt.close('all')


def robustness_banks(ns):
    if len(ns.bdf):
        ns.order = [b for b in ns.EVAL_BANKS if b in set(ns.bdf.bank)]
        piv = ns.bdf.pivot_table(index='name', columns='bank', values='lsd_hf', aggfunc='mean')[ns.order]
        piv = piv.loc[ns.df.set_index('name').index.intersection(piv.index)]
        fig, ax = plt.subplots(1, 2, figsize=(15, max(4.5, 0.34 * len(piv))))
        yy = np.arange(len(piv))
        w = 0.8 / len(ns.order)
        for j, b in enumerate(ns.order):
            ax[0].barh(yy + j * w, piv[b].values, w, label=b)
        ax[0].set(yticks=yy + 0.4 - w / 2, title='LSD-HF by evaluation bank (lower better)')
        ax[0].set_yticklabels(piv.index, fontsize=6)
        ax[0].invert_yaxis()
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3, axis='x')
        if {'sinc', 'unseen'} <= set(piv.columns):
            gap = (piv['unseen'] - piv['sinc']).sort_values()
            cols = ['#6acc64' if v < 0 else '#d65f5f' for v in gap.values]
            _gb = ax[1].barh(np.arange(len(gap)), gap.values, color=cols)
            ax[1].bar_label(_gb, fmt='%+.3f', fontsize=7, padding=3)
            if len(gap) == 1:
                ax[1].set_ylim(-1.2, 1.2)
            ax[1].axvline(0, c='k', lw=0.8)
            ax[1].set(yticks=np.arange(len(gap)), title='generalisation gap: LSD-HF(unseen) - LSD-HF(sinc)\nleft = degrades less')
            ax[1].set_yticklabels(gap.index, fontsize=6)
            ax[1].invert_yaxis()
            ax[1].grid(alpha=0.3, axis='x')
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'robustness_banks.png', dpi=140, bbox_inches='tight')
        plt.close('all')


def per_recipe_heatmap(ns):
    rec_rows = []
    for r in ns.rows:
        for key, vals in (r.get('by_recipe') or {}).items():
            bank, rec = key.split('|', 1)
            _row = dict(name=r['name'], bank=bank, recipe=rec or 'clean')
            _row.update(vals)
            rec_rows.append(_row)
    ns.rdf = pd.DataFrame(rec_rows)
    if len(ns.rdf):
        ns.rdf.to_csv(ns.WORK / 'results_by_recipe.csv', index=False)
        piv = ns.rdf.pivot_table(index='name', columns='recipe', values='lsd_hf', aggfunc='mean')
        if piv.shape[0] >= 2:
            z = (piv - piv.mean()) / piv.std().replace(0, 1)
            _vmin, _vmax, _lab = (-2, 2, 'z-score within recipe')
            _ttl = 'LSD-HF per degradation recipe, z-scored within recipe (green = better than the other configs on that recipe)'
        else:
            z = piv
            _vmin, _vmax = (float(piv.min().min()), float(piv.max().max()))
            _lab, _ttl = ('LSD-HF (lower better)', 'LSD-HF per degradation recipe (raw)')
        fig, ax = plt.subplots(figsize=(min(26, 1 + 0.5 * piv.shape[1]), max(4, 0.34 * piv.shape[0])))
        im = ax.imshow(z.values, aspect='auto', cmap='RdYlGn_r', vmin=_vmin, vmax=_vmax)
        ax.set(xticks=range(piv.shape[1]), yticks=range(piv.shape[0]))
        ax.set_xticklabels(piv.columns, rotation=90, fontsize=5)
        ax.set_yticklabels(piv.index, fontsize=6)
        ax.set_title(_ttl)
        fig.colorbar(im, ax=ax, shrink=0.7, label=_lab)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'per_recipe_heatmap.png', dpi=140, bbox_inches='tight')
        plt.close('all')


def curves(ns):
    cur = {d.name: pd.read_csv(d / 'curves.csv') for d in sorted(ns.RUNS.iterdir()) if (d / 'curves.csv').exists() and (d / 'result.json').exists()}
    if cur:
        panels = ['loss_g', 'loss_d', 'mag', 'pha', 'com', 'stft', 'adv', 'fm', 'sc', 'be', 'lm', 'bpd', 'hpc', 'desc']
        panels = [p for p in panels if any((p in c.columns for c in cur.values()))]
        fig, axs = plt.subplots(2, 7, figsize=(30, 7.5))
        for a, p in zip(axs.ravel(), panels):
            for n, c in cur.items():
                if p in c.columns and c[p].abs().sum() > 0:
                    a.plot(c['step'], c[p], lw=1, label=n[:22])
            a.set(title=p, xlabel='step')
            a.grid(alpha=0.3)
            if p in ('loss_g', 'mag', 'com', 'stft') and c[p].max() > 0:
                a.set_yscale('log')
        axs.ravel()[0].legend(fontsize=5, ncol=2)
        fig.suptitle('Loss curves (a flat line at zero means that term is disabled for every run)')
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'curves.png', dpi=140, bbox_inches='tight')
        plt.close('all')


def lsd_bands(ns):
    bcols = sorted([c for c in ns.df.columns if c.startswith('lsd_b')], key=lambda c: int(c.split('_b')[1].split('_')[0]))
    if bcols:
        centres = [(int(c.split('_b')[1].split('_')[0]) + int(c.split('_')[-1])) / 2 for c in bcols]
        sel = ns.df[ns.df.valid].head(8)
        _multi = len(ns.df) > 1
        fig, ax = plt.subplots(1, 2 if _multi else 1, figsize=(14 if _multi else 7.5, 5), squeeze=False)
        ax = ax[0]
        for _, r in sel.iterrows():
            st = '--' if _multi and r['name'] == ns.REF_RUN else '-'
            lw = 2.4 if _multi and r['name'] == ns.REF_RUN else 1.4
            ax[0].plot(centres, [r[c] for c in bcols], st, lw=lw, marker='o', ms=3, label=r['name'][:30])
        ax[0].axvline(ns.LR_SR / 2, c='k', ls=':', lw=1)
        ax[0].text(ns.LR_SR / 2 + 60, ax[0].get_ylim()[1] * 0.95, f'input cutoff {ns.LR_SR // 2} Hz', fontsize=7, va='top')
        ax[0].set(xlabel='frequency (Hz)', ylabel='LSD in band', title='Band-wise LSD (lower better)\nleft of the dotted line is HANDED OVER, right must be INVENTED')
        ax[0].legend(fontsize=6)
        base = ns.df[ns.df.name == ns.REF_RUN] if _multi else ns.df.iloc[:0]
        if len(base):
            b = base.iloc[0]
            for _, r in sel.iterrows():
                if r['name'] == b['name']:
                    continue
                ax[1].plot(centres, [r[c] - b[c] for c in bcols], marker='o', ms=3, lw=1.4, label=r['name'][:30])
            ax[1].axhline(0, c='k', lw=1)
            ax[1].axvline(ns.LR_SR / 2, c='k', ls=':', lw=1)
            ax[1].set(xlabel='frequency (Hz)', ylabel=f'LSD - LSD({ns.REF_RUN})', title=f'Improvement over {ns.REF_RUN} per band\nbelow zero = better')
            ax[1].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'lsd_bands.png', dpi=140)
        plt.close('all')
    else:
        print('no band-wise LSD columns - re-run training with the updated metrics module')


def spectrograms(ns):
    ns.samp = {d.name: d / 'samples.pt' for d in sorted(ns.RUNS.iterdir()) if (d / 'samples.pt').exists()}
    UTT = 0
    ZOOM = None
    MAX_MODELS = 6
    BANK = 'sinc'

    def load(n, bank=None):
        b = torch.load(ns.samp[n], map_location='cpu', weights_only=False)
        bank = bank or BANK
        items = b.get('banks', {}).get(bank, {}).get('items') or b['items']
        return (b, items[min(UTT, len(items) - 1)])
    if ns.samp:
        ns.order = [n for n in ns.samp if n == ns.REF_RUN]
        ns.order += [n for n in ns.df[ns.df.valid].name.tolist() if n in ns.samp and n not in ns.order]
        ns.order = ns.order[:MAX_MODELS]
        blob, it0 = load(ns.order[0])
        ns.sr = blob['sr']
        panels = [(f'narrowband input ({BANK})', it0['nb'].numpy())]
        for n in ns.order:
            panels.append((n, load(n)[1]['est'].numpy()))
        panels.append(('natural wideband (reference)', it0['ref'].numpy()))
        ncol = 3
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.1 * nrow), squeeze=False)
        for ax, (ttl, x) in zip(axs.ravel(), panels):
            if ZOOM:
                x = x[int(ZOOM[0] * ns.sr):int(ZOOM[1] * ns.sr)]
            ax.specgram(x, NFFT=512, Fs=ns.sr, noverlap=384, cmap='magma', vmin=-120, vmax=-20)
            ax.axhline(ns.LR_SR / 2, c='cyan', ls=':', lw=0.9)
            ax.set(ylim=(0, ns.sr / 2), title=ttl[:44])
            ax.set_xlabel('time (s)')
            ax.set_ylabel('Hz')
            ax.grid(False)
            ax.title.set_fontsize(8)
        for ax in axs.ravel()[len(panels):]:
            ax.axis('off')
        fig.suptitle(f"Spectrograms, held-out utterance {UTT}, bank '{BANK}', {ns.LR_SR // 1000} kHz -> {ns.sr // 1000} kHz. Dotted line = input cutoff; everything above it is synthesised.", y=1.0, fontsize=10)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'spectrograms.png', dpi=140)
        plt.close('all')

        def logmag(x):
            X = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, 512)[::128] * np.hanning(512), axis=-1))
            return 20 * np.log10(np.maximum(X, 1e-08)).T
        R = logmag(it0['ref'].numpy())
        fig, axs = plt.subplots(1, len(ns.order), figsize=(5.0 * len(ns.order), 3.4), squeeze=False)
        for ax, n in zip(axs[0], ns.order):
            E = logmag(load(n)[1]['est'].numpy())
            t = min(E.shape[1], R.shape[1])
            im = ax.imshow(np.abs(E[:, :t] - R[:, :t]), origin='lower', aspect='auto', cmap='inferno', vmin=0, vmax=40, extent=[0, t * 128 / ns.sr, 0, ns.sr / 2])
            ax.axhline(ns.LR_SR / 2, c='cyan', ls=':', lw=0.9)
            ax.set(title=f'|error| vs reference\n{n[:34]}', xlabel='time (s)', ylabel='Hz')
            ax.title.set_fontsize(8)
            ax.grid(False)
        fig.colorbar(im, ax=axs[0], shrink=0.8, label='dB')
        fig.savefig(ns.FIGS / 'spectrogram_error.png', dpi=140, bbox_inches='tight')
        plt.close('all')
    else:
        print('no samples.pt - re-run training with the updated trainer')


def radar(ns):
    RAD = [m for m in ('lsd_hf', 'pesq', 'stoi', 'estoi', 'si_sdr', 'mos', 'nisqa_mos', 'apd_ip', 'awpd_ip') if m in ns.df.columns]
    sel = ns.df[ns.df.valid].head(5)
    base = ns.df[ns.df.name == ns.REF_RUN] if len(ns.df) > 1 else ns.df.iloc[:0]
    sel = pd.concat([base, sel]).drop_duplicates('name')
    if len(sel) > 1 and RAD:
        norm = {}
        for m in RAD:
            v = sel[m].astype(float)
            lo, hi = (v.min(), v.max())
            rng = hi - lo or 1.0
            norm[m] = (hi - v) / rng if m in ns.LOWER else (v - lo) / rng
        ang = np.linspace(0, 2 * np.pi, len(RAD), endpoint=False).tolist()
        ang += ang[:1]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for i, (_, r) in enumerate(sel.iterrows()):
            v = [float(norm[m].loc[r.name]) for m in RAD]
            v += v[:1]
            dashed = r['name'] == ns.REF_RUN
            ax.plot(ang, v, lw=2.4 if dashed else 1.6, ls='--' if dashed else '-', label=r['name'][:30], color='k' if dashed else None)
            ax.fill(ang, v, alpha=0.08)
        ax.set(xticks=ang[:-1], ylim=(0, 1.05))
        ax.set_xticklabels([m + ('\n(lower better)' if m in ns.LOWER else '') for m in RAD], fontsize=7)
        ax.set_yticklabels([])
        ax.set_title('Every metric, rescaled so outward is always better\nblack dashed = reference arm baseline', fontsize=9)
        ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.32, 1.12))
        fig.savefig(ns.FIGS / 'radar.png', dpi=140)
        plt.close('all')


def robustness_curves(ns):
    if len(ns.rdf):

        def grab(rec, key):
            m = re.search(f'(?:^|_){key}=(-?[\\d.]+)', rec)
            return float(m.group(1)) if m else None
        ns.rdf['snr_db'] = ns.rdf.recipe.map(lambda r: grab(r, 'snr'))
        ns.rdf['nyq_hz'] = ns.rdf.recipe.map(lambda r: (grab(r, 'cutoff') or 0) / 2 or None)
        ns.rdf['pure_cut'] = ns.rdf.recipe.map(lambda r: re.fullmatch('cutoff=[\\d.]+', r) is not None)
        sel = set(ns.df[ns.df.valid].name.head(6)) | set(ns.df[ns.df.name == ns.REF_RUN].name)
        snr_sub = ns.rdf[(ns.rdf.bank == 'aurora') & ns.rdf.snr_db.notna() & ns.rdf.name.isin(sel) & ns.rdf.recipe.str.contains('noise=awgn')]
        cut_sub = ns.rdf[ns.rdf.pure_cut & ns.rdf.nyq_hz.notna() & ns.rdf.name.isin(sel)]
        plots = [(snr_sub, 'snr_db', f'input SNR (dB), cutoff fixed at {ns.LR_SR / 2000:g} kHz, AWGN'), (cut_sub, 'nyq_hz', 'input bandwidth (Hz), no other degradation')]
        mets = [m for m in ('lsd_hf', 'si_sdr', 'pesq', 'nisqa_mos', 'stoi') if m in ns.rdf.columns]
        fig, axs = plt.subplots(len(plots), len(mets), figsize=(4 * len(mets), 3.8 * len(plots)), squeeze=False)
        for ri, (sub, xk, xl) in enumerate(plots):
            for ci, m in enumerate(mets):
                a = axs[ri][ci]
                for n, g in sub.groupby('name'):
                    gg = g.groupby(xk)[m].mean().sort_index()
                    a.plot(gg.index, gg.values, marker='o', ms=3, lw=1.6, label=n[:26])
                if f'in_{m}' in sub.columns:
                    gi = sub.groupby(xk)[f'in_{m}'].mean().sort_index()
                    a.plot(gi.index, gi.values, marker='s', ms=3, lw=1.4, ls=':', color='0.4', label='unprocessed input')
                a.set(xlabel=xl, ylabel=m, title=f"{m} ({('lower' if m in ns.LOWER else 'higher')} better)")
                a.grid(alpha=0.3)
                if ci == 0:
                    a.legend(fontsize=6)
        fig.suptitle('Degradation sweeps on single-factor recipes. What matters is the gap above the unprocessed input.', y=1.01, fontsize=10)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'robustness_curves.png', dpi=140)
        plt.close('all')


def phase_metrics(ns):
    APD = [m for m in ('apd_ip', 'apd_gd', 'apd_iaf', 'awpd_ip', 'awpd_gd', 'awpd_iaf') if m in ns.df.columns]
    if APD:
        d = ns.df.sort_values('apd_ip')
        y = np.arange(len(d))
        fig, axs = plt.subplots(1, len(APD) + 1, figsize=(4.2 * (len(APD) + 1), max(4.5, 0.34 * len(d))), sharey=True)
        for a, m in zip(axs, APD):
            cc = d.backbone.map(ns.colors).fillna('#bbb').where(d.valid, '#ddd')
            _bc = a.barh(y, d[m].astype(float).values, color=cc)
            a.bar_label(_bc, fmt='%.3f', fontsize=6, padding=2)
            a.margins(x=0.18)
            if len(d) == 1:
                a.set_ylim(-1.2, 1.2)
            a.set(title=f'{m} (lower better)', yticks=y)
            a.tick_params(labelsize=7)
        a = axs[-1]
        _ba = a.barh(y, d.alpha.astype(float).values, color=['#6acc64' if v > ns.ALPHA_MIN else '#d65f5f' for v in d.alpha])
        a.bar_label(_ba, fmt='%.3f', fontsize=6, padding=2)
        a.margins(x=0.18)
        if len(d) == 1:
            a.set_ylim(-1.2, 1.2)
        a.axvline(ns.ALPHA_MIN, c='k', ls='--', lw=0.9)
        a.set(title=f'alpha (SI-SDR optimal scale)\nred = below {ns.ALPHA_MIN}: output uncorrelated')
        axs[0].set_yticklabels(d.name, fontsize=6)
        axs[0].invert_yaxis()
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'phase_metrics.png', dpi=140)
        plt.close('all')


def channel_ladder(ns):
    TIERS = [f'R{i}' for i in range(5)]
    tb = ns.bdf[ns.bdf.bank.isin(TIERS)] if len(ns.bdf) else ns.bdf
    if len(tb):
        mets = [m for m in ('lsd_hf', 'si_sdr', 'pesq', 'stoi', 'mos', 'nisqa_mos') if m in tb.columns]
        piv = tb.pivot_table(index='name', columns='bank', values=mets, aggfunc='mean')
        ns.display(piv.round(3))
        piv.to_csv(ns.WORK / 'channel_ladder.csv')
        sel = set(ns.df[ns.df.valid].name.head(6)) | set(ns.df[ns.df.name == ns.REF_RUN].name)
        _multi = len(sel) > 1
        _refnote = 'Black dashed = reference arm. ' if _multi else ''
        fig, axs = plt.subplots(1, len(mets), figsize=(3.9 * len(mets), 4.2), squeeze=False)
        for a, m in zip(axs[0], mets):
            for n, g in tb[tb.name.isin(sel)].groupby('name'):
                gg = g.set_index('bank')[m].reindex(TIERS)
                dashed = _multi and n == ns.REF_RUN
                a.plot(TIERS, gg.values, marker='o', ms=4, lw=2.4 if dashed else 1.3, ls='--' if dashed else '-', color='k' if dashed else None, label=n[:26])
            a.set(xlabel='channel tier', ylabel=m, title=f"{m} across R0-R4\n({('lower' if m in ns.LOWER else 'higher')} better)")
        axs[0][0].legend(fontsize=6)
        fig.suptitle(f'Channel-degradation ladder. {_refnote}A flatter line is a more robust model.', y=1.02, fontsize=10)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'channel_ladder.png', dpi=140)
        plt.close('all')
    else:
        print('no R0-R4 rows - add the tiers to EVAL_BANKS and re-run')


def aurora_snr(ns):
    au = ns.rdf[ns.rdf.bank == 'aurora'].copy() if len(ns.rdf) else ns.rdf
    if len(au):
        au['snr_db'] = au.recipe.str.extract('snr=(-?\\d+)').astype(float)
        au['noise'] = au.recipe.str.extract('noise=(\\w+)')
        mets = [m for m in ('lsd_hf', 'si_sdr', 'pesq', 'stoi', 'mos', 'nisqa_mos') if m in au.columns]
        print('averaged over both noise types and all five SNR levels:')
        ns.display(au.pivot_table(index='name', values=mets, aggfunc='mean').round(3))
        sel = set(ns.df[ns.df.valid].name.head(5)) | set(ns.df[ns.df.name == ns.REF_RUN].name)
        _multi = len(sel) > 1
        _refnote = f' Black dashed = reference arm ({ns.REF_RUN}).' if _multi else ''
        nz = sorted(au.noise.dropna().unique())
        fig, axs = plt.subplots(len(nz), len(mets), figsize=(3.7 * len(mets), 3.6 * len(nz)), squeeze=False)
        for ri, z in enumerate(nz):
            for ci, m in enumerate(mets):
                a = axs[ri][ci]
                for n, g in au[(au.noise == z) & au.name.isin(sel)].groupby('name'):
                    gg = g.groupby('snr_db')[m].mean().sort_index()
                    dashed = _multi and n == ns.REF_RUN
                    a.plot(gg.index, gg.values, marker='o', ms=4, lw=2.4 if dashed else 1.3, ls='--' if dashed else '-', color='k' if dashed else None, label=n[:24])
                a.set(xlabel='input SNR (dB)', ylabel=m, title=f'{m} - {z}')
                if ri == 0 and ci == 0:
                    a.legend(fontsize=6)
        fig.suptitle(f'Noisy protocol.{_refnote}', y=1.01, fontsize=10)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'aurora_snr.png', dpi=140)
        plt.close('all')


def itu_metrics(ns):
    ITU = [c for c in ('d_sf', 'd_loud', 'd_miss', 'd_sf_top', 'd_miss_top') if c in ns.df.columns]
    if ITU:
        t = ns.df[['name', 'valid'] + ITU].copy()
        ns.display(t.set_index('name').round(4))
        t.to_csv(ns.WORK / 'itu_metrics.csv', index=False)
        base = ns.df[ns.df.name == ns.REF_RUN] if len(ns.df) > 1 else ns.df.iloc[:0]
        d = ns.df[ns.df.valid]
        fig, axs = plt.subplots(1, len(ITU), figsize=(3.4 * len(ITU), max(4, 0.3 * len(d))), sharey=True)
        axs = np.atleast_1d(axs)
        y = np.arange(len(d))
        for a, m in zip(axs, ITU):
            _bc = a.barh(y, d[m].astype(float).values, color=d.backbone.map(ns.colors).fillna('#bbb'))
            a.bar_label(_bc, fmt='%.3f', fontsize=6, padding=2)
            a.margins(x=0.18)
            if len(d) == 1:
                a.set_ylim(-1.2, 1.2)
            _notes = []
            if len(base):
                a.axvline(float(base.iloc[0][m]), c='k', ls='--', lw=1)
                _notes.append('dashed = reference arm')
            if f'in_{m}' in d.columns and d[f'in_{m}'].notna().any():
                _iv = float(d[f'in_{m}'].astype(float).mean())
                a.axvline(_iv, c='0.4', ls=':', lw=1.4)
                a.text(_iv, 1.05, f'input {_iv:.2f}', fontsize=6, color='0.3', ha='center')
                _notes.append('dotted = unprocessed input')
            _ref = '; ' + ', '.join(_notes) if _notes else ''
            a.set(title=f'{m}\n(lower better{_ref})', yticks=y)
            a.tick_params(labelsize=7)
        axs[0].set_yticklabels(d.name, fontsize=6)
        axs[0].invert_yaxis()
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'itu_metrics.png', dpi=140)
        plt.close('all')
    else:
        print('no ITU columns - re-run training with the updated metrics module')


def stats_ci(ns):
    PU = {d.name: pd.read_csv(d / 'per_utt.csv') for d in sorted(ns.RUNS.iterdir()) if (d / 'per_utt.csv').exists()}
    if PU:
        prim = ns.EVAL_BANKS[0]
        STAT = [m for m in ('lsd_hf', 'si_sdr', 'pesq', 'stoi', 'estoi', 'mos', 'nisqa_mos') if m in next(iter(PU.values())).columns]
        desc = []
        for n, u in PU.items():
            g = u[u.bank == prim]
            for m in STAT:
                v = g[m].dropna()
                if not len(v):
                    continue
                sd = v.std(ddof=1)
                sem = sd / np.sqrt(len(v))
                desc.append(dict(name=n, metric=m, n=len(v), mean=v.mean(), sd=sd, sem=sem, ci_lo=v.mean() - 1.96 * sem, ci_hi=v.mean() + 1.96 * sem))
        ddf = pd.DataFrame(desc)
        print(f"--- descriptive statistics on the '{prim}' bank ---")
        ns.display(ddf.pivot_table(index='name', columns='metric', values=['mean', 'sd', 'sem']).round(4))
        ddf.to_csv(ns.WORK / 'stats_descriptive.csv', index=False)
        ref = [n for n in PU if n == ns.REF_RUN]
        if ref:
            rname = ref[0]
            R = PU[rname]
            key = ['bank', 'recipe_idx', 'utt']
            tests = []
            for n, u in PU.items():
                if n == rname:
                    continue
                j = R[key + STAT].merge(u[key + STAT], on=key, suffixes=('_ap', '_ours'))
                j = j[j.bank == prim]
                for m in STAT:
                    a, b = (j[f'{m}_ap'], j[f'{m}_ours'])
                    ok = a.notna() & b.notna()
                    if ok.sum() < 10 or (b[ok] - a[ok]).abs().sum() == 0:
                        continue
                    st, pv = wilcoxon(b[ok], a[ok])
                    dmean = float((b[ok] - a[ok]).mean())
                    better = dmean < 0 if m in ns.LOWER else dmean > 0
                    tests.append(dict(name=n, metric=m, n=int(ok.sum()), mean_delta=dmean, median_delta=float((b[ok] - a[ok]).median()), W=float(st), p=float(pv), verdict=('better' if better else 'worse') + ('' if pv < 0.05 else ' (not significant)')))
            tdf = pd.DataFrame(tests)
            if not len(tdf):
                print(f"\n[skipped] no Wilcoxon test: fewer than 10 paired utterances per metric on '{prim}'. STATS_FILES={len(next(iter(PU.values())))} rows - raise STATS_FILES (tiny uses 4, short uses 150).")
            if len(tdf):
                print(f'\n--- paired Wilcoxon signed-rank vs {rname}, per utterance ---')
                ns.display(tdf.set_index(['name', 'metric']).round(6))
                tdf.to_csv(ns.WORK / 'stats_wilcoxon.csv', index=False)
                best = tdf.groupby('name')['p'].min().sort_values().index[:6]
                sub = ddf[ddf['name'].isin(set(best) | {rname})]
                mm = [m for m in STAT if m in set(sub['metric'])]
                fig, axs = plt.subplots(1, len(mm), figsize=(3.3 * len(mm), 4.6), squeeze=False)
                for a, m in zip(axs[0], mm):
                    g = sub[sub['metric'] == m].sort_values('mean')
                    yy = np.arange(len(g))
                    a.errorbar(g['mean'], yy, xerr=1.96 * g['sem'], fmt='o', ms=5, capsize=3, color='#4878d0')
                    for i, (_, r) in enumerate(g.iterrows()):
                        if r['name'] == rname:
                            a.errorbar([r['mean']], [i], xerr=[1.96 * r['sem']], fmt='s', ms=7, capsize=3, color='k')
                    a.set(yticks=yy, title=f'{m}\n95% CI')
                    a.set_yticklabels(g['name'], fontsize=6)
                fig.suptitle('Mean with 95% confidence interval. Black square = reference arm. Non-overlapping intervals are the claim.', y=1.02, fontsize=10)
                fig.tight_layout()
                fig.savefig(ns.FIGS / 'stats_ci.png', dpi=140)
                plt.close('all')
    else:
        print('[skipped] no per_utt.csv found - the final evaluation writes it')


def complexity(ns):
    CX = [c for c in ('macs_g', 'flops_g', 'infer_ms', 'rtf') if c in ns.df.columns]
    t = ns.df[['name'] + CX].copy()
    t = t.rename(columns={'macs_g': 'GMACs', 'flops_g': 'GFLOPs', 'infer_ms': 'inference (ms/s audio)', 'rtf': 'RTF'})
    ns.display(t.set_index('name').round(4))
    t.to_csv(ns.WORK / 'complexity.csv', index=False)
    if 'rtf' in ns.df.columns and ns.df.rtf.notna().any():
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
        m2 = 'pesq' if 'pesq' in ns.df else 'si_sdr'
        ax.scatter(ns.df.rtf.astype(float), ns.df[m2].astype(float), s=70, c=ns.df.backbone.map(ns.colors).fillna('#bbb'))
        for _, r in ns.df.iterrows():
            ax.annotate(r['name'][:22], (float(r.rtf), float(r[m2])), fontsize=5, xytext=(3, 3), textcoords='offset points')
        ax.axvline(1.0, c='r', ls='--', lw=1)
        ax.set(xlabel='RTF (1 = real time)', ylabel=m2, title=f'{m2} against inference cost')
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'complexity.png', dpi=140)
        plt.close('all')
    else:
        print('RTF not recorded (install thop for MACs: pip install thop)')


def amplitude_spectrograms(ns):
    ns.samp = {d.name: d / 'samples.pt' for d in sorted(ns.RUNS.iterdir()) if (d / 'samples.pt').exists()}
    AMP_UTT, AMP_BANK, AMP_MAX = (0, 'sinc', 4)

    def _spec(x, n=512, hop=128):
        w = np.hanning(n)
        fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * w
        return 20 * np.log10(np.abs(np.fft.rfft(fr, axis=-1)).T + 1e-08)
    ns._spec = _spec
    if ns.samp:
        ns.order = [n for n in ns.samp if n == ns.REF_RUN][:1]
        ns.order += [n for n in ns.df[ns.df.valid].name.tolist() if n in ns.samp and n not in ns.order][:AMP_MAX - 1]

        def _load(n):
            b = torch.load(ns.samp[n], map_location='cpu', weights_only=False)
            it = (b.get('banks', {}).get(AMP_BANK, {}) or {}).get('items') or b['items']
            return (b, it[min(AMP_UTT, len(it) - 1)])
        ns._load = _load
        b0, it0 = _load(ns.order[0])
        ns.sr = b0['sr']
        ns.REF = _spec(it0['ref'].numpy())
        ns.NB_ = _spec(it0['nb'].numpy())
        ns.cut_row = int(ns.LR_SR / 2 / (ns.sr / 2) * ns.REF.shape[0])
        ext = [0, it0['ref'].shape[-1] / ns.sr, 0, ns.sr / 2]
        fig, axs = plt.subplots(2, len(ns.order) + 2, figsize=(3.5 * (len(ns.order) + 2), 7.2), squeeze=False, sharex=True, sharey=True, gridspec_kw=dict(hspace=0.32, wspace=0.12))
        panels = [('input (narrowband)', ns.NB_)] + [(n[:22], _spec(_load(n)[1]['est'].numpy())) for n in ns.order] + [('reference', ns.REF)]
        for col, (ttl, S) in enumerate(panels):
            a = axs[0][col]
            a.imshow(S, origin='lower', aspect='auto', cmap='magma', vmin=-100, vmax=0, extent=ext)
            a.axhline(ns.LR_SR / 2, c='cyan', ls=':', lw=0.9)
            a.set(title=ttl)
            a.set_ylabel('Hz' if col == 0 else '')
            a.title.set_fontsize(8)
            a.grid(False)
            b_ = axs[1][col]
            t = min(S.shape[1], ns.REF.shape[1])
            D = S[:, :t] - ns.REF[:, :t]
            im = b_.imshow(D, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-25, vmax=25, extent=ext)
            b_.axhline(ns.LR_SR / 2, c='k', ls=':', lw=0.9)
            b_.set(title='error vs reference (dB)', xlabel='time (s)')
            b_.set_ylabel('Hz' if col == 0 else '')
            b_.title.set_fontsize(8)
            b_.grid(False)
        axs[1][-1].set_visible(False)
        fig.colorbar(im, ax=axs[1], shrink=0.85, pad=0.015, label='dB (red = over-filled, blue = under-filled)')
        fig.suptitle('Amplitude: spectrograms (top) and signed error against the reference (bottom). Dotted line = input cutoff.', y=0.98, fontsize=10)
        fig.savefig(ns.FIGS / 'amplitude_spectrograms.png', dpi=140, bbox_inches='tight')
        plt.close('all')
    else:
        print('no samples.pt - run training first')


def amplitude_ltas(ns):
    if ns.samp:
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.4))
        f_ax = np.linspace(0, ns.sr / 2, ns.REF.shape[0])
        axs[0].plot(f_ax, ns.REF.mean(1), 'k', lw=2, label='reference')
        axs[0].plot(f_ax, ns.NB_.mean(1), color='0.6', ls=':', lw=1.6, label='input')
        for n in ns.order:
            axs[0].plot(f_ax, ns._spec(ns._load(n)[1]['est'].numpy()).mean(1), lw=1.2, label=n[:22])
        axs[0].axvline(ns.LR_SR / 2, c='k', ls=':', lw=1)
        axs[0].set(xlabel='frequency (Hz)', ylabel='dB', title='long-term average spectrum\n(does the filled band have the right shape?)')
        axs[0].legend(fontsize=6)
        hb = slice(ns.cut_row, ns.REF.shape[0])
        tt = np.arange(ns.REF.shape[1]) * 128 / ns.sr
        axs[1].plot(tt, ns.REF[hb].mean(0), 'k', lw=2, label='reference')
        for n in ns.order:
            S = ns._spec(ns._load(n)[1]['est'].numpy())
            t = min(S.shape[1], len(tt))
            axs[1].plot(tt[:t], S[hb, :t].mean(0), lw=1.2, label=n[:22])
        axs[1].set(xlabel='time (s)', ylabel='dB', title=f'high-band energy over time (>{ns.LR_SR // 2} Hz)\n(does it track the speech, or sit flat?)')
        axs[1].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'amplitude_ltas.png', dpi=140)
        plt.close('all')


def amp_vs_phase(ns):
    if len(ns.df) > 1 and {'lsd_hf', 'apd_ip'} <= set(ns.df.columns):
        ref = ns.df[ns.df.name == ns.REF_RUN] if len(ns.df) > 1 else ns.df.iloc[:0]
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.6))
        for a, (xm, ym) in zip(axs, [('lsd_hf', 'apd_ip'), ('si_sdr', 'alpha')]):
            for st, gg in ns.df.groupby('backbone'):
                a.scatter(gg[xm].astype(float), gg[ym].astype(float), s=70, label=st, color=ns.colors.get(st, '#bbb'))
            for _, r in ns.df.iterrows():
                a.annotate(r['name'].split('_')[1][:12], (float(r[xm]), float(r[ym])), fontsize=5, xytext=(3, 3), textcoords='offset points')
            if len(ref):
                a.axvline(float(ref.iloc[0][xm]), ls='--', c='k', lw=0.8)
                a.axhline(float(ref.iloc[0][ym]), ls='--', c='k', lw=0.8)
            a.set(xlabel=f'{xm} (amplitude)' if xm != 'si_sdr' else xm, ylabel=f'{ym} (phase)' if ym == 'apd_ip' else ym, title=f'{ym} vs {xm}\ndashed = reference')
            a.legend(fontsize=7)
        fig.suptitle('Amplitude gain vs phase cost', y=1.01)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'amp_vs_phase.png', dpi=140)
        plt.close('all')


STEPS = [results, leaderboard, robustness_banks, per_recipe_heatmap, curves, lsd_bands, spectrograms, radar, robustness_curves, phase_metrics, channel_ladder, aurora_snr, itu_metrics, stats_ci, complexity, amplitude_spectrograms, amplitude_ltas, amp_vs_phase]