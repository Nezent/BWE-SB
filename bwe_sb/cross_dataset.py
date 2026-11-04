"""Score the saved runs on a second corpus and plot the generalisation gap."""
from pathlib import Path
import json
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shutil
import subprocess
import sys
import time


def find_corpora(ns):
    if not ns.EXT_DATASETS:
        raise SystemExit('no second corpus given: pass --corpus NAME=DIR')
    for k, v in ns.EXT_DATASETS.items():
        if not Path(v).is_dir():
            raise SystemExit(f'corpus {k}: {v} is not a directory')
    print('corpora:', {k: str(v) for k, v in ns.EXT_DATASETS.items()})


def score_runs(ns):
    ns.ext_rows = []
    if ns.EXT_DATASETS:
        want = [n for n in ns.df[ns.df.name == ns.REF_RUN].name]
        want += [n for n in ns.df[ns.df.valid].name.tolist() if n not in want][:ns.EXT_MODELS]
        EXT = ns.WORK / 'ext'
        EXT.mkdir(exist_ok=True)
        for mname in want:
            mp = ns.RUNS / mname
            if not (mp / 'ckpt.pt').exists():
                print('no checkpoint for', mname)
                continue
            for dname, dpath in ns.EXT_DATASETS.items():
                outf = EXT / f'{mname}__{dname}.csv'
                if not outf.exists():
                    cmd = [sys.executable, '-m', 'bwe_sb.evaluate', '--model', str(mp), '--wavs', str(dpath), '--banks', *ns.EXT_BANKS, '--max-files', str(ns.EXT_FILES), '--out', str(outf)]
                    print('=' * 70)
                    print(mname, 'on', dname, flush=True)
                    r = subprocess.run(cmd, cwd=ns.ROOT, env=ns.ENV)
                    if r.returncode != 0:
                        raise RuntimeError(f'{mname} on {dname} failed ({r.returncode})')
                t = pd.read_csv(outf)
                t['dataset'] = dname
                t['model'] = mname
                ns.ext_rows.append(t)
    if ns.ext_rows:
        ns.edf = pd.concat(ns.ext_rows, ignore_index=True)
        ns.edf.to_csv(ns.WORK / 'cross_dataset.csv', index=False)
        mets = [m for m in ('lsd_hf', 'si_sdr', 'pesq', 'stoi', 'mos', 'nisqa_mos') if m in ns.edf.columns]
        print('\n--- cross-dataset results ---')
        ns.display(ns.edf.pivot_table(index=['model'], columns=['dataset', 'bank'], values=mets, aggfunc='mean').round(3))
    else:
        print('nothing evaluated yet')


def gap_figure(ns):
    if ns.ext_rows:
        mets = [m for m in ('lsd_hf', 'si_sdr', 'pesq', 'nisqa_mos', 'stoi') if m in ns.edf.columns and ns.edf[m].notna().any()]
        ds = sorted(ns.edf.dataset.unique())
        fig, axs = plt.subplots(1, len(mets), figsize=(4.0 * len(mets), 4.6), squeeze=False)
        vctk = ns.df.set_index('name')
        for a, m in zip(axs[0], mets):
            piv = ns.edf[ns.edf.bank == 'sinc'].pivot_table(index='model', columns='dataset', values=m, aggfunc='mean')
            piv.insert(0, 'VCTK (in-domain)', [float(vctk.loc[n, m]) if n in vctk.index else np.nan for n in piv.index])
            yy = np.arange(len(piv))
            w = 0.8 / piv.shape[1]
            for j, col in enumerate(piv.columns):
                a.barh(yy + j * w, piv[col].values, w, label=str(col))
            a.set(yticks=yy + 0.4 - w / 2, title=f"{m}\n({('lower' if m in ns.LOWER else 'higher')} better)")
            a.set_yticklabels(piv.index, fontsize=6)
            a.invert_yaxis()
        axs[0][0].legend(fontsize=7)
        fig.suptitle('In-domain VCTK against held-out corpora, clean band-limiting. A small gap is the generalisation claim.', y=1.02, fontsize=10)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'cross_dataset.png', dpi=140)
        plt.close('all')
        fig, axs = plt.subplots(1, len(mets), figsize=(4.0 * len(mets), 4.6), squeeze=False)
        for a, m in zip(axs[0], mets):
            piv = ns.edf.pivot_table(index='model', columns='bank', values=m, aggfunc='mean')
            piv = piv[[b for b in ns.EXT_BANKS if b in piv.columns]]
            yy = np.arange(len(piv))
            w = 0.8 / piv.shape[1]
            for j, col in enumerate(piv.columns):
                a.barh(yy + j * w, piv[col].values, w, label=str(col))
            a.set(yticks=yy + 0.4 - w / 2, title=f'{m} on held-out corpora')
            a.set_yticklabels(piv.index, fontsize=6)
            a.invert_yaxis()
        axs[0][0].legend(fontsize=7)
        fig.suptitle('Unseen corpus AND unseen degradation: the hardest cell in the study', y=1.02, fontsize=10)
        fig.tight_layout()
        fig.savefig(ns.FIGS / 'cross_dataset_degraded.png', dpi=140)
        plt.close('all')


STEPS = [find_corpora, score_runs, gap_figure]
