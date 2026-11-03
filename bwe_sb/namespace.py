"""The shared namespace the reporting functions read from and write to."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _display(x):
    if isinstance(x, pd.DataFrame):
        with pd.option_context("display.width", 200, "display.max_columns", 60):
            print(x.to_string())
    else:
        print(x)


def build(cfg, runs_dir, work_dir, figs_dir=None, split_dir=None, wav_dir=None, nisqa_dir=None, root=None):
    runs_dir, work_dir = Path(runs_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    figs = Path(figs_dir) if figs_dir else work_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    runs = sorted(p.name for p in runs_dir.iterdir() if (p / "result.json").exists()) if runs_dir.exists() else []
    t, s, d, b = cfg["task"], cfg["stft"], cfg["degradation"], cfg["bridge"]
    env = {**os.environ, "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return SimpleNamespace(
        WORK=work_dir, RUNS=runs_dir, FIGS=figs, SUBSET=Path(split_dir) if split_dir else work_dir / "splits",
        ROOT=root, REPO=root, ENV=env, VCTK=Path(wav_dir) if wav_dir else None,
        NISQA_DIR=Path(nisqa_dir) if nisqa_dir else root / "NISQA",
        TASK=t["name"], SR=t["hr_sampling_rate"], LR_SR=t["lr_sampling_rate"],
        NFFT=s["n_fft"], HOP=s["hop_size"], WIN=s["win_size"],
        EVAL_BANKS=list(d["eval_banks"]), EXT_BANKS=["sinc", "unseen", "R4"], EXT_DATASETS={}, EXT_MODELS=3, EXT_FILES=150,
        TRUNKS=["selssm"], GRID=[dict(name=r, backbone="selssm") for r in runs],
        REF_RUN=runs[0] if runs else "", STUDY="trunk", AMP_SET=list(cfg["critics"]),
        SB_STEPS=b["sb_steps"], SB_SIGMA=b["sb_sigma"], DRY_RUN=False,
        LOWER={"lsd", "lsd_hf", "apd_ip", "apd_gd", "apd_iaf", "awpd_ip", "awpd_gd", "awpd_iaf"},
        VIBRAVOX=cfg["active"].get("vibravox", True), HF_TOKEN=os.environ.get("HF_TOKEN"),
        VIB_PER_SPEAKER=cfg["vibravox"]["per_speaker"], VIB_MAX_SCAN=cfg["vibravox"]["max_scan"],
        VIB_EVAL_FILES=cfg["vibravox"]["eval_files"], MCDM_BANKS=None,
        display=_display, get_ipython=lambda: None, sys=sys)
