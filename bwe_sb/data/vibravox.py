from __future__ import annotations
"""Download, align and pair the Vibravox in-ear/headset corpus for the cross-sensor test."""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HF_DATASET, HF_CONFIG = "Cnam-LMSSC/vibravox", "speech_clean"
COL_IN, COL_OUT = "audio.soft_in_ear_microphone", "audio.headset_microphone"
SR = 16000

ALIGN_MAX_LAG = 400
ALIGN_MIN_CORR = 0.15

def best_lag(ref: np.ndarray, sig: np.ndarray, max_lag=ALIGN_MAX_LAG
             ) -> Tuple[int, float, int]:
    """Lag of `sig` relative to `ref`, the normalised peak height, and the polarity."""
    n = 1 << int(np.ceil(np.log2(len(ref) + len(sig))))
    a = ref - ref.mean()
    b = sig - sig.mean()
    cc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    cc = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
    i = int(np.argmax(np.abs(cc)))
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    peak = float(cc[i]) / denom
    return i - max_lag, abs(peak), (1 if peak >= 0 else -1)

def align_to(ref: np.ndarray, sig: np.ndarray, max_lag=ALIGN_MAX_LAG,
             min_corr=ALIGN_MIN_CORR) -> Tuple[np.ndarray, int, float, bool]:
    """Shift (and if needed invert) `sig` onto `ref`."""
    lag, r, sign = best_lag(ref, sig, max_lag)
    if r < min_corr:
        return sig, lag, r, False
    if sign < 0:
        sig = -sig
    if lag > 0:
        sig = np.concatenate([np.zeros(lag, sig.dtype), sig[:-lag]])
    elif lag < 0:
        sig = np.concatenate([sig[-lag:], np.zeros(-lag, sig.dtype)])
    return sig, lag, r, True

def residual_lags(lr_dir: Path, hr_dir: Path, ids: List[str], n=40) -> List[int]:
    """Re-measure the lag on the exported files."""
    import soundfile as sf
    out = []
    for fid in ids[:n]:
        a, _ = sf.read(hr_dir / f"{fid}.wav", dtype="float32")
        b, _ = sf.read(lr_dir / f"{fid}.wav", dtype="float32")
        m = min(len(a), len(b))
        out.append(best_lag(a[:m], b[:m])[0])
    return out

def export(split: str, per_speaker: int, lr_dir: Path, hr_dir: Path, list_path: Path,
           max_scan=4500, token: str = None, seed=1234) -> Dict:
    """Stream the split, align each pair, write wavs, and return a report."""
    from datasets import Audio, load_dataset
    import soundfile as sf

    token = token or os.environ.get("HF_TOKEN") or None
    ds = (load_dataset(HF_DATASET, HF_CONFIG, split=split, streaming=True, token=token)
          .select_columns([COL_IN, COL_OUT, "speaker_id"])
          .cast_column(COL_IN, Audio(sampling_rate=SR))
          .cast_column(COL_OUT, Audio(sampling_rate=SR)))

    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    ids, lags, scores, flips, dropped = [], [], [], 0, 0

    for n, row in enumerate(ds):
        if n >= max_scan:
            break
        spk = row["speaker_id"]
        if counts.get(spk, 0) >= per_speaker:
            continue
        hr = np.asarray(row[COL_OUT]["array"], dtype=np.float32)
        lr = np.asarray(row[COL_IN]["array"], dtype=np.float32)
        m = min(len(hr), len(lr))
        if m < SR // 2:
            continue
        hr, lr = hr[:m], lr[:m]
        lag, r, sign = best_lag(hr, lr)
        lr2, lag, r, ok = align_to(hr, lr)
        if not ok:
            dropped += 1
            continue
        if sign < 0:
            flips += 1
        counts[spk] = counts.get(spk, 0) + 1
        fid = f"{spk}_{counts[spk]:03d}"
        sf.write(hr_dir / f"{fid}.wav", hr, SR, "PCM_16")
        sf.write(lr_dir / f"{fid}.wav", lr2, SR, "PCM_16")
        ids.append(fid)
        lags.append(lag)
        scores.append(r)

    random.Random(seed).shuffle(ids)
    list_path.write_text("\n".join(ids) + "\n")
    rep = dict(split=split, files=len(ids), speakers=len(counts),
               lag_median=int(np.median(lags)) if lags else 0,
               lag_min=int(min(lags)) if lags else 0,
               lag_max=int(max(lags)) if lags else 0,
               corr_median=round(float(np.median(scores)), 4) if scores else 0.0,
               polarity_flips=flips, dropped_weak_corr=dropped, scanned=n + 1)
    print(f"{split}: {rep['files']} files / {rep['speakers']} speakers | "
          f"lag median {rep['lag_median']} range [{rep['lag_min']}, {rep['lag_max']}] | "
          f"median |corr| {rep['corr_median']} | {flips} polarity flips | "
          f"{dropped} dropped (weak correlation)", flush=True)
    return rep
