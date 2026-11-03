"""Write training/validation/test id lists for a <speaker>/<file>.wav corpus."""
import argparse
import random
from pathlib import Path


def read_ids(path):
    ids = [l.split("|")[0].strip() for l in Path(path).read_text().split("\n") if l.strip()]
    return [i if "/" in i else f"{i.split('_')[0]}/{i}" for i in ids]


def resolve(wav_dir, i):
    spk = i.split("/")[0]
    if not (wav_dir / spk).is_dir():
        return None
    for cand in (i, f"{i}_mic1", f"{i}_mic2"):
        if (wav_dir / f"{cand}.wav").exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-list", help="official training ids; omit to split speakers at random")
    ap.add_argument("--test-list", help="official test ids")
    ap.add_argument("--train-speakers", type=int, default=100)
    ap.add_argument("--test-speakers", type=int, default=8)
    ap.add_argument("--utts-per-speaker", type=int, default=0, help="0 = all")
    ap.add_argument("--cap", type=int, default=0, help="cap the training list; 0 = all")
    ap.add_argument("--val-holdout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    random.seed(a.seed)
    wav_dir, out = Path(a.wav_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.train_list and a.test_list:
        train = [r for r in map(lambda i: resolve(wav_dir, i), read_ids(a.train_list)) if r]
        test = [r for r in map(lambda i: resolve(wav_dir, i), read_ids(a.test_list)) if r]
    else:
        spk = sorted(p.name for p in wav_dir.iterdir() if p.is_dir())
        random.shuffle(spk)
        tr_spk, te_spk = spk[:a.train_speakers], spk[a.train_speakers:a.train_speakers + a.test_speakers]

        def pick(spks):
            ids = []
            for s in spks:
                w = sorted(p.stem for p in (wav_dir / s).glob("*.wav"))
                random.shuffle(w)
                ids += [f"{s}/{x}" for x in (w[:a.utts_per_speaker] if a.utts_per_speaker else w)]
            return ids
        train, test = pick(tr_spk), pick(te_spk)

    tr_spk = {i.split("/")[0] for i in train}
    te_spk = {i.split("/")[0] for i in test}
    if tr_spk & te_spk:
        raise SystemExit(f"test speakers overlap training: {sorted(tr_spk & te_spk)}")
    random.shuffle(train)
    val = []
    if a.cap and len(train) > a.cap:
        val = train[a.cap:a.cap + a.val_holdout]
        train = train[:a.cap]
    elif a.val_holdout:
        val, train = train[:a.val_holdout], train[a.val_holdout:]
    (out / "training.txt").write_text("\n".join(train) + "\n")
    (out / "validation.txt").write_text("\n".join(val) + "\n")
    (out / "test.txt").write_text("\n".join(test) + "\n")
    print(f"train {len(train)} files / {len(tr_spk)} speakers | validation {len(val)} | test {len(test)} files / {len(te_spk)} speakers (unseen)")


if __name__ == "__main__":
    main()
