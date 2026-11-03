"""Train one configuration and run the loss-curve health check on it."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bwe_sb import config as C
from bwe_sb import train as T
from bwe_sb import diagnostics as D
from bwe_sb import namespace as NS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/4kto16k.json")
    ap.add_argument("--tier", default="short")
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--splits", required=True, help="directory holding training.txt / validation.txt / test.txt")
    ap.add_argument("--work", default="work")
    ap.add_argument("--name", default=None)
    ap.add_argument("--nisqa-dir", default=None, help="a clone of github.com/gabrielmittag/NISQA; omit to skip NISQA")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config key, e.g. loss.be=5")
    a = ap.parse_args()
    cfg = C.load(a.config, a.tier, dict(kv.split("=", 1) for kv in a.set))
    name = a.name or "T_selssm"
    work = Path(a.work)
    runs = work / "runs"
    out = runs / name
    out.mkdir(parents=True, exist_ok=True)
    exp = C.build_exp(cfg, name, out, a.wav_dir, a.splits, a.nisqa_dir)
    (out / "exp.json").write_text(json.dumps(exp, indent=2))
    T.main(exp)
    R = NS.build(cfg, runs, work, split_dir=a.splits, wav_dir=a.wav_dir, nisqa_dir=a.nisqa_dir)
    D.training_health(R)


if __name__ == "__main__":
    main()
