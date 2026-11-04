"""Score the saved runs on another corpus of 16 kHz wavs and plot the generalisation gap."""
import argparse
import sys
from pathlib import Path

import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bwe_sb import config as C
from bwe_sb import namespace as NS
from bwe_sb import report
from bwe_sb import cross_dataset as X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/4kto16k.json")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--corpus", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config key, e.g. vibravox.max_scan=200")
    a = ap.parse_args()
    cfg = C.load(a.config, "short", dict(kv.split("=", 1) for kv in a.set))
    runs = Path(a.runs)
    R = NS.build(cfg, runs, runs.parent)
    R.EXT_DATASETS = {k: Path(v) for k, v in (kv.split("=", 1) for kv in a.corpus)}
    report.results(R)
    for step in X.STEPS:
        step(R)


if __name__ == "__main__":
    main()
