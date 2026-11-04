"""Oracle decomposition and bridge-step sweep on finished runs (needs the checkpoints and the wav directory)."""
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
from bwe_sb import diagnostics as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/4kto16k.json")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--skip", nargs="*", default=[], choices=["health", "oracle", "sweep"])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config key, e.g. vibravox.max_scan=200")
    a = ap.parse_args()
    cfg = C.load(a.config, "short", dict(kv.split("=", 1) for kv in a.set))
    runs = Path(a.runs)
    R = NS.build(cfg, runs, runs.parent, split_dir=a.splits, wav_dir=a.wav_dir)
    if "health" not in a.skip:
        D.training_health(R)
    if "oracle" not in a.skip:
        D.oracle_decomposition(R)
    if "sweep" not in a.skip:
        D.bridge_step_sweep(R)


if __name__ == "__main__":
    main()
