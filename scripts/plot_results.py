"""Build every table and figure from a directory of finished runs."""
import argparse
import sys
from pathlib import Path

import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bwe_sb import config as C
from bwe_sb import namespace as NS
from bwe_sb import report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/4kto16k.json")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=None, help="figure directory; default <runs>/../figures")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="subset of step names to run")
    a = ap.parse_args()
    cfg = C.load(a.config, "short")
    runs = Path(a.runs)
    R = NS.build(cfg, runs, runs.parent, figs_dir=a.out, split_dir=a.splits or runs.parent)
    done, failed = [], []
    for step in report.STEPS:
        if a.only and step.__name__ not in a.only:
            continue
        try:
            step(R)
            done.append(step.__name__)
        except Exception as e:
            failed.append((step.__name__, f"{type(e).__name__}: {str(e)[:100]}"))
        plt.close("all")
    print(f"\n{len(done)} steps ran, {len(failed)} failed -> {R.FIGS} ({len(list(R.FIGS.glob('*.png')))} png)")
    for n, e in failed:
        print(f"  {n}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
