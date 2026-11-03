"""Build a synthetic run and render every figure from it; run with python -m tests.test_report."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from bwe_sb import config as C
from tests import fixtures as F


def main():
    cfg = C.load(HERE / "configs" / "4kto16k.json", "short")
    work = Path(tempfile.mkdtemp(prefix="bwesb_report_"))
    runs = work / "runs"
    runs.mkdir()
    for f in ("training.txt", "validation.txt", "test.txt"):
        (work / f).write_text("p225/p225_001\n")
    grid = [dict(name="T_selssm", backbone="selssm", critics=cfg["critics"], bank="mixed", cond=64)]
    F.build(runs, grid, cfg["degradation"]["eval_banks"], C.model_hparams(cfg), eval_max_recipes=4, val_files=4, stats_files=15)
    r = subprocess.run([sys.executable, str(HERE / "scripts" / "plot_results.py"), "--config", str(HERE / "configs" / "4kto16k.json"),
                        "--runs", str(runs), "--splits", str(work)], capture_output=True, text=True)
    tail = [l for l in (r.stdout + r.stderr).split("\n") if "steps ran" in l or l.startswith("  ")]
    print("\n".join(tail[-6:]))
    n = len(list((work / "figures").glob("*.png")))
    ok = r.returncode == 0 and n >= 15
    print(f"{n} figures, exit {r.returncode} -> {'PASS' if ok else 'FAIL'}")
    shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
