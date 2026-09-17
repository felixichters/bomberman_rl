import argparse
import shutil
import sys
from pathlib import Path

MODEL_SUFFIXES = (".npz", ".joblib")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def collect(run_dirs, keep=8, pool=REPO_ROOT / "runs" / "pool"):
    candidates = []
    for raw in run_dirs:
        run = Path(raw)
        if not run.is_absolute():
            run = REPO_ROOT / run
        candidates += [p for p in sorted(run.glob("best.*")) if p.suffix in MODEL_SUFFIXES]
        candidates += [p for p in sorted((run / "checkpoints").glob("round_*"))
                       if p.suffix in MODEL_SUFFIXES]

    if not candidates:
        raise SystemExit("no checkpoints found")

    if len(candidates) > keep:
        step = len(candidates) / keep
        candidates = [candidates[int(i * step)] for i in range(keep)]

    pool.mkdir(parents=True, exist_ok=True)
    for stale in pool.glob("snap_*"):
        stale.unlink()
    for i, source in enumerate(candidates):
        shutil.copyfile(source, pool / f"snap_{i:03d}{source.suffix}")
    print(f"pool of {len(candidates)} snapshots in {pool.relative_to(REPO_ROOT)}:")
    for source in candidates:
        print(f"  {source.relative_to(REPO_ROOT)}")
    return candidates


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--keep", type=int, default=8)
    parser.add_argument("--pool", default=None)
    args = parser.parse_args(argv)
    collect(args.runs, args.keep, Path(args.pool) if args.pool else REPO_ROOT / "runs" / "pool")
    return 0


if __name__ == "__main__":
    sys.exit(main())
