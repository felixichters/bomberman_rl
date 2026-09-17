import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AGENT = "alphabomb"
AGENT_DIR = REPO_ROOT / "agent_code" / AGENT
ARCHIVE = REPO_ROOT / "final-project-agent-code.zip"
PYTHON = sys.executable

EXCLUDE_DIRS = {"__pycache__", "logs", "runs", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".log", ".pyc"}


def promote(model_path):
    source = Path(model_path)
    if not source.is_absolute():
        source = REPO_ROOT / source
    if not source.is_file():
        raise SystemExit(f"no such checkpoint: {source}")
    target = AGENT_DIR / "models" / f"model{source.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    for stale in (AGENT_DIR / "models").glob("model.*"):
        if stale.suffix in (".npz", ".joblib") and stale != target:
            stale.unlink()
    shutil.copyfile(source, target)
    print(f"promoted {source.relative_to(REPO_ROOT)} -> {target.relative_to(REPO_ROOT)}")
    return target


def files_to_ship():
    for path in sorted(AGENT_DIR.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.relative_to(AGENT_DIR).parts):
            continue
        if path.suffix in EXCLUDE_SUFFIXES:
            continue
        yield path


def build():
    members = list(files_to_ship())
    if not any(p.name == "callbacks.py" for p in members):
        raise SystemExit("callbacks.py is missing")
    if not any(p.suffix in (".npz", ".joblib") for p in members):
        raise SystemExit("no trained model in agent_code/%s/models/" % AGENT)

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in members:
            zf.write(path, Path(AGENT) / path.relative_to(AGENT_DIR))

    total = ARCHIVE.stat().st_size
    print(f"\n{ARCHIVE.name}  ({total / 1024:.0f} KiB, {len(members)} files)")
    for path in members:
        print(f"  {path.relative_to(AGENT_DIR)}  ({path.stat().st_size / 1024:.1f} KiB)")
    return ARCHIVE


def verify(archive, rounds=3):
    from tests.test_agent_contract import FRAMEWORK, UPSTREAM

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "clean"
        (root / "agent_code").mkdir(parents=True)
        for name in FRAMEWORK:
            blob = subprocess.run(["git", "show", f"{UPSTREAM}:{name}"], cwd=REPO_ROOT,
                                  capture_output=True, check=True).stdout
            (root / name).write_bytes(blob)
        shutil.copytree(REPO_ROOT / "assets", root / "assets")
        shutil.copytree(REPO_ROOT / "agent_code" / "random_agent", root / "agent_code" / "random_agent")
        (root / "logs").mkdir()

        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root / "agent_code")

        unpacked = next(p.parent for p in (root / "agent_code").rglob("callbacks.py")
                        if p.parent.name == AGENT)
        print(f"\nfirst directory containing callbacks.py: agent_code/{unpacked.name}")

        result = subprocess.run(
            [str(PYTHON), "main.py", "play", "--no-gui", "--n-rounds", str(rounds),
             "--agents", AGENT, "random_agent", "random_agent", "random_agent", "--save-stats"],
            cwd=root, capture_output=True, text=True, timeout=900,
            env={"PATH": "/usr/bin:/bin", "HOME": tmp},
        )
        ok = result.returncode == 0 and "Traceback" not in result.stderr
        print(f"clean-framework run: {'OK' if ok else 'FAILED'} (exit {result.returncode})")
        if not ok:
            print(result.stdout[-3000:])
            print(result.stderr[-3000:])
            return False

        agent_log = (unpacked / "logs" / f"{AGENT}.log").read_text()
        for bad in ("Traceback", "falling back to a safe move"):
            if bad in agent_log:
                print(f"agent log contains {bad!r}")
                return False
        stats = sorted((root / "results").glob("*.json"))
        if stats:
            import json

            by_agent = json.loads(stats[-1].read_text())["by_agent"][AGENT]
            print(f"result over {rounds} rounds: score={by_agent.get('score', 0)} "
                  f"coins={by_agent.get('coins', 0)} kills={by_agent.get('kills', 0)} "
                  f"suicides={by_agent.get('suicides', 0)} "
                  f"mean act {1000 * by_agent['time'] / max(by_agent['steps'], 1):.2f} ms")
        return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="checkpoint to promote before building")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)

    if args.model:
        promote(args.model)

    if not args.skip_tests:
        print("running the contract tests ...")
        code = subprocess.call([str(PYTHON), "-m", "pytest", "tests/test_agent_contract.py", "-q"],
                               cwd=REPO_ROOT)
        if code != 0:
            raise SystemExit("contract tests failed; not building an archive")

    archive = build()
    if not verify(archive, args.rounds):
        raise SystemExit("verification failed")
    print("\nsubmission is ready:", archive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
