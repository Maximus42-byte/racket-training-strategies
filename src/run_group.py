"""Run all configs of a group, one subprocess per config (a crash in one run
does not stop the others). Finished runs are skipped, interrupted runs resume.

    python -m src.run_group --group A
    python -m src.run_group --group A --override run.out_root=/content/drive/MyDrive/badminton data.root=/content/repo
    python -m src.run_group --group B --only mobilenet      # substring filter
"""
import argparse
import glob
import os
import subprocess
import sys
import time

from src.utils import REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--only", default="", help="run only configs whose name contains this")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfgs = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", args.group, "*.yaml")))
    cfgs = [c for c in cfgs if args.only in os.path.basename(c)]
    if not cfgs:
        sys.exit(f"no configs in configs/{args.group}; run `python -m src.make_configs --group {args.group}`")
    failed = []
    t0 = time.time()
    for i, c in enumerate(cfgs, 1):
        print(f"\n=== [{i}/{len(cfgs)}] {os.path.basename(c)} | {(time.time() - t0) / 60:.0f} min elapsed", flush=True)
        cmd = [sys.executable, "-m", "src.train", "--config", c]
        if args.override:
            cmd += ["--override", *args.override]
        if subprocess.call(cmd, cwd=REPO_ROOT) != 0:
            failed.append(c)
    print(f"\ngroup {args.group} finished in {(time.time() - t0) / 60:.0f} min; failed: {len(failed)}")
    for c in failed:
        print("  FAILED", c)


if __name__ == "__main__":
    main()
