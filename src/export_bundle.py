"""Collect everything needed to write the Results sections into one small zip.

    python -m src.export_bundle            # -> paper_bundle.zip in the repo root

Includes: split statistics, frame metadata, configs (incl. C/selection.json),
per-run metrics.json + test_preds.npz + log.csv, summary/seed/leakage/
per-class/bootstrap/efficiency/calibration files, figures, and a manifest
with run counts. Excludes checkpoints and frames.
"""
import glob
import os
import zipfile

from src.utils import REPO_ROOT, load_json, save_json

PATTERNS = [
    "data/index_meta.json", "data/splits/split_summary.json", "data/frame_meta.csv",
    "configs/base.yaml", "configs/*/*.yaml", "configs/C/selection.json",
    "results/*.csv", "results/*.json", "results/*/metrics.json", "results/*/test_preds.npz",
    "results/*/confusion_test.csv", "experiments/*/log.csv",
    "calibration/calibration.json", "figures/*.pdf",
]


def main():
    files = sorted({f for p in PATTERNS for f in glob.glob(os.path.join(REPO_ROOT, p))})
    runs = [load_json(f) for f in files if f.endswith("metrics.json")]
    groups = {}
    for m in runs:
        groups[m["name"][0]] = groups.get(m["name"][0], 0) + 1
    smoke_or_missing = [m["run_id"] for m in runs if m.get("env", {}).get("smoke")]
    manifest = dict(n_runs=len(runs), runs_by_prefix=groups, smoke_runs=smoke_or_missing,
                    devices=sorted({m["env"]["device"] for m in runs}),
                    missing_expected=[p for p in ("results/efficiency_cuda.csv", "results/efficiency_cpu.csv",
                                                  "results/summary.csv", "results/seed_summary.csv",
                                                  "results/leakage.csv", "data/splits/split_summary.json")
                                      if not os.path.exists(os.path.join(REPO_ROOT, p))])
    mpath = os.path.join(REPO_ROOT, "results", "bundle_manifest.json")
    save_json(manifest, mpath)
    files.append(mpath)
    out = os.path.join(REPO_ROOT, "paper_bundle.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(set(files)):
            z.write(f, os.path.relpath(f, REPO_ROOT))
    print(f"wrote {out}: {len(set(files))} files, {os.path.getsize(out) / 2**20:.1f} MB")
    print(manifest)


if __name__ == "__main__":
    main()
