"""Time one full epoch (train + validation) of the baseline for each backbone,
then project the cost of the 66-run plan.

    python -m src.calibrate --override run.out_root=/content/drive/MyDrive/badminton data.root=/content/repo

Outputs go to <out_root>/calibration/ so they never enter results/summary.csv.
Projection is an upper bound: frozen/partial runs skip most of the backward
pass and 160 px runs are cheaper, so real totals should be lower.
"""
import argparse
import os
import subprocess
import sys

from src.utils import REPO_ROOT, load_config, load_json, read_csv, save_json

BACKBONES = ["mobilenet_v3_large", "efficientnet_b0", "resnet18"]
RUNS_PER_BACKBONE = 22  # 66 / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--typical-epochs", type=int, default=12)
    args = ap.parse_args()
    base = load_config(os.path.join(REPO_ROOT, "configs", "base.yaml"), args.override)
    out_root = os.path.join(base["run"]["out_root"], "calibration")
    res = {}
    for bb in BACKBONES:
        name = f"calib_{bb}"
        cmd = [sys.executable, "-m", "src.train", "--config", os.path.join(REPO_ROOT, "configs", "base.yaml"),
               "--override", *args.override, f"model.backbone={bb}", "train.max_epochs=1",
               f"run.name={name}", "run.group=calib", f"run.out_root={out_root}"]
        if subprocess.call(cmd, cwd=REPO_ROOT) != 0:
            sys.exit(f"calibration failed for {bb}")
        log = read_csv(os.path.join(out_root, "experiments", f"{name}_s0", "log.csv"))[-1]
        m = load_json(os.path.join(out_root, "results", f"{name}_s0", "metrics.json"))
        res[bb] = dict(epoch_s=float(log["t_epoch_s"]), train_s=float(log["t_train_s"]),
                       peak_gpu_mem_mb=m["peak_gpu_mem_mb"], device=m["env"]["device"])
    for label, ep in (("worst case (20 epochs every run)", 20), (f"typical ({args.typical_epochs} epochs)", args.typical_epochs)):
        hours = sum(v["epoch_s"] for v in res.values()) * RUNS_PER_BACKBONE * ep / 3600
        res[label] = round(hours, 1)
        print(f"projected total for 66 runs, {label}: {hours:.1f} GPU-hours")
    for bb, v in res.items():
        if isinstance(v, dict):
            print(f"  {bb:20s} epoch {v['epoch_s']:.0f}s (train {v['train_s']:.0f}s)  peak mem {v['peak_gpu_mem_mb']} MB")
    save_json(res, os.path.join(out_root, "calibration.json"))


if __name__ == "__main__":
    main()
