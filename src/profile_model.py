"""Measure model cost independently of training.

    python -m src.profile_model --device cuda    # on Colab
    python -m src.profile_model --device cpu     # on the laptop
Writes results/efficiency_<device>.csv with one row per backbone x resolution:
  params_total, params_trainable_{frozen,partial,full}, size_mb (fp32 state_dict),
  macs_per_clip_g (T=8 frames), latency_ms_median / _p90 (batch = 1 clip).

MACs are counted with torch.utils.flop_counter (FLOPs / 2). It counts
convolutions and matrix multiplications only; element-wise ops (BN,
activations, pooling) are excluded. State this in the paper.
Latency: fp32, eval mode, batch of one clip, `--warmup` untimed iterations,
median and 90th percentile over `--iters` timed iterations. Weights are random
(latency does not depend on their values), so no download is needed.
"""
import argparse
import io
import os
import platform
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.models import build_model, count_params
from src.utils import REPO_ROOT, write_csv

BACKBONES = ["mobilenet_v3_large", "efficientnet_b0", "resnet18"]


def cfg_for(bb, transfer):
    return {"model": {"backbone": bb, "pretrained": False, "transfer": transfer, "head_dropout": 0.0}}


def macs(model, x):
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2


@torch.no_grad()
def latency(model, x, device, warmup, iters):
    times = []
    for i in range(warmup + iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-classes", type=int, default=18)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else (platform.processor() or platform.machine())
    rows = []
    for bb in BACKBONES:
        trainable = {}
        for t in ("frozen", "partial", "full"):
            trainable[t] = count_params(build_model(cfg_for(bb, t), args.n_classes))[1]
        model = build_model(cfg_for(bb, "full"), args.n_classes).eval()
        total = count_params(model)[0]
        buf = io.BytesIO(); torch.save(model.state_dict(), buf)
        size_mb = buf.getbuffer().nbytes / 2**20
        for res in (160, 224):
            x = torch.randn(1, args.frames, 3, res, res)
            m = macs(model, x)
            model_d = model.to(device)
            med, p90 = latency(model_d, x.to(device), device, args.warmup, args.iters)
            model = model_d.cpu()
            rows.append(dict(backbone=bb, resolution=res, params_total=total,
                             params_trainable_frozen=trainable["frozen"],
                             params_trainable_partial=trainable["partial"],
                             params_trainable_full=trainable["full"], size_mb=round(size_mb, 2),
                             macs_per_clip_g=round(m / 1e9, 3), latency_ms_median=round(med, 2),
                             latency_ms_p90=round(p90, 2), device=dev_name, threads=torch.get_num_threads()))
            print(rows[-1], flush=True)
    out = os.path.join(REPO_ROOT, "results", f"efficiency_{device.type}.csv")
    write_csv(rows, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
