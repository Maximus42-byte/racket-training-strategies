"""Result figures for the paper (Springer: 84 mm column / 174 mm full width,
sans-serif lettering, readable in grayscale: shapes and line styles, not colour).

    python -m src.make_figures            # writes figures/Fig2.pdf ... Fig5.pdf

Fig2  row-normalised test confusion matrix of the optimised pipeline whose
      seed-0 VALIDATION macro-F1 is highest (selection never uses test data)
Fig3  test macro-F1 vs total training time, all seed-0 session-split runs
Fig4  test macro-F1 vs MACs per clip, baseline and optimised per backbone
      (needs results/efficiency_cuda.csv from src.profile_model)
Fig5  validation macro-F1 per epoch for the four transfer strategies (group A)
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from src.utils import REPO_ROOT, load_json, read_csv

MM = 1 / 25.4
COL, FULL = 84 * MM, 174 * MM
BB = ["mobilenet_v3_large", "efficientnet_b0", "resnet18"]
BB_LABEL = {"mobilenet_v3_large": "MobileNetV3-L", "efficientnet_b0": "EfficientNet-B0", "resnet18": "ResNet-18"}
MARK = {"mobilenet_v3_large": "o", "efficientnet_b0": "s", "resnet18": "^"}
LS = {"scratch": ":", "frozen": "--", "partial": "-.", "full": "-"}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42, "axes.linewidth": 0.6})


def load_runs(root):
    return {m["run_id"]: m for m in (load_json(p) for p in glob.glob(os.path.join(root, "results", "*", "metrics.json")))}


def save(fig, root, name):
    os.makedirs(os.path.join(root, "figures"), exist_ok=True)
    fig.savefig(os.path.join(root, "figures", f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"figures/{name}.pdf")


def fig2(runs, root):
    cands = [m for m in runs.values() if m["name"].endswith("_opt") and m["seed"] == 0 and m["split"] == "session_split"]
    if not cands:
        print("Fig2 skipped: no C_*_opt_s0 results"); return
    m = max(cands, key=lambda r: r["val"]["macro_f1"])
    classes = load_json(os.path.join(root, "data", "index_meta.json"))["classes"]
    cm = np.array(m["test"]["confusion_matrix"], dtype=float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(FULL * 0.62, FULL * 0.55))
    im = ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    short = [c.replace("_", " ") for c in classes]
    ax.set_xticklabels(short, rotation=90); ax.set_yticklabels(short)
    ax.set_xlabel("Predicted stroke"); ax.set_ylabel("True stroke")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cmn[i, j] >= 0.05:
                ax.text(j, i, f"{cmn[i, j]:.2f}"[1:] if cmn[i, j] < 1 else "1", ha="center", va="center",
                        fontsize=4.5, color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("Row-normalised rate")
    ax.set_title(f"{BB_LABEL[m['backbone']]}, optimised pipeline (test)")
    save(fig, root, "Fig2")


def fig3(runs, root):
    rs = [m for m in runs.values() if m["seed"] == 0 and m["split"] == "session_split" and m["group"] in "ABC"]
    if not rs:
        print("Fig3 skipped"); return
    fig, ax = plt.subplots(figsize=(COL, COL * 0.75))
    fills = {"A": "none", "B": "0.6", "C": "black"}
    for m in rs:
        ax.scatter(m["train_time_s"] / 60, m["test"]["macro_f1"], marker=MARK[m["backbone"]],
                   facecolors=fills.get(m["group"], "none"), edgecolors="black", linewidths=0.6, s=18)
    for bb in BB:
        ax.scatter([], [], marker=MARK[bb], facecolors="none", edgecolors="black", label=BB_LABEL[bb])
    for g, lab in (("A", "A: transfer"), ("B", "B: single change"), ("C", "C: optimised/ablation")):
        ax.scatter([], [], marker="o", facecolors=fills[g], edgecolors="black", label=lab)
    ax.set_xlabel("Total training time (min)"); ax.set_ylabel("Test macro-F1")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.grid(True, lw=0.3, color="0.85")
    save(fig, root, "Fig3")


def fig4(runs, root):
    p = os.path.join(root, "results", "efficiency_cuda.csv")
    if not os.path.exists(p):
        print("Fig4 skipped: results/efficiency_cuda.csv missing"); return
    eff = {(r["backbone"], int(r["resolution"])): float(r["macs_per_clip_g"]) for r in read_csv(p)}
    fig, ax = plt.subplots(figsize=(COL, COL * 0.75))
    for bb in BB:
        pts = []
        for name, fill in ((f"A_{bb}_full_s0", "none"), (f"C_{bb}_opt_s0", "black")):
            if name in runs:
                m = runs[name]
                x = eff[(bb, int(m["resolution"]))]
                ax.scatter(x, m["test"]["macro_f1"], marker=MARK[bb], facecolors=fill, edgecolors="black", s=22, lw=0.6)
                pts.append((x, m["test"]["macro_f1"]))
        if len(pts) == 2:
            ax.annotate("", xy=pts[1], xytext=pts[0], arrowprops=dict(arrowstyle="->", lw=0.5, color="0.4"))
    for bb in BB:
        ax.scatter([], [], marker=MARK[bb], facecolors="none", edgecolors="black", label=BB_LABEL[bb])
    ax.scatter([], [], marker="o", facecolors="none", edgecolors="black", label="baseline")
    ax.scatter([], [], marker="o", facecolors="black", edgecolors="black", label="optimised")
    ax.set_xlabel("GMACs per clip (8 frames)"); ax.set_ylabel("Test macro-F1")
    ax.legend(frameon=False, loc="lower right"); ax.grid(True, lw=0.3, color="0.85")
    save(fig, root, "Fig4")


def fig5(root):
    fig, axes = plt.subplots(1, 3, figsize=(FULL, FULL * 0.28), sharey=True)
    found = False
    for ax, bb in zip(axes, BB):
        for t in ("scratch", "frozen", "partial", "full"):
            p = os.path.join(root, "experiments", f"A_{bb}_{t}_s0", "log.csv")
            if not os.path.exists(p):
                continue
            rows = read_csv(p); found = True
            ax.plot([int(r["epoch"]) + 1 for r in rows], [float(r["val_macro_f1"]) for r in rows],
                    ls=LS[t], color="black", lw=0.9, label=t)
        ax.set_title(BB_LABEL[bb]); ax.set_xlabel("Epoch"); ax.grid(True, lw=0.3, color="0.85")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("Validation macro-F1")
    axes[-1].legend(frameon=False, loc="lower right")
    if found:
        save(fig, root, "Fig5")
    else:
        print("Fig5 skipped: no group A logs"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=REPO_ROOT)
    args = ap.parse_args()
    runs = load_runs(args.root)
    fig2(runs, args.root); fig3(runs, args.root); fig4(runs, args.root); fig5(args.root)


if __name__ == "__main__":
    main()
