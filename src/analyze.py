"""Aggregate results. Pure numpy/sklearn; no plotting here (figures: Day 8).

    python -m src.analyze summary                 # results/summary.csv (one row per run)
    python -m src.analyze seeds                   # results/seed_summary.csv (mean +- std by config name)
    python -m src.analyze bootstrap --a A_resnet18_full_s0 --b C_resnet18_opt_s0 [--n 10000]
    python -m src.analyze perclass --runs A_resnet18_full_s0 C_resnet18_opt_s0
    python -m src.analyze leakage                 # session vs random split, same config (RQ4)
    python -m src.analyze lrcheck                 # group L: pick the base LR by validation macro-F1
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score

from src.utils import REPO_ROOT, load_json, write_csv, save_json


def results_root(args):
    return os.path.join(args.root, "results")


def load_all(root):
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", "metrics.json"))):
        out.append(load_json(p))
    return out


def summary(args):
    rows = []
    for m in load_all(results_root(args)):
        rows.append(dict(
            run_id=m["run_id"], name=m["name"], group=m["group"], seed=m["seed"], backbone=m["backbone"],
            transfer=m["transfer"], aug=m["aug"], schedule=m["schedule"], loss=m["loss"],
            resolution=m["resolution"], split=m["split"],
            val_macro_f1=round(m["val"]["macro_f1"], 4), test_macro_f1=round(m["test"]["macro_f1"], 4),
            test_acc=round(m["test"]["accuracy"], 4), test_macro_p=round(m["test"]["macro_precision"], 4),
            test_macro_r=round(m["test"]["macro_recall"], 4),
            test_mean_class_acc=round(m["test"]["mean_class_accuracy"], 4),
            params_trainable=m["params_trainable"], params_total=m["params_total"],
            best_epoch=m["best_epoch"], epochs_run=m["epochs_run"],
            train_time_min=round(m["train_time_s"] / 60, 2), time_to_best_min=round(m["time_to_best_s"] / 60, 2),
            peak_gpu_mem_mb=m["peak_gpu_mem_mb"], device=m["env"]["device"]))
    out = os.path.join(results_root(args), "summary.csv")
    write_csv(rows, out)
    print(f"{len(rows)} runs -> {out}")


def seeds(args):
    groups = defaultdict(list)
    for m in load_all(results_root(args)):
        groups[m["name"]].append(m)
    rows = []
    for name, ms in sorted(groups.items()):
        f1 = np.array([m["test"]["macro_f1"] for m in ms])
        acc = np.array([m["test"]["accuracy"] for m in ms])
        tt = np.array([m["train_time_s"] for m in ms]) / 60
        ddof = 1 if len(ms) > 1 else 0
        rows.append(dict(name=name, n_seeds=len(ms), seeds=" ".join(str(m["seed"]) for m in ms),
                         test_macro_f1_mean=round(f1.mean(), 4), test_macro_f1_std=round(f1.std(ddof=ddof), 4),
                         test_acc_mean=round(acc.mean(), 4), test_acc_std=round(acc.std(ddof=ddof), 4),
                         train_time_min_mean=round(tt.mean(), 2)))
    out = os.path.join(results_root(args), "seed_summary.csv")
    write_csv(rows, out)
    print(f"{len(rows)} configs -> {out}  (std uses ddof=1 when n_seeds > 1)")


def load_preds(root, rid):
    d = np.load(os.path.join(root, rid, "test_preds.npz"))
    return d["clip_ids"], d["labels"], d["logits"].argmax(1)


def bootstrap(args):
    root = results_root(args)
    ida, ya, pa = load_preds(root, args.a)
    idb, yb, pb = load_preds(root, args.b)
    if not (np.array_equal(ida, idb) and np.array_equal(ya, yb)):
        raise SystemExit("runs were evaluated on different test sets; paired bootstrap not valid")
    labels = np.unique(ya)
    rng = np.random.default_rng(args.seed)
    n = len(ya)
    fa = f1_score(ya, pa, labels=labels, average="macro", zero_division=0)
    fb = f1_score(ya, pb, labels=labels, average="macro", zero_division=0)
    diffs = np.empty(args.n)
    for i in range(args.n):
        idx = rng.integers(0, n, n)
        diffs[i] = (f1_score(ya[idx], pb[idx], labels=labels, average="macro", zero_division=0)
                    - f1_score(ya[idx], pa[idx], labels=labels, average="macro", zero_division=0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    res = dict(a=args.a, b=args.b, macro_f1_a=fa, macro_f1_b=fb, diff_b_minus_a=fb - fa,
               ci95=[float(lo), float(hi)], n_boot=args.n, seed=args.seed, n_test=int(n),
               note="clip-level paired percentile bootstrap; clips within a session are correlated")
    out = os.path.join(root, f"bootstrap_{args.a}_vs_{args.b}.json")
    save_json(res, out)
    print(json.dumps(res, indent=2))


def perclass(args):
    root = results_root(args)
    classes = load_json(os.path.join(args.root, "data", "index_meta.json"))["classes"]
    rows = [dict(cls=c) for c in classes]
    for rid in args.runs:
        m = load_json(os.path.join(root, rid, "metrics.json"))
        for i, f in enumerate(m["test"]["per_class_f1"]):
            rows[i][rid] = round(f, 4)
        cm = np.array(m["test"]["confusion_matrix"])
        np.savetxt(os.path.join(root, rid, "confusion_test.csv"), cm, fmt="%d", delimiter=",")
    out = os.path.join(root, "perclass_f1.csv")
    write_csv(rows, out)
    print("->", out)


def leakage(args):
    """Pair each E run with the same configuration on the session split (seed 0)."""
    ms = {m["run_id"]: m for m in load_all(results_root(args))}
    rows = []
    for rid, m in ms.items():
        if m["split"] != "random_split":
            continue
        bb = m["backbone"]
        twin = f"A_{bb}_full_s0" if "baseline" in m["name"] else f"C_{bb}_opt_s0"
        if twin not in ms:
            print("missing twin", twin); continue
        s = ms[twin]
        rows.append(dict(backbone=bb, config="baseline" if "baseline" in m["name"] else "optimised",
                         session_test_f1=round(s["test"]["macro_f1"], 4),
                         random_test_f1=round(m["test"]["macro_f1"], 4),
                         gap=round(m["test"]["macro_f1"] - s["test"]["macro_f1"], 4)))
    out = os.path.join(results_root(args), "leakage.csv")
    write_csv(rows, out)
    print("->", out)
    for r in rows:
        print(r)


def lrcheck(args):
    ms = [m for m in load_all(results_root(args)) if m["name"].startswith("L_")]
    if not ms:
        raise SystemExit("no group L results")
    ms.sort(key=lambda m: m["config"]["train"]["lr"])
    for m in ms:
        print(f"lr {m['config']['train']['lr']:.0e}  val macro-F1 {m['val']['macro_f1']:.4f}  "
              f"best epoch {m['best_epoch']}  time {m['train_time_s'] / 60:.1f} min")
    best = max(ms, key=lambda m: m["val"]["macro_f1"])
    print(f"\nselected lr = {best['config']['train']['lr']:.0e} (validation only). "
          f"Set train.lr in configs/base.yaml BEFORE generating group A.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary", "seeds", "bootstrap", "perclass", "leakage", "lrcheck"])
    ap.add_argument("--root", default=REPO_ROOT, help="folder containing results/ (and data/)")
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", nargs="*", default=[])
    args = ap.parse_args()
    {"summary": summary, "seeds": seeds, "bootstrap": bootstrap,
     "perclass": perclass, "leakage": leakage, "lrcheck": lrcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
