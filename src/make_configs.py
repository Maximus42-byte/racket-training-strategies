"""Generate configs for the run plan.

    python -m src.make_configs --group L      # 3: LR check (MobileNetV3 baseline, 5 epochs) -> sets base LR
    python -m src.make_configs --group A      # 12: 3 backbones x {scratch, frozen, partial, full}
    python -m src.make_configs --group B      # 15: baseline + one change each
    python -m src.make_configs --group C      # 15: optimised + 4 leave-one-out (needs A and B results)
    python -m src.make_configs --group D --third B_{bb}_sport_aware   # seeds 1,2 of 3 key configs
    python -m src.make_configs --group E      # 6: baseline + optimised on the random clip split

The baseline is A_<bb>_full (pretrained, full FT, no aug, constant LR, CE, 224 px).
Selection for C uses seed-0 VALIDATION macro-F1 only; test data are never used.
"""
import argparse
import glob
import os

import yaml

from src.utils import REPO_ROOT, load_json, save_json, save_yaml

BACKBONES = ["mobilenet_v3_large", "efficientnet_b0", "resnet18"]
LR_CHECK = [1e-4, 3e-4, 1e-3]  # approved 3-run LR check on the MobileNetV3 baseline
CFG_DIR = os.path.join(REPO_ROOT, "configs")

# single-factor changes relative to the baseline (group B)
B_CHANGES = {
    "generic_aug": {"aug": {"policy": "generic"}},
    "sport_aware": {"aug": {"policy": "sport_aware"}},
    "cosine": {"train": {"schedule": "cosine"}},
    "weighted_ce": {"train": {"loss": "weighted_ce"}},
    "res160": {"train": {"resolution": 160}},
}
# factor -> (baseline value, {option value: B run tag})
FACTORS = {
    "aug": ("none", {"generic": "generic_aug", "sport_aware": "sport_aware"}),
    "schedule": ("constant", {"cosine": "cosine"}),
    "loss": ("ce", {"weighted_ce": "weighted_ce"}),
    "resolution": (224, {160: "res160"}),
}
SECTION = {"aug": ("aug", "policy"), "schedule": ("train", "schedule"),
           "loss": ("train", "loss"), "resolution": ("train", "resolution")}


def write(group, name, body, seeds=(0,)):
    cfg = {"base": "../base.yaml", **body}
    cfg["run"] = dict(cfg.get("run", {}), name=name, group=group, seeds=list(seeds))
    path = os.path.join(CFG_DIR, group, f"{name}.yaml")
    save_yaml(cfg, path)
    return path


def val_f1(name, results_root, seed=0):
    p = os.path.join(results_root, "results", f"{name}_s{seed}", "metrics.json")
    if not os.path.exists(p):
        raise SystemExit(f"missing result {p}; finish the earlier group first")
    return load_json(p)["val"]["macro_f1"]


def optimised(bb, results_root):
    """Pick each factor's level by seed-0 validation macro-F1."""
    choice, evidence = {}, {}
    transfers = ["scratch", "frozen", "partial", "full"]
    scores = {t: val_f1(f"A_{bb}_{t}", results_root) for t in transfers}
    choice["transfer"] = max(scores, key=scores.get)
    evidence["transfer"] = scores
    base = val_f1(f"A_{bb}_full", results_root)
    for f, (bval, opts) in FACTORS.items():
        sc = {bval: base}
        sc.update({v: val_f1(f"B_{bb}_{tag}", results_root) for v, tag in opts.items()})
        choice[f] = max(sc, key=sc.get)
        evidence[f] = {str(k): v for k, v in sc.items()}
    return choice, evidence


def body_from(choice, split=None):
    body = {"model": {"backbone": None, "transfer": choice["transfer"]}, "train": {}, "aug": {}}
    for f in FACTORS:
        sec, key = SECTION[f]
        body[sec][key] = choice[f]
    if split:
        body["data"] = {"split_csv": f"data/splits/{split}.csv"}
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list("LABCDE"))
    ap.add_argument("--results-root", default=REPO_ROOT, help="folder containing results/")
    ap.add_argument("--third", help="group D: third key config name pattern, e.g. B_{bb}_sport_aware")
    args = ap.parse_args()
    made = []

    if args.group == "L":
        for lr in LR_CHECK:
            made.append(write("L", f"L_mobilenet_v3_large_lr{lr:.0e}",
                              {"model": {"backbone": "mobilenet_v3_large"},
                               "train": {"lr": lr, "max_epochs": 5, "patience": 5}}))

    elif args.group == "A":
        for bb in BACKBONES:
            for t in ["scratch", "frozen", "partial", "full"]:
                made.append(write("A", f"A_{bb}_{t}", {"model": {"backbone": bb, "transfer": t}}))

    elif args.group == "B":
        for bb in BACKBONES:
            for tag, change in B_CHANGES.items():
                body = {"model": {"backbone": bb}, **change}
                made.append(write("B", f"B_{bb}_{tag}", body))

    elif args.group == "C":
        record = {}
        for bb in BACKBONES:
            choice, ev = optimised(bb, args.results_root)
            record[bb] = {"choice": {k: str(v) for k, v in choice.items()}, "val_macro_f1": ev}
            body = body_from(choice); body["model"]["backbone"] = bb
            made.append(write("C", f"C_{bb}_opt", body))
            for f, (bval, _) in FACTORS.items():
                if choice[f] == bval:
                    print(f"[skip] C_{bb}_loo_{f}: optimised already uses the baseline value")
                    continue
                c2 = dict(choice); c2[f] = bval
                b2 = body_from(c2); b2["model"]["backbone"] = bb
                made.append(write("C", f"C_{bb}_loo_{f}", b2))
        save_json(record, os.path.join(CFG_DIR, "C", "selection.json"))

    elif args.group == "D":
        if not args.third:
            raise SystemExit("--third is required, e.g. --third B_{bb}_sport_aware")
        for bb in BACKBONES:
            for src in (f"A_{bb}_full", f"C_{bb}_opt", args.third.format(bb=bb)):
                grp = src.split("_")[0]
                path = os.path.join(CFG_DIR, grp, f"{src}.yaml")
                if not os.path.exists(path):
                    raise SystemExit(f"missing {path}")
                with open(path) as fh:
                    body = yaml.safe_load(fh)
                body.pop("base", None)
                # same run name as the seed-0 run, so seeds aggregate by name
                made.append(write("D", src, body, seeds=(1, 2)))
                with open(made[-1]) as fh:
                    fixed = yaml.safe_load(fh)
                fixed["run"]["group"] = grp  # keep the original group label
                save_yaml(fixed, made[-1])

    elif args.group == "E":
        sel = load_json(os.path.join(CFG_DIR, "C", "selection.json"))
        for bb in BACKBONES:
            made.append(write("E", f"E_{bb}_baseline_random",
                              {"model": {"backbone": bb}, "data": {"split_csv": "data/splits/random_split.csv"}}))
            ch = sel[bb]["choice"]
            choice = {"transfer": ch["transfer"], "aug": ch["aug"], "schedule": ch["schedule"],
                      "loss": ch["loss"], "resolution": int(ch["resolution"])}
            body = body_from(choice, split="random_split"); body["model"]["backbone"] = bb
            made.append(write("E", f"E_{bb}_opt_random", body))

    print(f"group {args.group}: {len(made)} configs")
    for m in made:
        print("  ", os.path.relpath(m, REPO_ROOT))


if __name__ == "__main__":
    main()
