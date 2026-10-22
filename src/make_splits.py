"""Create the two fixed splits used in all runs.

    python -m src.make_splits            # uses data/index.csv
Outputs:
    data/splits/session_split.csv   clip_id,split  (session-disjoint, 70/15/15)
    data/splits/random_split.csv    clip_id,split  (stratified random clips, 70/15/15)
    data/splits/split_summary.json  statistics for Table 1

Session split: all clips of a session go to one subset. Many random session
orders are assigned greedily to the subset with the largest relative deficit;
the assignment with the best score is kept. The score penalises (i) deviation
from 70/15/15 clip proportions, (ii) classes missing from val or test, and
(iii) class-distribution differences between subsets. Deterministic given --seed.
"""
import argparse
import collections
import os

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils import read_csv, write_csv, save_json, REPO_ROOT

SPLITS = ("train", "val", "test")


def session_split(rows, ratios, seed, restarts, w_missing=1.0, w_dist=0.5):
    labels = sorted({int(r["label_idx"]) for r in rows})
    C = max(labels) + 1
    sess = collections.defaultdict(lambda: np.zeros(C))
    for r in rows:
        sess[r["session"]][int(r["label_idx"])] += 1
    names = sorted(sess)
    M = np.stack([sess[n] for n in names])            # sessions x classes
    sizes = M.sum(1)
    total = sizes.sum()
    ratios = np.asarray(ratios)
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(restarts):
        order = rng.permutation(len(names))
        cur = np.zeros(3)
        assign = np.empty(len(names), dtype=int)
        for i in order:
            deficit = (ratios * total - cur) / (ratios * total)
            k = int(np.argmax(deficit))
            assign[i] = k
            cur[k] += sizes[i]
        frac = cur / total
        cls = np.stack([M[assign == k].sum(0) for k in range(3)])  # 3 x C
        missing = int((cls[1:] == 0).sum())
        dist = cls / np.maximum(cls.sum(1, keepdims=True), 1)
        dist_dev = np.abs(dist[1:] - dist[0]).sum(1).mean()
        score = np.abs(frac - ratios).sum() + w_missing * missing + w_dist * dist_dev
        if score < best_score:
            best_score, best = score, assign.copy()
    mapping = {n: SPLITS[k] for n, k in zip(names, best)}
    return {r["clip_id"]: mapping[r["session"]] for r in rows}, float(best_score)


def random_split(rows, ratios, seed):
    ids = [r["clip_id"] for r in rows]
    y = [int(r["label_idx"]) for r in rows]
    tr, rest, ytr, yrest = train_test_split(ids, y, train_size=ratios[0], stratify=y, random_state=seed)
    va, te = train_test_split(rest, train_size=ratios[1] / (ratios[1] + ratios[2]),
                              stratify=yrest, random_state=seed)
    out = {i: "train" for i in tr}
    out.update({i: "val" for i in va})
    out.update({i: "test" for i in te})
    return out


def summarise(rows, assign, n_classes):
    out = {}
    for s in SPLITS:
        sub = [r for r in rows if assign[r["clip_id"]] == s]
        counts = collections.Counter(int(r["label_idx"]) for r in sub)
        per_cls = [counts.get(c, 0) for c in range(n_classes)]
        out[s] = dict(n_clips=len(sub), n_sessions=len({r["session"] for r in sub}),
                      min_per_class=min(per_cls), max_per_class=max(per_cls),
                      per_class=per_cls, classes_missing=[c for c in range(n_classes) if per_cls[c] == 0])
    out["total_clips"] = len(rows)
    out["total_sessions"] = len({r["session"] for r in rows})
    # sessions shared across subsets (must be 0 for the session split)
    sess_sets = {s: {r["session"] for r in rows if assign[r["clip_id"]] == s} for s in SPLITS}
    out["sessions_shared_train_test"] = len(sess_sets["train"] & sess_sets["test"])
    out["sessions_shared_train_val"] = len(sess_sets["train"] & sess_sets["val"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(REPO_ROOT, "data", "index.csv"))
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "splits"))
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=5000)
    args = ap.parse_args()

    rows = read_csv(args.index)
    n_classes = max(int(r["label_idx"]) for r in rows) + 1
    sess_assign, score = session_split(rows, args.ratios, args.seed, args.restarts)
    rand_assign = random_split(rows, args.ratios, args.seed)

    for name, assign in (("session_split", sess_assign), ("random_split", rand_assign)):
        write_csv([{"clip_id": r["clip_id"], "split": assign[r["clip_id"]]} for r in rows],
                  os.path.join(args.out_dir, f"{name}.csv"))

    summary = {"session_split": summarise(rows, sess_assign, n_classes),
               "random_split": summarise(rows, rand_assign, n_classes),
               "session_split_score": score, "seed": args.seed, "ratios": args.ratios}
    save_json(summary, os.path.join(args.out_dir, "split_summary.json"))

    for name in ("session_split", "random_split"):
        s = summary[name]
        print(f"\n{name}: total {s['total_clips']} clips, {s['total_sessions']} sessions")
        for k in SPLITS:
            d = s[k]
            print(f"  {k:5s} clips {d['n_clips']:5d} ({d['n_clips'] / s['total_clips']:.3f})"
                  f"  sessions {d['n_sessions']:3d}  min/max per class {d['min_per_class']}/{d['max_per_class']}"
                  f"  missing classes {d['classes_missing']}")
        print(f"  sessions shared train-test: {s['sessions_shared_train_test']}")


if __name__ == "__main__":
    main()
