"""Shared helpers: config loading, seeding, small IO utilities.

Configs are YAML. A config may contain `base: <relative path>`; the base is
loaded first and the config's keys override it (deep merge). Command-line
overrides use dotted keys, e.g. `train.max_epochs=1`.
"""
import copy
import csv
import json
import os
import random

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def deep_update(base, new):
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_path(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        base_cfg = load_config(os.path.join(os.path.dirname(path), base))
        cfg = deep_update(base_cfg, cfg)
    for ov in overrides:
        key, val = ov.split("=", 1)
        set_by_path(cfg, key, yaml.safe_load(val))
    return cfg


def save_yaml(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fieldnames=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_id(cfg, seed):
    return f"{cfg['run']['name']}_s{seed}"


def resolve(cfg, rel):
    """Resolve a path relative to run.out_root (repo root by default)."""
    root = cfg.get("run", {}).get("out_root", REPO_ROOT)
    return rel if os.path.isabs(rel) else os.path.join(root, rel)
