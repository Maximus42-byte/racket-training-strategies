"""Train and evaluate one configuration.

    python -m src.train --config configs/A/A_mobilenet_v3_large_full.yaml
    python -m src.train --config configs/base.yaml --smoke            # 1 epoch, 4 batches
    python -m src.train --config ... --override train.max_epochs=1    # dotted overrides

Per run (run_id = <name>_s<seed>) under run.out_root:
  experiments/<run_id>/log.csv, last.pt   epoch log; resume checkpoint (deleted when done)
  models/<run_id>/best.pt                 best-validation-macro-F1 weights
  results/<run_id>/metrics.json           config, val/test metrics, timings, memory, params
  results/<run_id>/test_preds.npz         clip_ids, labels, logits (for bootstrap / errors)
A run whose metrics.json exists is skipped; an interrupted run resumes from last.pt.
"""
import argparse
import math
import os
import platform
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from torch.utils.data import DataLoader

from src.data import ClipDataset
from src.models import build_model, count_params, set_train_mode
from src.utils import (load_config, load_json, resolve, run_id, save_json, save_yaml,
                       seed_everything, write_csv, read_csv)


def compute_metrics(y, logits, n_classes):
    pred = logits.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(n_classes)),
                                                 average=None, zero_division=0)
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0)),
        macro_precision=float(p.mean()), macro_recall=float(r.mean()),
        mean_class_accuracy=float(balanced_accuracy_score(y, pred)),
        per_class_f1=[float(x) for x in f],
        confusion_matrix=confusion_matrix(y, pred, labels=list(range(n_classes))).tolist(),
    )


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    ys, outs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
        outs.append(out.float().cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(outs).numpy()


def make_scheduler(opt, cfg, steps_per_epoch):
    tc = cfg["train"]
    total = tc["max_epochs"] * steps_per_epoch
    warm = tc["warmup_epochs"] * steps_per_epoch
    lo = tc["min_lr_ratio"]

    def f(step):
        if tc["schedule"] == "constant":
            return 1.0
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    w = 1.0 / np.maximum(counts, 1)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def train_one(cfg, seed):
    rid = run_id(cfg, seed)
    res_dir, exp_dir, mod_dir = (resolve(cfg, os.path.join(d, rid)) for d in ("results", "experiments", "models"))
    if os.path.exists(os.path.join(res_dir, "metrics.json")) and not cfg.get("smoke"):
        print(f"[skip] {rid} already finished")
        return
    os.makedirs(exp_dir, exist_ok=True)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc, dc = cfg["train"], cfg["data"]
    amp = tc["amp"] and device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    n_classes = len(load_json(os.path.join(dc.get("root", "."), dc["index_meta"]))["classes"])
    ds_tr, ds_va, ds_te = (ClipDataset(cfg, s, train=(s == "train")) for s in ("train", "val", "test"))
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=tc["batch_size"], num_workers=dc["num_workers"], pin_memory=device.type == "cuda",
              persistent_workers=dc["num_workers"] > 0)
    dl_tr = DataLoader(ds_tr, shuffle=True, drop_last=True, generator=g, **kw)
    dl_va = DataLoader(ds_va, shuffle=False, **kw)
    dl_te = DataLoader(ds_te, shuffle=False, **kw)

    model = build_model(cfg, n_classes).to(device)
    if tc["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    total_p, train_p = count_params(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = make_scheduler(opt, cfg, len(dl_tr))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    weight = class_weights(ds_tr.labels(), n_classes).to(device) if tc["loss"] == "weighted_ce" else None
    crit = nn.CrossEntropyLoss(weight=weight)

    start_epoch, best_f1, best_epoch, elapsed, time_to_best, bad_epochs = 0, -1.0, -1, 0.0, 0.0, 0
    log_rows = []
    last_ckpt = os.path.join(exp_dir, "last.pt")
    if os.path.exists(last_ckpt) and not cfg.get("smoke"):
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_f1, best_epoch = ck["epoch"] + 1, ck["best_f1"], ck["best_epoch"]
        elapsed, time_to_best, bad_epochs = ck["elapsed"], ck["time_to_best"], ck["bad_epochs"]
        torch.set_rng_state(ck["rng_cpu"])
        log_rows = read_csv(os.path.join(exp_dir, "log.csv"))
        print(f"[resume] {rid} from epoch {start_epoch}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    max_batches = 4 if cfg.get("smoke") else None
    print(f"[run] {rid} | {cfg['model']['backbone']} {cfg['model']['transfer']} aug={cfg['aug']['policy']} "
          f"sched={tc['schedule']} loss={tc['loss']} res={tc['resolution']} split={os.path.basename(dc['split_csv'])} "
          f"| train/val/test {len(ds_tr)}/{len(ds_va)}/{len(ds_te)} | trainable {train_p:,}/{total_p:,}")

    for epoch in range(start_epoch, tc["max_epochs"]):
        t0 = time.time()
        set_train_mode(model, cfg["model"]["freeze_bn_in_frozen"])
        loss_sum, n = 0.0, 0
        for bi, (x, y) in enumerate(dl_tr):
            if max_batches and bi >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * y.size(0); n += y.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.time() - t0
        yv, lv = predict(model, dl_va, device, amp)
        mv = compute_metrics(yv, lv, n_classes)
        t_epoch = time.time() - t0
        elapsed += t_epoch
        improved = mv["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, time_to_best, bad_epochs = mv["macro_f1"], epoch, elapsed, 0
            os.makedirs(mod_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(mod_dir, "best.pt"))
        else:
            bad_epochs += 1
        log_rows.append(dict(epoch=epoch, train_loss=round(loss_sum / max(n, 1), 5),
                             val_macro_f1=round(mv["macro_f1"], 5), val_acc=round(mv["accuracy"], 5),
                             lr=opt.param_groups[0]["lr"], t_train_s=round(t_train, 2),
                             t_epoch_s=round(t_epoch, 2), elapsed_s=round(elapsed, 2)))
        write_csv(log_rows, os.path.join(exp_dir, "log.csv"))
        print(f"  ep {epoch:2d} loss {loss_sum / max(n, 1):.4f} val F1 {mv['macro_f1']:.4f} "
              f"acc {mv['accuracy']:.4f} {'*' if improved else ' '} {t_epoch:.0f}s", flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=epoch, best_f1=best_f1, best_epoch=best_epoch,
                        elapsed=elapsed, time_to_best=time_to_best, bad_epochs=bad_epochs,
                        rng_cpu=torch.get_rng_state()), last_ckpt)
        if bad_epochs >= tc["patience"]:
            print(f"  early stop at epoch {epoch} (best {best_epoch})")
            break

    peak_mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    model.load_state_dict(torch.load(os.path.join(mod_dir, "best.pt"), map_location=device))
    yt, lt = predict(model, dl_te, device, amp)
    yv, lv = predict(model, dl_va, device, amp)
    os.makedirs(res_dir, exist_ok=True)
    np.savez_compressed(os.path.join(res_dir, "test_preds.npz"), clip_ids=np.array(ds_te.clip_ids()),
                        labels=yt, logits=lt)
    out = dict(
        run_id=rid, seed=seed, group=cfg["run"]["group"], name=cfg["run"]["name"],
        backbone=cfg["model"]["backbone"], transfer=cfg["model"]["transfer"], aug=cfg["aug"]["policy"],
        schedule=tc["schedule"], loss=tc["loss"], resolution=tc["resolution"],
        split=os.path.basename(dc["split_csv"]).replace(".csv", ""),
        params_total=total_p, params_trainable=train_p,
        best_epoch=best_epoch, epochs_run=len(log_rows),
        train_time_s=round(elapsed, 1), time_to_best_s=round(time_to_best, 1),
        peak_gpu_mem_mb=None if peak_mem is None else round(peak_mem, 1),
        val=compute_metrics(yv, lv, n_classes), test=compute_metrics(yt, lt, n_classes),
        env=dict(device=torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
                 torch=torch.__version__, amp=amp, smoke=bool(cfg.get("smoke"))),
        config=cfg,
    )
    if cfg.get("smoke"):
        print(f"[smoke ok] val F1 {out['val']['macro_f1']:.4f} test F1 {out['test']['macro_f1']:.4f} "
              f"peak mem {out['peak_gpu_mem_mb']} MB, {elapsed:.0f}s")
        shutil.rmtree(exp_dir, ignore_errors=True); shutil.rmtree(mod_dir, ignore_errors=True)
        return
    save_yaml(cfg, os.path.join(res_dir, "config.yaml"))
    save_json(out, os.path.join(res_dir, "metrics.json"))
    os.remove(last_ckpt)
    print(f"[done] {rid} best ep {best_epoch} val F1 {out['val']['macro_f1']:.4f} "
          f"test F1 {out['test']['macro_f1']:.4f} acc {out['test']['accuracy']:.4f} | {elapsed / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="*", help="overrides run.seeds")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 4 batches, nothing kept")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    if args.smoke:
        cfg["smoke"] = True
        cfg["train"]["max_epochs"] = 1
    for seed in (args.seeds if args.seeds else cfg["run"]["seeds"]):
        train_one(cfg, seed)


if __name__ == "__main__":
    main()
