# Badminton stroke recognition — training-strategy study

Code for the article *"Research on Training Optimization Strategies for Deep
Learning Models in Racket Sports Technique Recognition"*.

Task: 18-class clip-level stroke classification on VideoBadminton with a
TSN-style model: 8 frames per clip, a shared ImageNet-pretrained 2D backbone
(MobileNetV3-Large, EfficientNet-B0, ResNet-18), temporal average pooling and a
linear head. 69 runs: an LR check (L, 3) and groups A–E (66) vary transfer strategy, augmentation, LR
schedule, loss and resolution, on a fixed session-disjoint 70/15/15 split.

## Layout

```
configs/      base.yaml (= baseline) + generated A/ B/ C/ D/ E/ configs
src/          all code (run as modules from the repo root: python -m src.<name>)
data/         index.csv, index_meta.json, splits/, frame_meta.csv   (frames go to local disk)
experiments/  per-run log.csv and resume checkpoint
models/       per-run best.pt
results/      per-run metrics.json + test_preds.npz; summary tables
figures/      Day 8
```

Every run stores seed, model, optimiser, LR, schedule, batch size, epochs,
augmentation (resolved config inside metrics.json and config.yaml), the best
checkpoint and all metrics. Run id = `<config name>_s<seed>`.

## Workflow

On Colab use `notebooks/colab_runner.ipynb`; it runs the same commands.
Keep the repo on Google Drive (results persist) and the frames on local disk
(`data.frames_root=/content/frames`, fast).

### Day 2 · Section 1 — index and split
1. `python -m src.make_index --video-root <VideoBadminton> --inspect`
   Check the parsed date/time examples and the session counts.
2. `python -m src.make_index --video-root <VideoBadminton> --session-key <date|date_time>`
3. `python -m src.make_splits` → `data/splits/*.csv`, `split_summary.json` (Table 1).

### Day 2 · Section 2 — frames
1. `python -m src.extract_frames --video-root <VideoBadminton> --frames-root /content/frames --workers 8`
2. `tar -cf <drive>/frames.tar -C /content frames` (once; restore with `tar -xf` in later sessions).
3. Check `data/frame_meta.csv` for `unreadable` / `decode_failed` rows.

### Day 3 — smoke test, calibration, profiling
1. `python -m src.train --config configs/base.yaml --smoke --override data.frames_root=/content/frames`
2. `python -m src.calibrate --override data.frames_root=/content/frames` → projected GPU-hours.
3. `python -m src.profile_model --device cuda` (Colab) and `--device cpu` (laptop) → `results/efficiency_*.csv`.

### Day 3 (end) — LR check, approved extra group L (3 runs)
1. `python -m src.make_configs --group L && python -m src.run_group --group L --override data.frames_root=/content/frames`
2. `python -m src.analyze lrcheck` → selected LR (validation macro-F1 only).
3. Write that value into `train.lr` in `configs/base.yaml`; it is then fixed for all other runs.

### Days 4–7 — run plan (each line: generate configs, then run)
1. A (12): `python -m src.make_configs --group A && python -m src.run_group --group A --override data.frames_root=/content/frames`
2. B (15): same with `--group B`.
3. C (≤15): `--group C` (selects options on seed-0 **validation** macro-F1 of A and B; writes `configs/C/selection.json`).

Then:
1. D (18): `python -m src.make_configs --group D --third 'B_{bb}_sport_aware'` (third key config: decide after C), run `--group D`.
2. E (6): `--group E` (baseline and optimised on the random clip split).
3. Analysis: `python -m src.analyze summary`, `seeds`, `leakage`,
   `bootstrap --a A_<bb>_full_s0 --b C_<bb>_opt_s0`, `perclass --runs ...`.

### Day 8 — figures and hand-off
1. `python -m src.analyze perclass --runs A_<bb>_full_s0 C_<bb>_opt_s0 ...` (all three backbones)
2. `python -m src.make_figures` → `figures/Fig2.pdf` … `Fig5.pdf`
3. `python -m src.export_bundle` → `paper_bundle.zip` (results only; no checkpoints or frames)

Interrupted runs resume from `experiments/<run_id>/last.pt`; finished runs are skipped.

## Notes for the paper

- Macro-F1 is the primary metric; mean class accuracy (= macro recall) is
  reported for comparability with the original benchmark.
- MACs: `torch.utils.flop_counter` (FLOPs/2), convolutions and matmuls only.
- Latency: fp32, batch of one 8-frame clip, median of 100 runs after 20 warm-up runs.
- Bootstrap: clip-level paired percentile bootstrap (10,000 resamples);
  clips within a session are correlated, so intervals may be narrow.
- Determinism: seeds fix initialisation, data order and augmentation, but
  `cudnn.benchmark=True` and resumed runs make results reproducible only
  approximately. Report seed variance (group D) rather than single runs.

## Licence and citation

Licence: to be chosen by the author before the repository is made public.
Dataset: VideoBadminton is not redistributed here; obtain it from its authors and
use it under their release terms. Cite the article (details added on publication).
