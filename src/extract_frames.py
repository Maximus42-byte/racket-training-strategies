"""Extract candidate frames once, so training never decodes video.

    python -m src.extract_frames --video-root /path/to/VideoBadminton \
        --frames-root /content/frames --workers 8

For T=8 segments and N_CAND=3 candidates per segment, frame (k, j) is taken at
relative position (k + (j + 0.5) / N_CAND) / T of the clip. Candidate j=1 is the
segment centre (TSN-style evaluation frame); training with temporal jitter
picks j uniformly from {0, 1, 2}. Frames are resized to a short side of
`--short-side` px (aspect kept) and saved as JPEG:
    <frames-root>/<clip_id>/f{k}_{j}.jpg
Resumable: clips whose folder already holds all frames are skipped.
Also writes data/frame_meta.csv (n_frames, fps, width, height per clip).
"""
import argparse
import os
from multiprocessing import Pool

import cv2

from src.utils import read_csv, write_csv, REPO_ROOT

T, N_CAND = 8, 3


def candidate_positions(n_frames, t=T, n_cand=N_CAND):
    idx = []
    for k in range(t):
        for j in range(n_cand):
            pos = (k + (j + 0.5) / n_cand) / t * n_frames
            idx.append(min(int(pos), n_frames - 1))
    return idx


def count_frames(path):
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.grab():
        n += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def resize_short(img, short):
    h, w = img.shape[:2]
    s = short / min(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def process(job):
    clip_id, path, out_root, short, quality = job
    out_dir = os.path.join(out_root, clip_id)
    names = [f"f{k}_{j}.jpg" for k in range(T) for j in range(N_CAND)]
    n, fps, w, h = count_frames(path)
    meta = dict(clip_id=clip_id, n_frames=n, fps=round(fps, 3), width=w, height=h, status="ok")
    if all(os.path.exists(os.path.join(out_dir, x)) for x in names):
        meta["status"] = "skipped"
        return meta
    if n == 0:
        meta["status"] = "unreadable"
        return meta
    os.makedirs(out_dir, exist_ok=True)
    wanted = candidate_positions(n)
    needed = sorted(set(wanted))
    cap = cv2.VideoCapture(path)
    frames, i, ptr = {}, 0, 0
    while ptr < len(needed) and cap.grab():
        if i == needed[ptr]:
            ok, img = cap.retrieve()
            if ok:
                frames[i] = resize_short(img, short)
            ptr += 1
        i += 1
    cap.release()
    for name, fi in zip(names, wanted):
        if fi not in frames:  # decode failure: fall back to nearest decoded frame
            if not frames:
                meta["status"] = "decode_failed"
                return meta
            fi = min(frames, key=lambda x: abs(x - fi))
            meta["status"] = "fallback"
        cv2.imwrite(os.path.join(out_dir, name), frames[fi], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--frames-root", default=os.path.join(REPO_ROOT, "data", "frames"))
    ap.add_argument("--index", default=os.path.join(REPO_ROOT, "data", "index.csv"))
    ap.add_argument("--meta-out", default=os.path.join(REPO_ROOT, "data", "frame_meta.csv"))
    ap.add_argument("--short-side", type=int, default=256)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="process only N clips (testing)")
    args = ap.parse_args()

    rows = read_csv(args.index)
    if args.limit:
        rows = rows[: args.limit]
    jobs = [(r["clip_id"], os.path.join(args.video_root, r["path"]), args.frames_root,
             args.short_side, args.quality) for r in rows]
    metas = []
    with Pool(args.workers) as pool:
        for i, m in enumerate(pool.imap_unordered(process, jobs, chunksize=8), 1):
            metas.append(m)
            if i % 250 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} clips", flush=True)
    metas.sort(key=lambda m: m["clip_id"])
    write_csv(metas, args.meta_out)
    bad = [m for m in metas if m["status"] in ("unreadable", "decode_failed")]
    fb = sum(m["status"] == "fallback" for m in metas)
    print(f"done. unreadable/failed: {len(bad)}, fallback frames used in {fb} clips")
    for m in bad[:10]:
        print("  ", m)


if __name__ == "__main__":
    main()
