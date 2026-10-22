"""Build data/index.csv from the raw VideoBadminton folder.

Usage (inspect first, then write the index):
    python -m src.make_index --video-root /path/to/VideoBadminton --inspect
    python -m src.make_index --video-root /path/to/VideoBadminton \
        --session-key date_time

Label: taken from the clip's parent folder name (default) or from a regex
group `label` in the filename (--label-from filename --label-regex ...).

Session: the dataset paper states that file names encode recording date and
time. The exact pattern is not documented, so the regex is configurable:
  --session-regex must contain named groups `date` and (optionally) `time`.
  --session-key date       -> one session per recording day
  --session-key date_time  -> one session per (day, time)
Run --inspect to see real file names and how many sessions each key gives,
then choose. The default regex is a guess and MUST be checked.
"""
import argparse
import collections
import os
import re
import sys

from src.utils import write_csv, save_json, REPO_ROOT

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
DEFAULT_SESSION_REGEX = r"(?P<date>(?:19|20)\d{2}[-_]?\d{2}[-_]?\d{2})(?!\d)[-_ T]?(?P<time>(?:\d{4}(?:\d{2})?|\d{2}[-:.]\d{2}(?:[-:.]\d{2})?)(?!\d))?"


def list_videos(root):
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(VIDEO_EXT):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def parse(path, args, sess_re, label_re):
    stem = os.path.splitext(os.path.basename(path))[0]
    if args.label_from == "folder":
        label = os.path.basename(os.path.dirname(path))
    else:
        m = label_re.search(stem)
        label = m.group("label") if m else None
    m = sess_re.search(stem)
    date = m.group("date") if m else None
    time = m.groupdict().get("time") if m else None
    return stem, label, date, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "index.csv"))
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--label-from", choices=["folder", "filename"], default="folder")
    ap.add_argument("--label-regex", default=r"(?P<label>[A-Za-z][A-Za-z_ \-]+)$")
    ap.add_argument("--session-regex", default=DEFAULT_SESSION_REGEX)
    ap.add_argument("--session-key", choices=["date", "date_time"], default="date_time")
    args = ap.parse_args()

    sess_re = re.compile(args.session_regex)
    label_re = re.compile(args.label_regex)
    paths = list_videos(args.video_root)
    if not paths:
        sys.exit(f"No videos found under {args.video_root}")

    rows, failed = [], []
    for p in paths:
        stem, label, date, time = parse(p, args, sess_re, label_re)
        if label is None or date is None:
            failed.append(p)
            continue
        rows.append(dict(path=os.path.relpath(p, args.video_root), stem=stem,
                         label=label, date=date, time=time or ""))

    labels = sorted({r["label"] for r in rows})
    by_date = collections.Counter(r["date"] for r in rows)
    by_dt = collections.Counter((r["date"], r["time"]) for r in rows)

    if args.inspect:
        print(f"videos found: {len(paths)} | parsed: {len(rows)} | failed: {len(failed)}")
        print("\n20 example file names:")
        for p in paths[:: max(1, len(paths) // 20)][:20]:
            print("  ", os.path.relpath(p, args.video_root))
        if failed:
            print("\nexamples that FAILED to parse:")
            for p in failed[:10]:
                print("  ", os.path.relpath(p, args.video_root))
        print("\nparsed examples (CHECK that date/time are the recording session, not clip ids):")
        for r in rows[:: max(1, len(rows) // 10)][:10]:
            print(f"   date={r['date']!s:10s} time={r['time']!s:8s} label={r['label']:25s} <- {r['stem']}")
        print(f"\nclasses ({len(labels)}):")
        cls_count = collections.Counter(r["label"] for r in rows)
        for c in labels:
            print(f"   {c:35s} {cls_count[c]}")
        print(f"\nsessions with key=date:      {len(by_date)} "
              f"(clips/session min {min(by_date.values())}, max {max(by_date.values())})")
        print(f"sessions with key=date_time: {len(by_dt)} "
              f"(clips/session min {min(by_dt.values())}, max {max(by_dt.values())})")
        print("\nIf date_time gives ~1 clip per session, `time` is the clip time, "
              "not the recording start: use --session-key date.")
        return

    if failed:
        sys.exit(f"{len(failed)} files failed to parse; run --inspect and fix the regex.")

    # stable clip ids; disambiguate duplicate stems
    seen = collections.Counter()
    for r in rows:
        seen[r["stem"]] += 1
        r["clip_id"] = r["stem"] if seen[r["stem"]] == 1 else f"{r['stem']}__{seen[r['stem']]}"
        r["session"] = r["date"] if args.session_key == "date" else f"{r['date']}_{r['time']}"
        r["label_idx"] = labels.index(r["label"])
    fields = ["clip_id", "path", "label", "label_idx", "session", "date", "time"]
    write_csv(rows, args.out, fields)
    save_json({"classes": labels, "video_root": os.path.abspath(args.video_root),
               "session_key": args.session_key, "session_regex": args.session_regex},
              os.path.join(os.path.dirname(args.out), "index_meta.json"))
    n_sess = len({r['session'] for r in rows})
    print(f"wrote {args.out}: {len(rows)} clips, {len(labels)} classes, {n_sess} sessions")


if __name__ == "__main__":
    main()
