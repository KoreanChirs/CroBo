#!/usr/bin/env python3

###### example usage ######
# symlink mode
"""
python3 class_organize.py \
  --videos_dir data/kinetics400/train \
  --csv_path data/kinetics400/annotations/train.csv \
  --out_dir data/kinetics400/train_by_class \
  --mode symlink
"""
# move mode
"""
python3 class_organize.py \
  --videos_dir data/kinetics400/train \
  --csv_path data/kinetics400/annotations/train.csv \
  --out_dir data/kinetics400/train_by_class \
  --mode move
"""
###### example usage ######

import os
import csv
import argparse
import shutil
from pathlib import Path

def safe_label(name: str) -> str:
    """
    Folder-safe label name.
    Kinetics labels are usually safe, but just in case:
    """
    return name.strip().replace("/", "_")

def build_expected_filename(youtube_id: str, timestart: str, time_end: str) -> str:
    """
    Matches your file pattern: KLAJyjasmY8_000172_000182.mp4
    CSV has timestart/time_end in seconds; filenames use 6-digit zero padding.
    """
    ts = int(timestart)
    te = int(time_end)
    return f"{youtube_id}_{ts:06d}_{te:06d}.mp4"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_move(src: Path, dst: Path, mode: str, overwrite: bool):
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return "exists"

    if mode == "symlink":
        # relative symlink (more portable if you move the dataset folder)
        rel_src = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel_src)
        return "linked"
    elif mode == "move":
        ensure_dir(dst.parent)
        shutil.move(str(src), str(dst))
        return "moved"
    elif mode == "copy":
        ensure_dir(dst.parent)
        shutil.copy2(str(src), str(dst))
        return "copied"
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    ap = argparse.ArgumentParser(
        description="Organize Kinetics-400 train videos into class folders using train.csv"
    )
    ap.add_argument("--videos_dir", default="data/kinetics400/train",
                    help="Directory containing mp4 files (flat).")
    ap.add_argument("--csv_path", default="data/kinetics400/annotations/train.csv",
                    help="Path to train.csv (columns: label,youtube_id,timestart,time_end,...)")
    ap.add_argument("--out_dir", default="data/kinetics400/train_by_class",
                    help="Output directory where class folders will be created.")
    ap.add_argument("--mode", choices=["symlink", "move", "copy"], default="symlink",
                    help="How to place videos into class folders. symlink recommended.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing files/links in output.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N CSV rows (0 = all). Useful for testing.")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir).resolve()
    csv_path = Path(args.csv_path).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not videos_dir.is_dir():
        raise FileNotFoundError(f"videos_dir not found: {videos_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"csv_path not found: {csv_path}")

    ensure_dir(out_dir)

    total = 0
    found = 0
    missing = 0
    done = 0
    skipped_exists = 0

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"label", "youtube_id", "timestart", "time_end"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must include columns {required}, got {reader.fieldnames}")

        for row in reader:
            total += 1
            if args.limit and total > args.limit:
                break

            label = safe_label(row["label"])
            youtube_id = row["youtube_id"].strip()
            ts = row["timestart"]
            te = row["time_end"]

            fname = build_expected_filename(youtube_id, ts, te)
            src = videos_dir / fname
            if not src.exists():
                # Some datasets may have .mkv etc, but you said mp4. Keep simple.
                missing += 1
                continue

            found += 1
            class_dir = out_dir / label
            ensure_dir(class_dir)
            dst = class_dir / fname

            status = link_or_move(src, dst, args.mode, args.overwrite)
            if status == "exists":
                skipped_exists += 1
            else:
                done += 1

            if total % 5000 == 0:
                print(f"[{total}] found={found}, missing={missing}, done={done}, exists={skipped_exists}")

    print("\n=== Summary ===")
    print(f"CSV rows processed : {total}")
    print(f"Videos found       : {found}")
    print(f"Videos missing     : {missing}")
    print(f"Placed ({args.mode}): {done}")
    print(f"Skipped (exists)   : {skipped_exists}")
    print(f"Output dir         : {out_dir}")

if __name__ == "__main__":
    main()