#!/usr/bin/env python3
"""
Convert all SVO2 files found under a directory to MP4, placing each output
at  <session>/recordings/MP4/<camera_id>.mp4  — the layout expected by
convert_droid_data_to_lerobot.py.

SVO2 files are MCAP containers with HEVC (H.265) video inside.
This script extracts the HEVC bitstream and re-encodes to H264 MP4 using
ffmpeg — no ZED SDK or container required.

Input tree assumed:
    <input_dir>/
      <session>/
        recordings/
          SVO/
            <camera_id>.svo2
          MP4/            <- created by this script
            <camera_id>.mp4
        trajectory.h5

Usage:
    python convert_svo2_to_mp4.py --input_dir /path/to/data [--jobs 4]
"""

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mcap.reader import make_reader

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)


def extract_hevc(svo2_path: Path) -> bytes:
    """Extract the raw HEVC bitstream from an SVO2 (MCAP) file."""
    chunks = []
    with open(svo2_path, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages():
            if "side_by_side" in channel.topic:
                chunks.append(message.data[8:])  # skip 8-byte per-frame header
    if not chunks:
        raise ValueError(f"No side_by_side frames found in {svo2_path}")
    return b"".join(chunks)


def convert_one(svo2: Path) -> tuple[Path, bool]:
    """Convert a single SVO2 file to MP4. Returns (out_file, success)."""
    mp4_dir = svo2.parent.parent / "MP4"
    out_file = mp4_dir / (svo2.stem + ".mp4")

    if out_file.exists() and out_file.stat().st_size > 0:
        log.info("[SKIP] %s (already exists)", out_file)
        return out_file, True

    mp4_dir.mkdir(parents=True, exist_ok=True)
    log.info("[CONV] %s ...", svo2)

    try:
        hevc_data = extract_hevc(svo2)
    except Exception as e:
        log.error("[FAIL] could not read %s: %s", svo2, e)
        return out_file, False

    # Decode HEVC, crop left stereo half (1280x720), encode to H264 MP4
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "hevc", "-i", "pipe:0",
            "-vf", "crop=1280:720:0:0",
            "-c:v", "libx264", "-crf", "18",
            str(out_file),
        ],
        input=hevc_data,
        capture_output=True,
    )

    if out_file.exists() and out_file.stat().st_size > 0:
        log.info("[DONE] %s", out_file)
        return out_file, True
    else:
        log.error("[FAIL] output missing or empty — %s", svo2)
        log.error("ffmpeg stderr: %s", result.stderr.decode()[-500:])
        out_file.unlink(missing_ok=True)
        return out_file, False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True, help="Root directory containing session folders with SVO2 files")
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel conversions (default: 4)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        log.error("input_dir does not exist: %s", input_dir)
        sys.exit(1)

    svo2_files = sorted(input_dir.rglob("*.svo2"))
    if not svo2_files:
        log.warning("No .svo2 files found under %s", input_dir)
        sys.exit(0)

    log.info("Found %d SVO2 files. Running %d conversions at a time.", len(svo2_files), args.jobs)

    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(convert_one, f): f for f in svo2_files}
        for future in as_completed(futures):
            _, success = future.result()
            if success:
                done += 1
            else:
                failed += 1

    log.info("Conversion complete. %d succeeded, %d failed.", done, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
