#!/usr/bin/env python
"""
Convert golf video clips + CVAT annotations into TrackNetV3-compatible dataset layout.

Supports two annotation formats exported from CVAT:
  cvat_xml  — "CVAT for video 1.1" XML (default, from a CVAT video task with Points labels)
  csv       — Simple CSV with columns: Frame, Visibility, X, Y

Input layout (--src_dir):
    {src_dir}/
        {clip_name}.mp4       # video clip
        {clip_name}.xml       # CVAT XML export  (or .csv for csv mode)

Output layout (--dst_dir  default: data):
    {dst_dir}/{split}/match{match_id}/
        video/
            {clip_name}.mp4
        frame/
            {clip_name}/
                0.png, 1.png, ...
                median.npz
        csv/
            {clip_name}_ball.csv    columns: Frame, Visibility, X, Y

Usage:
    python tools/convert_golf_dataset.py \\
        --src_dir data/raw/session1 \\
        --split train --match_id 1

    python tools/convert_golf_dataset.py \\
        --src_dir data/raw/session2 \\
        --split val --match_id 2 --ann_format csv

After conversion delete any stale .npz cache files in the data directory
(img_config_*.npz, data_l*_s*_*.npz) before running train.py.
"""

import os
import shutil
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from xml.etree import ElementTree as ET


IMG_FORMAT = 'png'


# ---------------------------------------------------------------------------
# Annotation parsers
# ---------------------------------------------------------------------------

def _parse_cvat_track(xml_path: Path, num_frames: int) -> list[dict]:
    """Parse CVAT for video 1.1 XML (video task, Points label type).

    Handles:
    - Track-based exports where all frames (incl. interpolated) are present.
    - Tracks that do not cover the full clip (missing frames → invisible).
    - occluded="1" is treated as visible because a coordinate is still provided.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Locate the ball track by label name (case-insensitive)
    track_el = None
    for t in root.findall('track'):
        label = t.get('label', '').lower()
        if label in ('ball', 'golf_ball', 'golf ball', 'shuttlecock'):
            track_el = t
            break
    if track_el is None:
        tracks = root.findall('track')
        if tracks:
            track_el = tracks[0]
        else:
            raise ValueError(
                f'{xml_path}: no <track> element found. '
                'Export the CVAT task as "CVAT for video 1.1" format.'
            )

    # Build frame_id → (x, y, visibility) mapping from the track elements
    frame_ann: dict[int, tuple[int, int, int]] = {}
    for pts in track_el.findall('points'):
        frame_id = int(pts.get('frame'))
        outside = int(pts.get('outside', '0'))
        if outside:
            frame_ann[frame_id] = (0, 0, 0)
        else:
            raw = pts.get('points', '0,0')
            # Take only the first point (e.g. "123.45,234.56" or "x1,y1;x2,y2")
            first = raw.split(';')[0]
            x, y = (float(v) for v in first.split(','))
            frame_ann[frame_id] = (round(x), round(y), 1)

    # Fill all frames; frames absent from the track are invisible
    rows = []
    for f in range(num_frames):
        if f in frame_ann:
            x, y, vis = frame_ann[f]
        else:
            x, y, vis = 0, 0, 0
        rows.append({'Frame': f, 'Visibility': vis, 'X': x, 'Y': y})

    return rows


def _parse_simple_csv(csv_path: Path, num_frames: int) -> list[dict]:
    """Parse a CSV with columns: Frame, Visibility, X, Y.

    Missing frames are filled as invisible.
    """
    df = pd.read_csv(csv_path).sort_values('Frame').fillna(0)
    frame_ann = {
        int(row['Frame']): (int(row['X']), int(row['Y']), int(row['Visibility']))
        for _, row in df.iterrows()
    }
    rows = []
    for f in range(num_frames):
        if f in frame_ann:
            x, y, vis = frame_ann[f]
        else:
            x, y, vis = 0, 0, 0
        rows.append({'Frame': f, 'Visibility': vis, 'X': x, 'Y': y})
    return rows


# ---------------------------------------------------------------------------
# Frame extraction and median
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, out_dir: Path, max_frames: int | None) -> tuple[list, int]:
    """Extract all frames from a video and save as {frame_id}.png.

    Returns (frames_rgb_list, frame_count).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    frames_rgb = []
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_id >= max_frames:
            break
        cv2.imwrite(str(out_dir / f'{frame_id}.{IMG_FORMAT}'), frame)
        frames_rgb.append(frame[..., ::-1])  # BGR → RGB
        frame_id += 1

    cap.release()
    return frames_rgb, frame_id


def save_median(frames_rgb: list, out_dir: Path) -> None:
    """Compute per-pixel median over all frames and save as median.npz."""
    median = np.median(np.stack(frames_rgb, axis=0), axis=0)
    np.savez(str(out_dir / 'median.npz'), median=median)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--src_dir', required=True,
                        help='Directory with .mp4 clips and annotation files')
    parser.add_argument('--dst_dir', default='data',
                        help='Root dataset directory (default: data)')
    parser.add_argument('--split', default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split (default: train)')
    parser.add_argument('--match_id', type=int, default=1,
                        help='Match ID for output directory naming (default: 1)')
    parser.add_argument('--ann_format', default='cvat_xml',
                        choices=['cvat_xml', 'csv'],
                        help='Annotation format (default: cvat_xml)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames to extract per clip (default: all)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print planned actions without writing any files')
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    match_out = Path(args.dst_dir) / args.split / f'match{args.match_id}'
    video_out = match_out / 'video'
    frame_out = match_out / 'frame'
    csv_out = match_out / 'csv'

    if not args.dry_run:
        for d in (video_out, frame_out, csv_out):
            d.mkdir(parents=True, exist_ok=True)

    ann_ext = '.xml' if args.ann_format == 'cvat_xml' else '.csv'
    clips = sorted(src_dir.glob('*.mp4'))

    if not clips:
        print(f'No .mp4 files found in {src_dir}')
        return

    ok_count = 0
    skip_count = 0

    for mp4 in clips:
        rally_id = mp4.stem
        ann_path = src_dir / (rally_id + ann_ext)

        if not ann_path.exists():
            print(f'[SKIP] {mp4.name}: annotation not found ({ann_path.name})')
            skip_count += 1
            continue

        if args.dry_run:
            print(f'[DRY]  {mp4.name} → match{args.match_id}/frame/{rally_id}/, csv/{rally_id}_ball.csv')
            continue

        print(f'Processing {mp4.name} ...')

        # 1. Extract frames
        rally_frame_dir = frame_out / rally_id
        frames_rgb, num_frames = extract_frames(mp4, rally_frame_dir, args.max_frames)
        print(f'  {num_frames} frames extracted')

        if num_frames == 0:
            print(f'  [ERROR] no frames extracted, skipping')
            skip_count += 1
            continue

        # 2. Median image
        save_median(frames_rgb, rally_frame_dir)

        # 3. Parse annotation
        if args.ann_format == 'cvat_xml':
            rows = _parse_cvat_track(ann_path, num_frames)
        else:
            rows = _parse_simple_csv(ann_path, num_frames)

        visible = sum(r['Visibility'] for r in rows)
        print(f'  {visible}/{num_frames} frames labeled visible')

        if visible == 0:
            print(f'  [WARN] no visible labels — check annotation file')

        # 4. Write label CSV
        label_csv = csv_out / f'{rally_id}_ball.csv'
        pd.DataFrame(rows, columns=['Frame', 'Visibility', 'X', 'Y']).to_csv(
            label_csv, index=False
        )

        # 5. Copy video (skip if already there)
        dst_video = video_out / mp4.name
        if not dst_video.exists():
            shutil.copy2(mp4, dst_video)

        ok_count += 1

    print(f'\nDone: {ok_count} clip(s) converted, {skip_count} skipped.')
    print(f'Output: {match_out}')
    if ok_count:
        print(
            '\nBefore training, delete stale cache files if they exist:\n'
            f'  {Path(args.dst_dir) / "img_config_*.npz"}\n'
            f'  {Path(args.dst_dir) / "data_l*_s*_*.npz"}'
        )


if __name__ == '__main__':
    main()
