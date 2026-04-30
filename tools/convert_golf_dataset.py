#!/usr/bin/env python
"""
Convert golf impact-frame folders + CVAT image-task XML into the golf dataset layout.

Input layout:
    {frames_dir}/
        {shot_name}/
            frame_XXXXXX.png        # non-contiguous original frame numbers are fine

    {ann_file}                      # CVAT "CVAT 1.1 for images" XML
                                    # image names like: IMG_1660/IMG_1660_impact_frame_0285.png

Output layout:
    {dst_dir}/{split}/{shot_name}/
        frame/
            {shot_name}/
                0.png, 1.png, ...   # re-indexed 0-based, sorted by original frame number
                median.npz
        csv/
            {shot_name}.csv         # columns: Frame, Visibility, X, Y [, blur_type]

Shots present in {frames_dir} but absent from the XML are included with Visibility=0
on every frame. Use --skip_unannotated to omit them instead.

Usage:
    python tools/convert_golf_dataset.py \\
        --frames_dir impact_frames \\
        --ann_file xml/annotations.xml \\
        --split train

    # dry run first
    python tools/convert_golf_dataset.py \\
        --frames_dir impact_frames \\
        --ann_file xml/annotations.xml \\
        --split train --dry_run

After conversion delete stale cache files before training:
    del data\\img_config_*.npz data\\data_l*_s*_*.npz
"""

import os
import re
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict
from xml.etree import ElementTree as ET


IMG_FORMAT = 'png'
_FRAME_NUM_RE = re.compile(r'(\d+)\.png$', re.IGNORECASE)


def _frame_num(filename: str) -> int:
    m = _FRAME_NUM_RE.search(filename)
    if m is None:
        raise ValueError(f'Cannot parse frame number from: {filename}')
    return int(m.group(1))


def parse_cvat_image_xml(ann_file: Path) -> dict[str, dict[int, dict]]:
    """Parse a CVAT 'for images' XML and return per-shot annotation dicts.

    Returns:
        {shot_name: {orig_frame_num: {'x': int, 'y': int, 'blur_type': str}}}

    Frames without a <points> child are absent from the inner dict (invisible).
    """
    tree = ET.parse(ann_file)
    root = tree.getroot()

    clips: dict[str, dict[int, dict]] = defaultdict(dict)

    for img_el in root.findall('image'):
        name = img_el.get('name', '')
        parts = name.replace('\\', '/').split('/')
        shot_name = parts[0]
        orig_num = _frame_num(parts[-1])

        pts_el = img_el.find('points[@label="ball"]')
        if pts_el is None:
            continue

        raw = pts_el.get('points', '0,0')
        x, y = (float(v) for v in raw.split(';')[0].split(','))

        blur_type = ''
        for attr in pts_el.findall('attribute'):
            if attr.get('name') == 'visibility':
                blur_type = (attr.text or '').strip()
                break

        clips[shot_name][orig_num] = {'x': round(x), 'y': round(y), 'blur_type': blur_type}

    return dict(clips)


def copy_and_index_frames(
    src_dir: Path,
    dst_dir: Path,
    ann_map: dict[int, dict],
) -> list[dict]:
    """Copy frames as 0.png, 1.png... into dst_dir and build label rows.

    Frames sorted by original frame number, re-indexed 0-based.
    Returns list of row dicts: Frame, Visibility, X, Y, blur_type.
    """
    png_files = sorted(src_dir.glob('*.png'), key=lambda p: _frame_num(p.name))
    if not png_files:
        raise ValueError(f'No .png files found in {src_dir}')

    dst_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for new_idx, src_path in enumerate(png_files):
        shutil.copy2(src_path, dst_dir / f'{new_idx}.{IMG_FORMAT}')
        orig_num = _frame_num(src_path.name)
        if orig_num in ann_map:
            ann = ann_map[orig_num]
            rows.append({'Frame': new_idx, 'Visibility': 1,
                         'X': ann['x'], 'Y': ann['y'], 'blur_type': ann['blur_type']})
        else:
            rows.append({'Frame': new_idx, 'Visibility': 0, 'X': 0, 'Y': 0, 'blur_type': ''})

    return rows


def generate_median(frame_dir: Path) -> None:
    """Compute per-pixel RGB median over all frames and save as median.npz."""
    pngs = sorted(frame_dir.glob('*.png'), key=lambda p: int(p.stem))
    frames = [np.array(Image.open(p).convert('RGB')) for p in pngs]
    median = np.median(np.stack(frames, axis=0), axis=0)
    np.savez(str(frame_dir / 'median.npz'), median=median)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--frames_dir', default='impact_frames',
                        help='Directory containing shot subfolders (default: impact_frames)')
    parser.add_argument('--ann_file', default='xml/annotations.xml',
                        help='CVAT image-task XML file (default: xml/annotations.xml)')
    parser.add_argument('--dst_dir', default='data',
                        help='Root dataset directory (default: data)')
    parser.add_argument('--split', default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--skip_unannotated', action='store_true',
                        help='Skip shots with no annotations in the XML')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print planned actions without writing any files')
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    ann_file = Path(args.ann_file)
    split_out = Path(args.dst_dir) / args.split

    if not frames_dir.is_dir():
        raise SystemExit(f'frames_dir not found: {frames_dir}')
    if not ann_file.is_file():
        raise SystemExit(f'ann_file not found: {ann_file}')

    print(f'Parsing {ann_file} ...')
    ann_by_shot = parse_cvat_image_xml(ann_file)
    print(f'  Annotations for {len(ann_by_shot)} shot(s): {sorted(ann_by_shot)}')

    shot_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    if not shot_dirs:
        raise SystemExit(f'No subdirectories found in {frames_dir}')

    ok, skipped = 0, 0

    for shot_dir in shot_dirs:
        shot_name = shot_dir.name
        ann_map = ann_by_shot.get(shot_name, {})

        if not ann_map and args.skip_unannotated:
            print(f'[SKIP] {shot_name}: no annotations')
            skipped += 1
            continue

        png_count = len(list(shot_dir.glob('*.png')))
        visible = len(ann_map)
        status = 'annotated' if ann_map else 'unannotated'
        print(f'{shot_name}: {png_count} frames, {visible} visible  [{status}]')

        # Output paths
        dst_frame_dir = split_out / shot_name / 'frame' / shot_name
        dst_csv = split_out / shot_name / 'csv' / f'{shot_name}.csv'

        if args.dry_run:
            print(f'  -> {dst_frame_dir}/')
            print(f'  -> {dst_csv}')
            continue

        (split_out / shot_name / 'csv').mkdir(parents=True, exist_ok=True)

        try:
            rows = copy_and_index_frames(shot_dir, dst_frame_dir, ann_map)
        except ValueError as e:
            print(f'  [ERROR] {e}')
            skipped += 1
            continue

        generate_median(dst_frame_dir)

        df = pd.DataFrame(rows, columns=['Frame', 'Visibility', 'X', 'Y', 'blur_type'])
        df.to_csv(dst_csv, index=False)

        ok += 1

    print(f'\nDone: {ok} shot(s) converted, {skipped} skipped.')
    if not args.dry_run and ok:
        print(f'Output: {split_out}')
        print('\nDelete stale cache files before training:\n'
              f'  {Path(args.dst_dir) / "img_config_*.npz"}\n'
              f'  {Path(args.dst_dir) / "data_l*_s*_*.npz"}')


if __name__ == '__main__':
    main()
