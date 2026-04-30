#!/usr/bin/env python
"""
Crop all frames, labels, and backgrounds to a fixed launch-zone ROI.

Reads from a converted golf dataset directory and writes a new directory
with every frame, CSV, and median.npz cropped to the specified region.
Labels whose ball coordinates fall outside the ROI are set to Visibility=0.

Default ROI (session 2026-04-30, 1080x1920 portrait frames):
    x_min=196, y_min=24, width=800, height=1475  →  x=[196,996), y=[24,1499)
    Measured from bottom-left corner (196,1499) of the Paint selection box.

Network resolution note
-----------------------
After cropping, frames are 800x1475 (portrait, aspect ~0.54:1).
Before training, set WIDTH=288, HEIGHT=512 in utils/general.py so the
dataset loader resizes to a portrait resolution that preserves this ratio
(aspect 0.5625:1, <4% distortion). Both 288 and 512 are divisible by 32,
which is required by the U-Net downsampling depth.

Usage
-----
    # Crop data/train → data_roi/train  (default ROI)
    python tools/build_roi_clips.py --split train

    # Custom ROI
    python tools/build_roi_clips.py --split train \\
        --x_min 196 --y_min 24 --width 800 --height 1475

    # Different source / destination roots
    python tools/build_roi_clips.py --split train \\
        --src_dir data --dst_dir data_roi

    # Dry run
    python tools/build_roi_clips.py --split train --dry_run
"""

import argparse
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image


def crop_image(img_array: np.ndarray, x_min: int, y_min: int,
               width: int, height: int) -> np.ndarray:
    return img_array[y_min:y_min + height, x_min:x_min + width]


def remap_labels(df: pd.DataFrame, x_min: int, y_min: int,
                 width: int, height: int) -> pd.DataFrame:
    """Shift coordinates into ROI space; mark out-of-bounds frames invisible."""
    df = df.copy()
    for i, row in df.iterrows():
        if row['Visibility'] == 0:
            continue
        x_new = row['X'] - x_min
        y_new = row['Y'] - y_min
        if 0 <= x_new < width and 0 <= y_new < height:
            df.at[i, 'X'] = round(x_new)
            df.at[i, 'Y'] = round(y_new)
        else:
            df.at[i, 'Visibility'] = 0
            df.at[i, 'X'] = 0
            df.at[i, 'Y'] = 0
    return df


def process_shot(
    src_shot_dir: Path,
    dst_shot_dir: Path,
    shot_name: str,
    x_min: int,
    y_min: int,
    width: int,
    height: int,
    dry_run: bool,
) -> dict:
    src_frame_dir = src_shot_dir / 'frame' / shot_name
    src_csv = src_shot_dir / 'csv' / f'{shot_name}.csv'
    src_median = src_shot_dir / 'median.npz'

    dst_frame_dir = dst_shot_dir / 'frame' / shot_name
    dst_csv_dir = dst_shot_dir / 'csv'
    dst_csv = dst_csv_dir / f'{shot_name}.csv'
    dst_median = dst_shot_dir / 'median.npz'

    issues = []

    if not src_frame_dir.is_dir():
        return {'shot': shot_name, 'frames': 0, 'skipped': True,
                'reason': 'frame dir not found'}
    if not src_csv.exists():
        issues.append('CSV not found')
    if not src_median.exists():
        issues.append('median.npz not found — run build_golf_backgrounds.py first')

    if issues and not dry_run:
        return {'shot': shot_name, 'frames': 0, 'skipped': True,
                'reason': '; '.join(issues)}

    pngs = sorted(src_frame_dir.glob('*.png'), key=lambda p: int(p.stem))
    n_frames = len(pngs)

    if dry_run:
        return {'shot': shot_name, 'frames': n_frames, 'skipped': False}

    dst_frame_dir.mkdir(parents=True, exist_ok=True)
    dst_csv_dir.mkdir(parents=True, exist_ok=True)

    # Crop frames
    for png in pngs:
        img = np.array(Image.open(png).convert('RGB'))
        cropped = crop_image(img, x_min, y_min, width, height)
        Image.fromarray(cropped).save(dst_frame_dir / png.name)

    # Remap labels
    df = pd.read_csv(src_csv)
    col_order = list(df.columns)
    df = remap_labels(df, x_min, y_min, width, height)
    df[col_order].to_csv(dst_csv, index=False)

    # Crop background
    median_full = np.load(src_median)['median']  # (H, W, 3) float
    median_cropped = crop_image(median_full.astype(np.uint8), x_min, y_min, width, height)
    np.savez(str(dst_median), median=median_cropped.astype(np.float64))

    # Copy shot-level placeholder median if present (fallback path)
    src_fallback = src_shot_dir / 'frame' / shot_name / 'median.npz'
    dst_fallback = dst_frame_dir / 'median.npz'
    if src_fallback.exists():
        fallback_full = np.load(src_fallback)['median']
        fallback_cropped = crop_image(
            fallback_full.astype(np.uint8), x_min, y_min, width, height
        )
        np.savez(str(dst_fallback), median=fallback_cropped.astype(np.float64))

    return {'shot': shot_name, 'frames': n_frames, 'skipped': False}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--src_dir', default='data',
                        help='Source dataset root (default: data)')
    parser.add_argument('--dst_dir', default='data_roi',
                        help='Destination dataset root (default: data_roi)')
    parser.add_argument('--split', default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--x_min', type=int, default=196,
                        help='ROI left edge in original frame (default: 196)')
    parser.add_argument('--y_min', type=int, default=24,
                        help='ROI top edge in original frame (default: 24)')
    parser.add_argument('--width', type=int, default=800,
                        help='ROI width in pixels (default: 800)')
    parser.add_argument('--height', type=int, default=1475,
                        help='ROI height in pixels (default: 1475)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print planned actions without writing files')
    args = parser.parse_args()

    src_split = Path(args.src_dir) / args.split
    dst_split = Path(args.dst_dir) / args.split

    if not src_split.is_dir():
        raise SystemExit(f'Source split not found: {src_split}')

    x_min, y_min = args.x_min, args.y_min
    width, height = args.width, args.height
    x_max, y_max = x_min + width, y_min + height

    print(f'ROI: x=[{x_min}, {x_max})  y=[{y_min}, {y_max})  '
          f'size={width}x{height}')
    print(f'src: {src_split}  ->  dst: {dst_split}')
    if args.dry_run:
        print('(dry run — no files written)\n')

    shot_dirs = sorted([
        d for d in src_split.iterdir()
        if d.is_dir() and (d / 'frame' / d.name).is_dir()
    ])
    if not shot_dirs:
        raise SystemExit(f'No converted shots found in {src_split}')

    ok, skipped = 0, 0
    for shot_dir in shot_dirs:
        shot_name = shot_dir.name
        dst_shot_dir = dst_split / shot_name
        result = process_shot(
            shot_dir, dst_shot_dir, shot_name,
            x_min, y_min, width, height,
            dry_run=args.dry_run,
        )
        if result['skipped']:
            print(f'[SKIP] {shot_name}: {result.get("reason", "")}')
            skipped += 1
        else:
            print(f'  {shot_name}: {result["frames"]} frames cropped  '
                  f'-> {dst_shot_dir}')
            ok += 1

    print(f'\nDone: {ok} shot(s) cropped, {skipped} skipped.')
    if not args.dry_run and ok:
        print(f'\nOutput: {dst_split}')
        print('\nBefore training with ROI data:')
        print('  1. Set WIDTH=288, HEIGHT=512 in utils/general.py')
        print(f'  2. Pass --data_dir {args.dst_dir} to train.py')
        print('  3. Delete stale cache files: '
              f'{Path(args.dst_dir) / "img_config_*.npz"} '
              f'{Path(args.dst_dir) / "data_l*_s*_*.npz"}')


if __name__ == '__main__':
    main()
