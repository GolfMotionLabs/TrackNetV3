#!/usr/bin/env python
"""
Build background images for golf shot sequences.

The background (median.npz) is required by dataset.py at training time
when bg_mode='concat'. This script produces better backgrounds than the
rough placeholder generated during frame conversion.

Modes
-----
session  (default)
    Cross-shot median: takes one representative frame from each shot,
    stacks them, and computes the per-pixel median.
    Best for fixed-camera setups — the ball appears at a different position
    in every shot so the median suppresses it while preserving the static scene.
    Outputs the SAME background for every shot in the split.

shot
    Per-shot median: computes the median of all frames within each shot
    individually. The ball moves within the clip so it is partially suppressed,
    but less reliably than session mode. Useful when shots come from different
    camera positions.

manual
    Uses a single user-provided image (--bg_image) for all shots.
    Best quality when a clean empty-scene frame is available.

Output
------
data/{split}/{shot_name}/median.npz   (RGB float64, original frame resolution)

This is the primary lookup path in dataset.py and takes precedence over the
fallback placeholder at frame/{shot_name}/median.npz.

Usage
-----
    # Recommended: cross-shot median for a fixed camera session
    python tools/build_golf_backgrounds.py --split train

    # Per-shot median
    python tools/build_golf_backgrounds.py --split train --mode shot

    # User-provided clean background
    python tools/build_golf_backgrounds.py --split train --mode manual --bg_image bg.png

    # Save a preview PNG alongside each median.npz to inspect quality
    python tools/build_golf_backgrounds.py --split train --save_preview
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def load_frame(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert('RGB'))


def pick_representative_frame(frame_dir: Path) -> Path:
    """Return the middle frame path from a shot's frame directory."""
    pngs = sorted(frame_dir.glob('*.png'), key=lambda p: int(p.stem))
    pngs = [p for p in pngs if p.stem != 'median']  # exclude any stray files
    return pngs[len(pngs) // 2]


def load_shot_frames(frame_dir: Path) -> np.ndarray:
    """Load all PNG frames from a shot frame directory as a (N, H, W, 3) array."""
    pngs = sorted(frame_dir.glob('*.png'), key=lambda p: int(p.stem))
    return np.stack([load_frame(p) for p in pngs], axis=0)


def save_background(median: np.ndarray, out_path: Path, save_preview: bool) -> None:
    np.savez(str(out_path), median=median)
    if save_preview:
        preview_path = out_path.with_suffix('.png')
        Image.fromarray(median.clip(0, 255).astype('uint8')).save(preview_path)
        print(f'  Preview: {preview_path}')


def build_session_background(
    split_dir: Path,
    shot_names: list[str],
    save_preview: bool,
) -> None:
    """Compute one cross-shot median and write it to all shots."""
    print('Mode: session  (cross-shot median)')

    # Collect one representative frame per shot
    representative_frames = []
    for shot_name in shot_names:
        frame_dir = split_dir / shot_name / 'frame' / shot_name
        if not frame_dir.is_dir():
            print(f'  [SKIP] {shot_name}: frame dir not found')
            continue
        rep = pick_representative_frame(frame_dir)
        representative_frames.append(load_frame(rep))
        print(f'  Loaded {rep.name} from {shot_name}')

    if not representative_frames:
        raise SystemExit('No frames loaded — nothing to compute.')

    stack = np.stack(representative_frames, axis=0)  # (N, H, W, 3)
    median = np.median(stack, axis=0)                 # (H, W, 3)
    print(f'  Median computed from {len(representative_frames)} frames  '
          f'[shape {stack.shape[1]}x{stack.shape[2]}]')

    for shot_name in shot_names:
        out_path = split_dir / shot_name / 'median.npz'
        save_background(median, out_path, save_preview)
        print(f'  Saved -> {out_path}')


def build_shot_backgrounds(
    split_dir: Path,
    shot_names: list[str],
    save_preview: bool,
) -> None:
    """Compute a per-shot median from each shot's own frames."""
    print('Mode: shot  (per-shot median)')

    for shot_name in shot_names:
        frame_dir = split_dir / shot_name / 'frame' / shot_name
        if not frame_dir.is_dir():
            print(f'  [SKIP] {shot_name}: frame dir not found')
            continue

        frames = load_shot_frames(frame_dir)
        median = np.median(frames, axis=0)
        out_path = split_dir / shot_name / 'median.npz'
        save_background(median, out_path, save_preview)
        print(f'  {shot_name}: median of {len(frames)} frames -> {out_path}')


def build_manual_backgrounds(
    split_dir: Path,
    shot_names: list[str],
    bg_image: Path,
    save_preview: bool,
) -> None:
    """Copy a user-provided background image to all shots."""
    print(f'Mode: manual  (bg_image={bg_image})')

    if not bg_image.is_file():
        raise SystemExit(f'bg_image not found: {bg_image}')

    bg = load_frame(bg_image).astype(np.float64)

    for shot_name in shot_names:
        out_path = split_dir / shot_name / 'median.npz'
        save_background(bg, out_path, save_preview)
        print(f'  {shot_name} -> {out_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data_dir', default='data',
                        help='Root dataset directory (default: data)')
    parser.add_argument('--split', default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--mode', default='shot',
                        choices=['session', 'shot', 'manual'],
                        help='Background source mode (default: shot)')
    parser.add_argument('--bg_image',
                        help='Path to clean background image (manual mode only)')
    parser.add_argument('--save_preview', action='store_true',
                        help='Save a .png preview alongside each median.npz')
    args = parser.parse_args()

    split_dir = Path(args.data_dir) / args.split
    if not split_dir.is_dir():
        raise SystemExit(f'Split directory not found: {split_dir}\n'
                         'Run convert_golf_dataset.py first.')

    shot_names = sorted([
        d.name for d in split_dir.iterdir()
        if d.is_dir() and (d / 'frame' / d.name).is_dir()
    ])
    if not shot_names:
        raise SystemExit(f'No converted shots found in {split_dir}')

    print(f'Found {len(shot_names)} shot(s) in {split_dir}')

    if args.mode == 'session':
        build_session_background(split_dir, shot_names, args.save_preview)
    elif args.mode == 'shot':
        build_shot_backgrounds(split_dir, shot_names, args.save_preview)
    elif args.mode == 'manual':
        if not args.bg_image:
            raise SystemExit('--bg_image is required for manual mode')
        build_manual_backgrounds(
            split_dir, shot_names, Path(args.bg_image), args.save_preview
        )

    print('\nDone. Delete stale dataset cache files before training:')
    print(f'  {Path(args.data_dir) / "img_config_*.npz"}')
    print(f'  {Path(args.data_dir) / "data_l*_s*_*.npz"}')


if __name__ == '__main__':
    main()
