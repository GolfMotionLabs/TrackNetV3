# Task Plan

## Goal
Adapt TrackNetV3 into a golf-ball tracking pipeline for 240 fps impact-centered clips with ROI cropping, golf-specific backgrounds, heatmap debugging, and later trajectory rectification.

## Phases

| # | Phase | Status | Notes |
|---|-------|--------|-------|
| 1 | Define goal & scope | ✅ done | Goal is to validate and implement a golf-specific adaptation of TrackNetV3 for per-frame ball localization first, then trajectory cleanup later. |
| 2 | Research & discovery | ✅ done | TrackNetV3 repo and paper reviewed. Main mismatches identified: domain, object size, blur shape, background logic, dataset format, and debugging needs. |
| 3 | Design / plan approach | ✅ done | Chosen approach: reuse TrackNetV3 structure, retrain TrackNet first, add golf preprocessing/components, delay rectifier retraining until tracker works. |
| 4 | Implementation | 🔄 in progress | Build golf dataset pipeline, ROI/background preprocessing, heatmap target adaptation, debug outputs, and training/inference wrappers. |
| 5 | Testing & validation | 🔲 todo | Validate detectability via heatmaps first, then overlap fusion, then temporal smoothing / rectification. |
| 6 | Done | 🔲 todo | Pipeline produces reliable golf-ball heatmaps and stable framewise trajectories on held-out clips. |

**Status legend:** 🔲 todo · 🔄 in progress · ✅ done · ❌ blocked


### In Scope
- Golf-specific adaptation of TrackNetV3 tracking pipeline
- Dataset conversion and preprocessing for golf clips
- ROI-based training and inference
- Background image creation and concatenation support
- Golf-specific heatmap target generation
- Heatmap visualization/debugging tools
- TrackNet retraining and validation
- Basic trajectory smoothing after raw detections

### Out of Scope for Phase 1
- Full production deployment
- Multi-camera fusion
- Physics-based carry/spin/speed estimation
- Final rectifier redesign beyond simple temporal cleanup
- Full end-to-end retraining of every original repo component before tracker validation


## File Plan

### New files


- [x] `tools/convert_golf_dataset.py`
  - Converts impact_frames/{shot_name}/ folders + CVAT image-task XML into golf dataset layout.
  - Output: data/{split}/{shot_name}/frame/{shot_name}/ and csv/{shot_name}.csv (no match/ball naming).

- [x] `tools/build_golf_backgrounds.py`
  - Builds median.npz backgrounds per shot (default) or cross-shot, or from a manual image.
  - Default: per-shot median of all frames. Session mode ready for future calibration phase.

- [ ] `tools/build_roi_clips.py`
  - Crop launch-zone ROI from source clips and remap labels into ROI coordinates.

- [ ] `tools/inspect_heatmaps.py`
  - Visualize predicted heatmaps, target heatmaps, decoded peaks, and overlays for debugging.

- [ ] `configs/golf_tracknet.yaml`
  - Central config for golf training/inference settings such as ROI size, clip length, background mode, and heatmap target mode.

- [ ] `docs/golf_dataset_format.md`
  - Describe annotation schema, directory layout, and conversion rules.

- [ ] `docs/golf_training_workflow.md`
  - Step-by-step training, validation, and debugging instructions.

- [ ] `docs/golf_ablation_plan.md`
  - Track experiments for heatmap target shape, clip length, background source, and fusion strategy.

### Existing files to modify

- [~] `dataset.py`
  - [x] Golf dataset layout loads cleanly (_get_split and CSV paths fixed).
  - [ ] Make target heatmap generation configurable for circle vs ellipse.
  - [ ] Support ROI dimensions and golf-specific background sources.

- [ ] `preprocess.py`
  - Add entry points or flags for golf dataset preprocessing and background generation modes.

- [ ] `train.py`
  - Add golf config loading and expose key parameters such as clip length, target mode, sigma settings, ROI paths, and debug options.

- [ ] `test.py`
  - Add golf evaluation mode and optional heatmap export during validation.

- [ ] `predict.py`
  - Support short impact-centered golf clips / frame folders and save per-frame debug artifacts.

- [~] `utils/general.py`
  - [x] get_rally_dirs() updated to work with golf flat structure (no match{N} dependency).
  - [ ] Add helper functions for ROI coordinate transforms, background loading, heatmap decoding, and visualization.

- [ ] `README.md`
  - Add golf workflow documentation once the first working version is validated.

## Implementation Breakdown

### Phase 1 — Dataset and annotation pipeline
- [X] Define golf annotation schema:
  - frame index
  - visibility
  - x, y center
  - optional blur metadata
- [X] Label pilot dataset in CVAT
- [X] Export labels and convert into TrackNetV3-compatible CSV format
- [X] Verify loader can read converted data end to end

### Phase 2 — Preprocessing and input construction
- [ ] Implement ROI crop generation for all frames
- [ ] Remap labels from full-frame coordinates to ROI coordinates
- [x] Build background images for shots (per-shot median, all frames)
- [ ] Validate ROI/background pairs visually

### Phase 3 — Heatmap target adaptation
- [ ] Implement configurable circular Gaussian target
- [ ] Implement larger circular target variant
- [ ] Implement elliptical Gaussian target variant
- [ ] Add config switches for target mode and sigma values

### Phase 4 — TrackNet training
- [ ] Train TrackNet only on golf ROI clips
- [ ] Start with background concatenation enabled
- [ ] Add basic augmentations:
  - motion blur
  - brightness shifts
  - small translations
  - small scale changes
- [ ] Save checkpoints and validation visualizations

### Phase 5 — Detectability validation
- [ ] Export predicted heatmaps on held-out clips
- [ ] Review heatmap quality on:
  - sharp-ball frames
  - streaked-ball frames
  - no-ball frames
- [ ] Compare target variants and clip lengths
- [ ] Decide whether tracker is strong enough to proceed

### Phase 6 — Overlap fusion and temporal cleanup
- [ ] Implement overlapping window inference
- [ ] Add Gaussian weighting fusion
- [ ] Add confidence-weighted fusion
- [ ] Decode raw trajectory
- [ ] Apply basic smoothing / outlier rejection / short-gap interpolation

### Phase 7 — Optional rectifier phase
- [ ] Benchmark whether simple smoothing is sufficient
- [ ] If not, prepare retraining path for original InpaintNet-style rectifier
- [ ] Compare learned rectifier vs heuristic smoother

## Testing & Validation Plan

### Data integrity
- [ ] Confirm every clip has matching frames, labels, ROI crop, and background image
- [ ] Confirm coordinate transforms are correct after cropping
- [ ] Confirm no mislabeled frame indexing issues

### Tracking quality
- [ ] Verify the model activates on ball/streak frames
- [ ] Verify false positives stay low on non-ball frames
- [ ] Measure localization error on visible frames
- [ ] Compare performance across sharp vs streaked frames

### Ablations
- [ ] Compare background modes:
  - empty frame
  - pre-impact median
  - custom background
- [ ] Compare target modes:
  - circle
  - large circle
  - ellipse
- [ ] Compare clip lengths:
  - 5
  - 7
  - 8
- [ ] Compare overlap fusion:
  - Gaussian
  - confidence-weighted

### Exit criteria for Phase 1
- [ ] Held-out clips show clear, reliable heatmap peaks on most relevant ball frames
- [ ] Failure cases are diagnosable from saved debug outputs
- [ ] Raw decoded trajectory is good enough to justify smoothing / rectification work

## Risks

- [ ] Ball blur may be too elongated for stock point-target supervision
- [ ] Background images built from contaminated frames may hurt more than help
- [ ] ROI chosen too tightly may clip the ball path
- [ ] ROI chosen too loosely may waste resolution
- [ ] Dataset size may be too small to separate blur cases cleanly
- [ ] Converted dataset format may silently mismatch original repo assumptions

## Immediate Next Actions

- [x] Create pilot annotation set in CVAT
- [x] Write dataset conversion script
- [x] Verify dataset loader reads converted golf data end-to-end
- [x] Implement golf background builder
- [ ] Implement ROI crop pipeline
- [ ] Add configurable heatmap target generation
- [ ] Add heatmap export/debug utility
- [ ] Run first TrackNet-only training experiment
