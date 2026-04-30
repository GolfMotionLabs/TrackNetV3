# Progress Log

## Session History

### Session 1 — 2026-04-28
- Initialized planning files (task_plan.md, findings.md, progress.md)
- Wrote full task_plan.md (all phases, deliverables, file plan, risks)

### Session 2 — 2026-04-28
- Created tools/convert_golf_dataset.py (initial version, mp4-based — superseded)

### Session 3 — 2026-04-30
- Rewrote convert_golf_dataset.py for actual data layout:
  - Input: impact_frames/{shot_name}/frame_XXXXXX.png + single CVAT image-task XML
  - Parses CVAT "for images" XML (not video task); groups by clip name from image path
  - Preserves blur_type attribute (sharp/blurred/streak/uncertain) as extra CSV column
  - Frames re-indexed 0-based after sorting by original frame number
  - Unannotated shots included with Visibility=0 by default (--skip_unannotated to omit)
- Redesigned output layout to be golf-specific (no "match", no "ball"):
  - data/{split}/{shot_name}/frame/{shot_name}/0.png, 1.png, ...
  - data/{split}/{shot_name}/csv/{shot_name}.csv
- Modified utils/general.py — get_rally_dirs(): removed hardcoded match{N} sort; now walks
  {split}/{entry}/frame/{sub}/ generically, works for both golf and legacy badminton layouts
- Modified dataset.py — _get_split(): replaced match{}-based parse with relpath split
- Modified dataset.py — _gen_input_from_rally_dir(): removed _ball suffix from all CSV paths
- Smoke-tested dataset loader end-to-end: Shuttlecock_Trajectory_Dataset instantiates and
  returns correctly shaped tensors from converted golf data

### Session 6 — 2026-04-30
- Set WIDTH=416, HEIGHT=768 in utils/general.py
  - ROI aspect ratio 0.5424 vs 416/768=0.5417, only 0.1% distortion
  - Both divisible by 32 (required by U-Net downsampling)
  - 320K pixels vs original 148K — preserves more ball detail after resize

### Session 5 — 2026-04-30
- Created tools/build_roi_clips.py
  - Fixed ROI: x=[196,996), y=[24,1499), 800x1475px portrait
  - Crops frames, backgrounds (median.npz), and CSVs; labels outside ROI → Visibility=0
  - Verified: IMG_1660 frame 0 (ball at y=1552, below ROI bottom y=1499) correctly silenced
  - Output to data_roi/ (data/ originals preserved)
- Identified WIDTH=288, HEIGHT=512 as the portrait resolution to set before training
  (aspect ratio 0.5625 vs ROI 0.542, <4% distortion, both divisible by 32)
- Ran pipeline: 26 shots cropped, 0 skipped

### Session 4 — 2026-04-30
- Created tools/build_golf_backgrounds.py
  - Three modes: shot (default), session (cross-shot), manual (user image)
  - Default shot mode: per-pixel median of all frames within each shot
  - Outputs data/{split}/{shot_name}/median.npz at the primary lookup path,
    overriding the placeholder written by the conversion script
  - --save_preview writes a .png alongside each npz for visual inspection
  - Session mode available for later calibration-phase replacement
- Ran shot-mode background build on all 12 converted shots; previews saved

---

## Change Log

| Session | Change | File(s) |
|---------|--------|---------|
| 1 | Created planning files | task_plan.md, findings.md, progress.md |
| 2 | Initial conversion script (mp4 input, superseded) | tools/convert_golf_dataset.py |
| 3 | Rewrote conversion for frame-folder + CVAT image XML input | tools/convert_golf_dataset.py |
| 3 | Golf-specific output layout (no match/ball naming) | tools/convert_golf_dataset.py |
| 3 | Generic rally dir discovery, drop match{N} sort | utils/general.py |
| 3 | Golf-compatible split parsing and CSV path | dataset.py |
| 4 | Background builder with shot/session/manual modes | tools/build_golf_backgrounds.py |
| 5 | ROI crop pipeline (frames, labels, backgrounds) | tools/build_roi_clips.py |
| 6 | Set network resolution WIDTH=416, HEIGHT=768 | utils/general.py |

## Blockers
_None._
