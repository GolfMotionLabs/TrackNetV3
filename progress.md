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

## Blockers
_None._
