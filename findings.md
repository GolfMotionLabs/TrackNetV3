# Findings & Decisions

## Deliverables

- [x] Golf dataset conversion pipeline — converts CVAT image-task XML + frame folders into loader-compatible layout
- [ ] ROI crop preprocessing for frames and labels
- [x] Golf background generation module — per-shot median (default), cross-shot session, or manual image
- [ ] Heatmap target generator with configurable target shape
- [ ] Training script/config for golf TrackNet
- [ ] Validation/inference heatmap export utilities
- [ ] Initial overlap fusion implementation
- [ ] Basic temporal smoothing / gap-fill module
- [ ] Documentation for training, inference, and debugging

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-29 | Reuse TrackNetV3 as the base instead of starting from scratch | The overall heatmap + temporal pipeline is suitable for tiny fast objects, but requires golf-specific retraining and preprocessing. |
| 2026-04-29 | Retrain the tracking module first, postpone rectifier retraining | Per-frame detectability is the first real risk; trajectory repair is only useful once the tracker outputs good heatmaps. |
| 2026-04-29 | Add ROI cropping around the launch zone | Full-frame resizing wastes resolution; ROI preserves ball pixels and reduces irrelevant background. |
| 2026-04-29 | Use background concatenation as an input mode | TrackNetV3 already benefits from background input, and the golf setup has an even more stable camera/background. |
| 2026-04-29 | Do not use badminton checkpoints as-is | Domain gap is too large: different object, motion, blur, and scene statistics. |
| 2026-04-29 | Generate target heatmaps from point labels in code | This allows quick experiments with circular vs elliptical targets without relabeling data. |
| 2026-04-29 | Use CVAT for annotation | Free, video-friendly, and well suited for short tracked sequences with framewise center labels. |
| 2026-04-29 | Add heatmap visualization/debug export early | The first milestone is to verify that the ball is detectable at all, frame by frame. |

## Open Questions

- [ ] Should the first training target be circular Gaussian, larger circular Gaussian, or elliptical Gaussian for streaked frames?
- [ ] Will ROI be fixed per camera setup, or derived per clip from a known ball/tee position?
- [ ] What background source is best: empty scene frame, pre-impact median, or custom provided background?
- [ ] Should the first inference fusion method use original Gaussian overlap weighting or confidence-weighted fusion?
- [ ] Is TrackNet input length best kept at 8 frames, or should 5 and 7 frame variants be supported from the start?
- [ ] Should initial labels include blur-type metadata (`sharp`, `short_streak`, `long_streak`) for later analysis?

## Dataset Layout — Confirmed Structure

Actual input data (discovered 2026-04-30):
- Frames: `impact_frames/{shot_name}/frame_XXXXXX.png` (~20 frames per shot, non-contiguous numbering)
- Annotations: single `xml/annotations.xml` — CVAT **image task** format (not video task)
  - Each `<image>` element: `name="{shot_name}/{shot_name}_impact_frame_{NNNN}.png"`
  - Ball annotation as `<points label="ball">` child with `points="x,y"`
  - Extra attributes: `visibility` (sharp/blurred/streak/uncertain), `usable_for_training` (yes/no)
- 28 shot folders total; 3 annotated as of 2026-04-30; rest present as empty `<image>` elements

Chosen output layout:
```
data/{split}/{shot_name}/frame/{shot_name}/0.png, 1.png, ...
data/{split}/{shot_name}/frame/{shot_name}/median.npz
data/{split}/{shot_name}/csv/{shot_name}.csv
```

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-29 | Reuse TrackNetV3 as the base instead of starting from scratch | The overall heatmap + temporal pipeline is suitable for tiny fast objects, but requires golf-specific retraining and preprocessing. |
| 2026-04-29 | Retrain the tracking module first, postpone rectifier retraining | Per-frame detectability is the first real risk; trajectory repair is only useful once the tracker outputs good heatmaps. |
| 2026-04-29 | Add ROI cropping around the launch zone | Full-frame resizing wastes resolution; ROI preserves ball pixels and reduces irrelevant background. |
| 2026-04-29 | Use background concatenation as an input mode | TrackNetV3 already benefits from background input, and the golf setup has an even more stable camera/background. |
| 2026-04-29 | Do not use badminton checkpoints as-is | Domain gap is too large: different object, motion, blur, and scene statistics. |
| 2026-04-29 | Generate target heatmaps from point labels in code | This allows quick experiments with circular vs elliptical targets without relabeling data. |
| 2026-04-29 | Use CVAT for annotation | Free, video-friendly, and well suited for short tracked sequences with framewise center labels. |
| 2026-04-29 | Add heatmap visualization/debug export early | The first milestone is to verify that the ball is detectable at all, frame by frame. |
| 2026-04-30 | Remove match/ball naming from dataset layout | Golf has no concept of matches or rallies; naming should reflect shots only. |
| 2026-04-30 | Background: per-shot median of all frames in the shot | Chosen over cross-shot session median for now. ~20 frames per shot with fast-moving ball gives reasonable suppression. Session mode (cross-shot) and manual mode available for future calibration phase. |
| 2026-04-30 | Include unannotated shots as all-Visibility=0 sequences | Valid TN training examples; model needs to learn background-only sequences too. |

## Open Questions

- [ ] Should the first training target be circular Gaussian, larger circular Gaussian, or elliptical Gaussian for streaked frames?
- [ ] Will ROI be fixed per camera setup, or derived per clip from a known ball/tee position?
- [x] What background source is best? → Per-shot median for now; calibration-phase empty scene deferred.
- [ ] Should the first inference fusion method use original Gaussian overlap weighting or confidence-weighted fusion?
- [ ] Is TrackNet input length best kept at 8 frames, or should 5 and 7 frame variants be supported from the start?
- [x] Should initial labels include blur-type metadata? → Yes, stored as blur_type column in CSV.

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| CVAT export was image task (not video task) — no `<track>` elements | Parse `<image>` elements grouped by clip name from the image path |
| Original frame numbering non-contiguous (e.g. 285, 286, 289…) | Sort by extracted frame number, re-index to 0-based sequential in output |
| Windows console UnicodeEncodeError on → character in print | Replaced with ASCII -> |
| `get_rally_dirs` crashed on golf paths due to `int(s.split('match')[-1])` | Rewrote to walk frame subdirs generically without assuming match{N} format |