# Progress Log

## Session History

### Session 1 — 2026-04-28
- Initialized planning files (task_plan.md, findings.md, progress.md)
- Wrote full task_plan.md (all phases, deliverables, file plan, risks)

### Session 2 — 2026-04-28
- Created tools/convert_golf_dataset.py
  - Supports CVAT for video 1.1 XML export (primary) and simple CSV fallback
  - Extracts frames, generates median.npz, writes Frame/Visibility/X/Y CSVs
  - Matches TrackNetV3 directory layout: {split}/match{N}/video|frame|csv/
  - --dry_run flag for safe preview before writing

---

## Change Log
<!-- Summarize file edits and decisions made each session -->

| Session | Change | File(s) |
|---------|--------|---------|
| 1 | Created planning files | task_plan.md, findings.md, progress.md |

## Blockers
<!-- Anything stopping forward progress -->
_None yet._
