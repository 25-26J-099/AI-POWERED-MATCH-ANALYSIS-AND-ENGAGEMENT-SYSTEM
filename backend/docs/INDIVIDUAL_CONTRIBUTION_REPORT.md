# Individual Contribution Report

## Contribution Summary

The most defensible statement of individual ownership for this repository is:

> Video preprocessing, player and ball tracking, re-identification, hybrid event detection, and structured football-data export that power the analytics and commentary modules.

This contribution is the upstream analysis backbone of the project. It is distinct from the downstream commentary-generation components, even though the final repository contains integrated orchestration across all team contributions.

## Boundaries of Ownership

### Primary ownership

- Low-quality video preprocessing and enhancement
- Tracking pipeline for players and ball
- Identity continuity across occlusion and re-entry
- Rule-based plus ML-assisted event detection
- Structured export for analytics and downstream commentary
- Pipeline orchestration from raw match video to machine-readable outputs

### Adjacent but not primary ownership

- Play-by-play commentary generation
- Dual-commentator interaction design
- Audience-adaptive expert commentary
- TTS voice styling and commentary presentation

Those downstream areas exist in the merged product, but they should not be presented as the core of this individual contribution.

## Proposal-to-Implementation Matrix

| Proposal objective | Implementation evidence | Status |
| --- | --- | --- |
| Video preprocessing for low-quality single-camera footage | `app/event_detection/video_preprocessor.py` implements stabilization, enhancement, and super-resolution fallback paths | Implemented |
| Lightweight player and ball tracking using YOLO/ByteTrack-style flow | `app/event_detection/tracker.py` and `app/event_detection/pipeline.py` wire detection and tracking into the main pipeline | Implemented |
| Player re-identification across exits and re-entries | `app/event_detection/player_reid.py` and `app/event_detection/robust_reid.py` provide gallery-based and temporal identity continuity logic | Implemented |
| Hybrid event detection engine | `app/event_detection/strategic_hybrid_detector.py` routes events across rule, ML, and hybrid flows | Implemented |
| Structured event export for later processing | `app/event_detection/statsbomb_export.py` plus pipeline export paths produce machine-readable outputs | Implemented |
| Low-resource deployment orientation | Lightweight design choices and fallback logic exist, but full benchmark proof is limited | Partially validated |
| Novel annotated amateur-football dataset | Repo includes synthetic tactical data and evaluation artifacts, but not a clearly packaged released amateur dataset | Under-supported |
| Industry-standard evaluation and validation | Tactical evaluation artifacts exist, but broad real-match evaluation for tracking, Re-ID, and events is incomplete | Partially validated |

## Code Evidence

- Pipeline orchestration: `app/event_detection/pipeline.py`
- Preprocessing: `app/event_detection/video_preprocessor.py`
- Re-identification: `app/event_detection/player_reid.py`
- Hybrid event detection: `app/event_detection/strategic_hybrid_detector.py`
- API exposure of the analysis path: `app/routes/video.py`
- Analytics handoff: `app/services/merged_pipeline_service.py`

## Evidence-Based Strengths

- The repository contains a coherent path from uploaded video to event export and analytics handoff.
- The implementation reflects the proposal architecture well at a subsystem level.
- The tactical module documents its own limitations clearly rather than overstating performance.
- The backend exposes testable APIs and stores artifacts in a structured manner.

## Current Gaps

### Evidence gap

The strongest weakness is not lack of code. It is lack of broad validation evidence. Proposal-level claims about low-resource performance, dataset contribution, and real-match robustness need stronger measured support.

### Dataset gap

The repo currently demonstrates tactical learning mainly through synthetic or derived supervision artifacts. That is useful for prototyping, but it is not equivalent to a released, labeled amateur-football dataset.

### Reporting gap

There is not yet a compact report that ties proposal objectives to implemented outputs, file locations, and validation status. This makes the contribution look less mature than it actually is.

### Reproducibility gap

Environment setup was fragile because `pydantic-settings` was missing from `requirements.txt`. That issue is now documented and patched in the dependency list, but the environment should still be validated on a clean machine.

## Research-Safe Interpretation of Tactical Results

Current tactical evaluation artifacts show that a production-integrated tactical GNN pipeline exists, but the attached metrics are derived from the synthetic tactical dataset under `data/tactical_gnn/gnn_synthetic_augmented.jsonl`.

Reported evaluation snapshot from `artifacts/tactical_gnn_eval/summary.md`:

- Formation accuracy: `0.3757`
- Team shape accuracy: `0.6889`
- Attacking structure accuracy: `0.7450`
- Defensive block accuracy: `0.8518`
- Defensive shape accuracy: `0.6377`

These values are still useful as engineering evidence, but they should be described as offline tactical-model results on the currently available dataset, not as proof of robust real-match tactical performance.

## Recommended Viva Wording

Use a sentence close to this:

> My individual contribution was the match-analysis backbone of the system: preprocessing low-quality single-camera football video, tracking players and the ball, maintaining player identity through re-identification, detecting events through a hybrid rule-based and ML-assisted pipeline, and exporting structured match data for analytics and commentary generation.

## What To Improve Next

- Add a short real-match evaluation section with tracking, Re-ID, and event-detection metrics
- Record hardware, runtime, and FPS measurements for low-resource deployment claims
- Package any real annotated clips or at least document their annotation workflow and sample counts
- Keep tactical claims narrow unless validated on real labeled data

