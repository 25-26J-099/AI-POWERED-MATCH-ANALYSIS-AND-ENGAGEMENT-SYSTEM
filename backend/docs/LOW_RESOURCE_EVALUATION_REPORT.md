# Low-Resource Evaluation Report

This document is a lightweight template for reporting evidence that supports proposal claims without overstating what has been validated.

## Purpose

Use this report to record real-match evidence for:

- Tracking quality
- Re-identification quality
- Event-detection quality
- Runtime and low-resource behavior

This should be updated whenever you evaluate the pipeline on new amateur-football clips.

## Evaluation Scope

- Video source:
- Number of matches:
- Number of clips:
- Camera type:
- Average resolution:
- Average FPS:
- Lighting conditions:
- Hardware used:
- GPU or CPU mode:

## Tracking Metrics

| Metric | Value | Notes |
| --- | --- | --- |
| Detection precision | TBD | |
| Detection recall | TBD | |
| MOTA | TBD | |
| MOTP | TBD | |
| IDF1 | TBD | |
| ID switches | TBD | |

## Re-Identification Metrics

| Metric | Value | Notes |
| --- | --- | --- |
| Re-entry match accuracy | TBD | |
| Identity switch recovery rate | TBD | |
| Occlusion recovery success rate | TBD | |

## Event Detection Metrics

| Event type | Precision | Recall | F1 | Notes |
| --- | --- | --- | --- | --- |
| Pass | TBD | TBD | TBD | |
| Shot | TBD | TBD | TBD | |
| Tackle | TBD | TBD | TBD | |
| Possession change | TBD | TBD | TBD | |
| Out of bounds | TBD | TBD | TBD | |

## Runtime Metrics

| Metric | Value | Notes |
| --- | --- | --- |
| Average processing FPS | TBD | |
| End-to-end runtime per match | TBD | |
| Peak memory usage | TBD | |
| Model load time | TBD | |

## Current Repository Evidence

The repository already includes tactical evaluation artifacts for the GNN-backed tactical module:

- Dataset: `data/tactical_gnn/gnn_synthetic_augmented.jsonl`
- Evaluation summary: `artifacts/tactical_gnn_eval/summary.md`
- Training summary: `checkpoints/tactical_gnn/training_summary.json`

These are valuable, but they do not replace real low-resource match evaluation for the full analysis pipeline.

## Research-Safe Conclusion Template

Use language like this in reports:

> The current implementation demonstrates an end-to-end low-resource football analysis pipeline with working preprocessing, tracking, re-identification, event-detection, and export stages. Tactical-model results are currently supported by offline evaluation on the available dataset. Broader claims about real-match robustness and low-resource deployment should be interpreted as partially validated until additional real-world benchmarking is completed.

