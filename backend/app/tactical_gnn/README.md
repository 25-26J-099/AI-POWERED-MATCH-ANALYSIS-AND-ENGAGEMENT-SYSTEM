# Tactical GNN Module

This package adds graph-based tactical snapshot analysis to the backend while preserving the existing heuristic commentary flow as a production fallback.

## Files

- `dataset.py`: real-data discovery, schema adaptation, and label normalization
- `schemas.py`: dataset validation, label vocabularies, and runtime config objects
- `utils.py`: shared event and freeze-frame normalization plus config helpers
- `features.py`: node and edge feature engineering
- `graph_builder.py`: deterministic freeze-frame graph construction and batching
- `model.py`: tactical GNN model with optional PyTorch Geometric support and pure-PyTorch fallback
- `training.py`: train and validation pipeline with early stopping and metadata export
- `evaluate.py`: offline evaluation CLI with per-head metrics and confusion matrix export
- `comparison.py`: side-by-side GNN versus heuristic comparison helper
- `examples.py`: qualitative example generation for research reporting
- `example_dataset.json`: minimal labeled sample schema example
- `example_repo_events.json`: example using the repo's actual `events.json` export style
- `backend/scripts/train_tactical_gnn.py`: repo-native training wrapper for the dataset stored under `backend/data/tactical_gnn/`

## Supported Dataset Inputs

The loader detects and adapts these real repo-facing formats:

1. `events.json` or `*_events.json`
   Repo event export with embedded `freeze_frame`
2. `freeze_frames.json`
   Freeze-frame sidecar export from analysis jobs
3. `statsbomb_events.json`
   StatsBomb-style event export when freeze-frame data is present
4. generic labeled `.json`, `.jsonl`, or `.csv`
   Direct tactical-learning datasets

The loader inspects the actual file structure instead of assuming a single example schema.

## Included Dataset

The provided dataset has been placed at:

- `backend/data/tactical_gnn/gnn_synthetic_augmented.jsonl`

Its records contain:

- `event_id`
- `event_type`
- `event_location`
- optional `event_end_location`
- `attacking_right`
- `pitch_size`
- `freeze_frame`
- `labels`
- `metadata`
- `label_source`

## Required and Optional Fields

Required for a usable training sample:

- event id
- freeze-frame player locations

Strongly preferred:

- event location
- teammate or team id information
- actor team or player id

Optional but used when available:

- keeper flag
- actor flag
- explicit `attacking_right`
- true labels for any prediction head
- lineup formation files for formation supervision

## Label Handling

The adapter keeps label provenance explicit.

- `ground_truth`: label came directly from the dataset or lineup file
- `pseudo_heuristic`: label was weakly derived from the existing heuristic tactical logic
- `missing`: no usable label was available

Pseudo-labels are never silently treated as ground truth. Training and evaluation metadata record the source counts per head.

For `gnn_synthetic_augmented.jsonl`, the adapter also normalizes raw dataset labels into the live commentary-facing vocabulary used by inference.

Examples:

- `compact` -> `Compact Shape`
- `vertical_support` -> `Vertical Support Structure`
- `mid_block` -> `Mid Block`
- `unknown` -> `Unclear` for formation

Some source values map many-to-one because the backend vocabulary is intentionally narrower than the synthetic dataset. The original raw labels are preserved in sample metadata.

## Graph Construction

Each visible player becomes one node. By default the graph uses deterministic k-nearest-neighbor edges, but `radius` graphs are also supported through config.

Node features include:

- normalized x and y
- teammate, keeper, and actor flags
- distance and angle to event location
- team-centroid relative coordinates
- nearest-neighbor distance
- local density

Edge features include:

- Euclidean distance
- delta x
- delta y
- same-team flag

Before graph construction, coordinates are normalized so the possession side attacks left-to-right. The commentary module and the GNN module share the same normalization logic.

## Training

Run from the `backend` directory:

```bash
python -m app.tactical_gnn.training --data data/tactical_gnn/gnn_synthetic_augmented.jsonl --output checkpoints/tactical_gnn
```

Or use the repo-native wrapper that defaults to the placed dataset:

```bash
python scripts/train_tactical_gnn.py
```

Outputs:

- `model.pt`
- `label_maps.json`
- `training_config.json`
- `metrics.json`
- `dataset_report.json`
- `training_summary.json`

If a head has no usable supervision or only one class, training skips that head safely and records why in `training_summary.json`.

## Evaluation

```bash
python -m app.tactical_gnn.evaluate --data data/tactical_gnn/gnn_synthetic_augmented.jsonl --checkpoint checkpoints/tactical_gnn/model.pt --output artifacts/tactical_gnn_eval
```

Outputs:

- `metrics.json`
- `confusion_matrix.csv` for formation when labels exist
- `sample_predictions.jsonl`
- `summary.md`

## Qualitative Examples

```bash
python -m app.tactical_gnn.examples --data data/tactical_gnn/gnn_synthetic_augmented.jsonl --checkpoint checkpoints/tactical_gnn/model.pt --output artifacts/tactical_gnn_examples
```

Outputs:

- `examples.json`
- `examples.csv`
- `examples.md`

Each example includes:

- event id and event type
- compact freeze-frame summary
- GNN prediction with confidences
- heuristic prediction
- final commentary-facing tactical labels
- tactical description
- fallback status
- disagreement heads

## Inference and Comparison

The live commentary flow still uses:

`commentary_service -> merged.run_pipeline -> tac_commentary.process_event -> get_tactical_analysis -> predict_tactical_snapshot`

Offline comparison is available through `compare_tactical_predictions(...)` without changing the production path.

## Research-Honest Claim

After these additions, the repository can honestly claim:

- a production-integrated tactical GNN inference path exists
- heuristic fallback is preserved
- real export schemas and the provided synthetic tactical dataset are supported for training and evaluation
- label provenance is tracked explicitly
- quantitative and qualitative offline evaluation tooling is implemented

It cannot honestly claim robust real-match tactical performance unless the reported metrics are produced from a real labeled dataset rather than synthetic supervision.
