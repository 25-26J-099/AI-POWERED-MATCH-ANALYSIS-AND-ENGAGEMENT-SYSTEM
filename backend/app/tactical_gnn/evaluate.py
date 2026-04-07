from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from app.tactical_gnn.dataset import prepare_tactical_dataset
from app.tactical_gnn.graph_builder import build_graph_from_snapshot
from app.tactical_gnn.model import create_model
from app.tactical_gnn.schemas import LABEL_HEADS, TacticalGNNConfig
from app.tactical_gnn.utils import config_from_settings, ensure_directory, resolve_device

LOGGER = logging.getLogger(__name__)


def _load_checkpoint(checkpoint_path: str, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TacticalGNNConfig(**checkpoint.get("config", {}))
    label_maps = checkpoint.get("label_maps", config.label_maps)
    model = create_model(
        config=config,
        input_dim=int(checkpoint["input_dim"]),
        edge_dim=int(checkpoint["edge_dim"]),
        label_maps=label_maps,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, {
        "config": config,
        "label_maps": label_maps,
        "active_heads": checkpoint.get("active_heads", LABEL_HEADS),
    }


def _load_split_event_ids(checkpoint_path: str, split: str | None) -> set[str] | None:
    if not split:
        return None
    checkpoint_dir = Path(checkpoint_path).resolve().parent
    manifest_path = checkpoint_dir / "split_manifest.json"
    if not manifest_path.exists():
        LOGGER.warning("Requested split '%s' but no split manifest was found at %s", split, manifest_path)
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = f"{split}_event_ids"
    event_ids = manifest.get(key)
    if not isinstance(event_ids, list):
        LOGGER.warning("Requested split '%s' but %s is missing from %s", split, key, manifest_path)
        return None
    return {str(event_id) for event_id in event_ids}


def evaluate_checkpoint(
    data_path: str,
    checkpoint_path: str,
    output_dir: str,
    *,
    allow_pseudo_labels: bool = True,
    device: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    samples, dataset_report = prepare_tactical_dataset(data_path, allow_pseudo_labels=allow_pseudo_labels)
    output_path = ensure_directory(output_dir)
    runtime_device = resolve_device(device)
    model, bundle = _load_checkpoint(checkpoint_path, runtime_device)
    label_maps = bundle["label_maps"]
    active_heads: list[str] = bundle["active_heads"]
    split_event_ids = _load_split_event_ids(checkpoint_path, split)
    if split_event_ids is not None:
        samples = [sample for sample in samples if str(sample.get("event_id")) in split_event_ids]

    metrics: dict[str, Any] = {
        "dataset_report": dataset_report.to_dict(),
        "checkpoint_path": checkpoint_path,
        "device": runtime_device,
        "evaluation_split": split or "all",
        "evaluated_samples": len(samples),
        "heads": {},
    }
    prediction_rows: list[dict[str, Any]] = []
    formation_true: list[str] = []
    formation_pred: list[str] = []

    per_head_true: dict[str, list[str]] = {head: [] for head in active_heads}
    per_head_pred: dict[str, list[str]] = {head: [] for head in active_heads}
    per_head_conf: dict[str, list[float]] = {head: [] for head in active_heads}
    per_head_sources: dict[str, Counter[str]] = {head: Counter() for head in active_heads}

    for sample in samples:
        graph = build_graph_from_snapshot(
            {
                "location": sample.get("event_location"),
                "type_name": sample.get("event_type"),
                "attacking_right": sample.get("attacking_right", True),
            },
            {
                "freeze_frame": sample.get("freeze_frame", []),
                "attacking_right": sample.get("attacking_right", True),
            },
            config=bundle["config"],
        )
        graph.x = graph.x.to(runtime_device)
        graph.edge_index = graph.edge_index.to(runtime_device)
        graph.edge_attr = graph.edge_attr.to(runtime_device)
        graph.batch = graph.batch.to(runtime_device)
        graph.teammate_mask = graph.teammate_mask.to(runtime_device)
        with torch.no_grad():
            outputs = model(graph)

        row = {
            "event_id": sample["event_id"],
            "event_type": sample.get("event_type"),
            "label_sources": sample.get("label_sources", {}),
        }
        for head in active_heads:
            probabilities = torch.softmax(outputs[head][0].cpu(), dim=-1)
            confidence, index = torch.max(probabilities, dim=-1)
            pred_label = label_maps[head][int(index.item())]
            true_label = sample.get("labels", {}).get(head)
            row[f"{head}_pred"] = pred_label
            row[f"{head}_confidence"] = float(confidence.item())
            row[f"{head}_true"] = true_label
            if true_label:
                per_head_true[head].append(true_label)
                per_head_pred[head].append(pred_label)
                per_head_conf[head].append(float(confidence.item()))
                per_head_sources[head][sample.get("label_sources", {}).get(head, "missing")] += 1
                if head == "formation":
                    formation_true.append(true_label)
                    formation_pred.append(pred_label)
        prediction_rows.append(row)

    for head in active_heads:
        truths = per_head_true[head]
        preds = per_head_pred[head]
        if truths:
            metrics["heads"][head] = {
                "support": len(truths),
                "accuracy": float(accuracy_score(truths, preds)),
                "macro_f1": float(f1_score(truths, preds, average="macro", zero_division=0)),
                "micro_f1": float(f1_score(truths, preds, average="micro", zero_division=0)),
                "label_support": dict(Counter(truths)),
                "prediction_distribution": dict(Counter(preds)),
                "label_source_support": dict(per_head_sources[head]),
                "mean_confidence": float(sum(per_head_conf[head]) / len(per_head_conf[head])),
            }
        else:
            metrics["heads"][head] = {"support": 0, "status": "no labels available for evaluation"}

    if formation_true:
        ordered_labels = label_maps["formation"]
        matrix = confusion_matrix(formation_true, formation_pred, labels=ordered_labels)
        with (output_path / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["true/pred", *ordered_labels])
            for label, row in zip(ordered_labels, matrix.tolist()):
                writer.writerow([label, *row])

    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_path / "sample_predictions.jsonl").open("w", encoding="utf-8") as outfile:
        for row in prediction_rows:
            outfile.write(json.dumps(row) + "\n")

    summary_lines = [
        "# Tactical GNN Evaluation",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Dataset path: `{data_path}`",
        f"- Usable samples: `{dataset_report.usable_samples}`",
        f"- Evaluated samples: `{len(samples)}`",
        f"- Dropped samples: `{dataset_report.dropped_samples}`",
        f"- Evaluation split: `{split or 'all'}`",
        f"- Active heads: `{', '.join(active_heads)}`",
        "",
    ]
    for head, head_metrics in metrics["heads"].items():
        summary_lines.append(f"## {head}")
        if head_metrics.get("support", 0) == 0:
            summary_lines.append("- No labels available")
        else:
            summary_lines.append(f"- Support: `{head_metrics['support']}`")
            summary_lines.append(f"- Accuracy: `{head_metrics['accuracy']:.4f}`")
            summary_lines.append(f"- Macro F1: `{head_metrics['macro_f1']:.4f}`")
            summary_lines.append(f"- Micro F1: `{head_metrics['micro_f1']:.4f}`")
        summary_lines.append("")
    (output_path / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    LOGGER.info("Evaluation complete: %s", output_path)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained tactical GNN checkpoint.")
    parser.add_argument("--data", required=True, help="Dataset file or directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--output", required=True, help="Output directory for evaluation artifacts.")
    parser.add_argument("--disable-pseudo-labels", action="store_true", help="Evaluate only on true labels when present.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto.")
    parser.add_argument(
        "--split",
        choices=("train", "val", "all"),
        default="val",
        help="Subset to evaluate when a split manifest is available. Defaults to val.",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    evaluate_checkpoint(
        args.data,
        args.checkpoint,
        args.output,
        allow_pseudo_labels=not args.disable_pseudo_labels,
        device=args.device or config_from_settings().device,
        split=None if args.split == "all" else args.split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
