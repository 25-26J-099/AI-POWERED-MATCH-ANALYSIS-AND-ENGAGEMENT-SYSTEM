from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.tactical_gnn.dataset import DatasetPreparationReport, prepare_tactical_dataset
from app.tactical_gnn.graph_builder import FreezeFrameGraph, build_graph_from_snapshot, collate_graphs
from app.tactical_gnn.model import create_model
from app.tactical_gnn.schemas import LABEL_HEADS, TacticalGNNConfig
from app.tactical_gnn.utils import config_from_settings, ensure_directory, resolve_device, set_seed

LOGGER = logging.getLogger(__name__)


def load_tactical_samples(
    dataset_path: str | Path,
    *,
    allow_pseudo_labels: bool = True,
) -> tuple[list[dict[str, Any]], DatasetPreparationReport]:
    return prepare_tactical_dataset(dataset_path, allow_pseudo_labels=allow_pseudo_labels)


def build_label_maps(samples: list[dict[str, Any]], config: TacticalGNNConfig) -> dict[str, list[str]]:
    label_maps = {head: list(config.label_maps[head]) for head in LABEL_HEADS}
    for sample in samples:
        labels = sample.get("labels") or {}
        for head in LABEL_HEADS:
            label = labels.get(head)
            if label and label not in label_maps[head]:
                label_maps[head].append(label)
    return label_maps


def infer_active_heads(samples: list[dict[str, Any]], label_maps: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    active_heads: list[str] = []
    reasons: dict[str, str] = {}
    for head in LABEL_HEADS:
        present_labels = sorted({sample.get("labels", {}).get(head) for sample in samples if sample.get("labels", {}).get(head)})
        if not present_labels:
            reasons[head] = "no labels available"
            continue
        if len(present_labels) < 2:
            reasons[head] = f"only one class present ({present_labels[0]})"
            continue
        active_heads.append(head)
        reasons[head] = f"{len(present_labels)} labels available"
        for label in present_labels:
            if label not in label_maps[head]:
                label_maps[head].append(label)
    return active_heads, reasons


class TacticalSnapshotDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        config: TacticalGNNConfig,
        label_maps: dict[str, list[str]],
        active_heads: list[str],
    ) -> None:
        self.samples = samples
        self.config = config
        self.label_maps = label_maps
        self.active_heads = set(active_heads)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
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
            config=self.config,
        )
        labels = sample.get("labels") or {}
        targets = {}
        for head in LABEL_HEADS:
            label = labels.get(head)
            if head in self.active_heads and label in self.label_maps[head]:
                targets[head] = self.label_maps[head].index(label)
            else:
                targets[head] = -100
        return {"graph": graph, "targets": targets}


def collate_training_batch(items: list[dict[str, Any]]) -> tuple[FreezeFrameGraph, dict[str, torch.Tensor]]:
    graph_batch = collate_graphs([item["graph"] for item in items])
    target_batch = {
        head: torch.tensor([item["targets"][head] for item in items], dtype=torch.long)
        for head in LABEL_HEADS
    }
    return graph_batch, target_batch


def _split_samples(
    samples: list[dict[str, Any]],
    val_split: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2 or val_split <= 0.0:
        return shuffled, []
    val_count = max(1, int(math.ceil(len(shuffled) * val_split)))
    return shuffled[val_count:], shuffled[:val_count]


def _move_graph_to_device(graph: FreezeFrameGraph, device: str) -> FreezeFrameGraph:
    graph.x = graph.x.to(device)
    graph.edge_index = graph.edge_index.to(device)
    graph.edge_attr = graph.edge_attr.to(device)
    graph.batch = graph.batch.to(device)
    graph.teammate_mask = graph.teammate_mask.to(device)
    return graph


def _compute_multi_head_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    criterion: nn.Module,
    active_heads: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    metrics: dict[str, float] = {}
    for head in active_heads:
        head_loss = criterion(outputs[head], targets[head])
        losses.append(head_loss)
        metrics[f"{head}_loss"] = float(head_loss.detach().cpu().item())
    if not losses:
        raise ValueError("no active heads available for training")
    return torch.stack(losses).mean(), metrics


def evaluate(model: nn.Module, loader: DataLoader, device: str, active_heads: list[str]) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_batches = 0
    correct = {head: 0 for head in active_heads}
    total = {head: 0 for head in active_heads}

    model.eval()
    with torch.no_grad():
        for graph, targets in loader:
            graph = _move_graph_to_device(graph, device)
            targets = {head: tensor.to(device) for head, tensor in targets.items()}
            outputs = model(graph)
            loss, _ = _compute_multi_head_loss(outputs, targets, criterion, active_heads)
            total_loss += float(loss.cpu().item())
            total_batches += 1
            for head in active_heads:
                valid_mask = targets[head] != -100
                if torch.any(valid_mask):
                    predicted = outputs[head][valid_mask].argmax(dim=-1)
                    correct[head] += int((predicted == targets[head][valid_mask]).sum().item())
                    total[head] += int(valid_mask.sum().item())
    metrics = {"loss": total_loss / max(total_batches, 1)}
    for head in active_heads:
        metrics[f"{head}_accuracy"] = float(correct[head] / total[head]) if total[head] else 0.0
    return metrics


def _save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sample_ids(samples: list[dict[str, Any]]) -> list[str]:
    return [str(sample.get("event_id")) for sample in samples if sample.get("event_id") is not None]


def train(
    data_path: str,
    output_dir: str,
    config: TacticalGNNConfig | None = None,
    *,
    allow_pseudo_labels: bool = True,
    patience: int = 5,
) -> dict[str, Any]:
    runtime_config = config or config_from_settings()
    samples, data_report = load_tactical_samples(data_path, allow_pseudo_labels=allow_pseudo_labels)
    if not samples:
        raise ValueError("no tactical samples were loaded")

    label_maps = build_label_maps(samples, runtime_config)
    runtime_config.label_maps = label_maps
    active_heads, head_reasons = infer_active_heads(samples, label_maps)
    if not active_heads:
        raise ValueError("dataset does not contain enough labels to train any prediction head")

    train_samples, val_samples = _split_samples(samples, runtime_config.val_split, runtime_config.seed)
    train_dataset = TacticalSnapshotDataset(train_samples, runtime_config, label_maps, active_heads)
    val_dataset = TacticalSnapshotDataset(val_samples, runtime_config, label_maps, active_heads)

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(runtime_config.training_batch_size, max(len(train_dataset), 1)),
        shuffle=True,
        collate_fn=collate_training_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(runtime_config.training_batch_size, max(len(val_dataset), 1)),
        shuffle=False,
        collate_fn=collate_training_batch,
    ) if val_samples else None

    example_graph, _ = collate_training_batch([train_dataset[0]])
    model = create_model(
        runtime_config,
        input_dim=example_graph.x.shape[-1],
        edge_dim=example_graph.edge_attr.shape[-1] if example_graph.edge_attr.numel() else 4,
        label_maps=label_maps,
    )
    device = resolve_device(runtime_config.device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=runtime_config.learning_rate,
        weight_decay=runtime_config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    output_path = ensure_directory(output_dir)
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, runtime_config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for graph, targets in train_loader:
            graph = _move_graph_to_device(graph, device)
            targets = {head: tensor.to(device) for head, tensor in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(graph)
            loss, batch_metrics = _compute_multi_head_loss(outputs, targets, criterion, active_heads)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            batches += 1

        metrics = {"epoch": float(epoch), "train_loss": epoch_loss / max(batches, 1)}
        metrics.update(batch_metrics)
        current_val_loss = metrics["train_loss"]
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, active_heads)
            metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
            current_val_loss = val_metrics["loss"]

        history.append(metrics)
        LOGGER.info("Epoch %s metrics: %s", epoch, metrics)

        if current_val_loss <= best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(runtime_config),
                    "label_maps": label_maps,
                    "input_dim": int(example_graph.x.shape[-1]),
                    "edge_dim": int(example_graph.edge_attr.shape[-1] if example_graph.edge_attr.numel() else 4),
                    "history": history,
                    "active_heads": active_heads,
                    "dataset_report": data_report.to_dict(),
                },
                output_path / "model.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                LOGGER.info("Early stopping triggered after epoch %s", epoch)
                break

    final_metrics = history[-1] if history else {}
    training_summary = {
        "usable_samples": len(samples),
        "dropped_samples": data_report.dropped_samples,
        "active_heads": active_heads,
        "head_status": head_reasons,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "allow_pseudo_labels": allow_pseudo_labels,
        "final_metrics": final_metrics,
    }
    split_manifest = {
        "train_event_ids": _sample_ids(train_samples),
        "val_event_ids": _sample_ids(val_samples),
        "split_seed": runtime_config.seed,
        "val_split": runtime_config.val_split,
        "allow_pseudo_labels": allow_pseudo_labels,
    }

    _save_json(output_path / "metrics.json", history)
    _save_json(output_path / "label_maps.json", label_maps)
    _save_json(output_path / "training_config.json", asdict(runtime_config))
    _save_json(output_path / "dataset_report.json", data_report.to_dict())
    _save_json(output_path / "training_summary.json", training_summary)
    _save_json(output_path / "split_manifest.json", split_manifest)

    result = {
        "checkpoint_path": str(output_path / "model.pt"),
        "label_maps_path": str(output_path / "label_maps.json"),
        "training_config_path": str(output_path / "training_config.json"),
        "metrics_path": str(output_path / "metrics.json"),
        "dataset_report_path": str(output_path / "dataset_report.json"),
        "training_summary_path": str(output_path / "training_summary.json"),
        "split_manifest_path": str(output_path / "split_manifest.json"),
        **training_summary,
    }
    LOGGER.info(
        "Training summary: usable_samples=%s dropped=%s active_heads=%s final_metrics=%s",
        result["usable_samples"],
        result["dropped_samples"],
        result["active_heads"],
        result["final_metrics"],
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the tactical GNN snapshot classifier.")
    parser.add_argument("--data", required=True, help="Path to a tactical dataset file or directory.")
    parser.add_argument("--output", required=True, help="Directory for checkpoints and metadata.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional training epoch override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto.")
    parser.add_argument("--disable-pseudo-labels", action="store_true", help="Train only on true labels when present.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_settings()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.training_batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device
    set_seed(config.seed)
    result = train(
        args.data,
        args.output,
        config=config,
        allow_pseudo_labels=not args.disable_pseudo_labels,
        patience=args.patience,
    )
    LOGGER.info("Training complete: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
