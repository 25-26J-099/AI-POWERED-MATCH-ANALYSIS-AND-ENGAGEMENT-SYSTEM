from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from app.tactical_gnn.comparison import compare_tactical_predictions
from app.tactical_gnn.dataset import prepare_tactical_dataset
from app.tactical_gnn.utils import config_from_settings, ensure_directory

LOGGER = logging.getLogger(__name__)


def _freeze_frame_summary(sample: dict[str, Any]) -> dict[str, Any]:
    players = sample.get("freeze_frame", [])
    return {
        "players_visible": len(players),
        "teammates_visible": sum(1 for player in players if player.get("teammate")),
        "opponents_visible": sum(1 for player in players if not player.get("teammate")),
        "keepers_visible": sum(1 for player in players if player.get("keeper")),
        "actors_visible": sum(1 for player in players if player.get("actor")),
    }


def _example_category(sample: dict[str, Any], comparison: dict[str, Any]) -> str:
    if comparison["used_fallback"]:
        return "fallback"
    if comparison["final_tactical_labels"].get("graph_metadata", {}).get("num_nodes", 0) <= 7:
        return "sparse"
    if comparison["disagreement_heads"]:
        return "disagreement"
    if comparison["gnn"].get("formation_confidence", 0.0) >= 0.6:
        return "high_confidence"
    return "representative"


def generate_examples(
    data_path: str,
    output_dir: str,
    *,
    checkpoint_path: str | None = None,
    allow_pseudo_labels: bool = True,
) -> dict[str, Any]:
    samples, dataset_report = prepare_tactical_dataset(data_path, allow_pseudo_labels=allow_pseudo_labels)
    output_path = ensure_directory(output_dir)
    config = config_from_settings()
    if checkpoint_path:
        config.model_path = checkpoint_path

    records: list[dict[str, Any]] = []
    for sample in samples:
        comparison = compare_tactical_predictions(
            event_id=sample["event_id"],
            event_data={
                "location": sample.get("event_location"),
                "type_name": sample.get("event_type"),
                "attacking_right": sample.get("attacking_right", True),
            },
            freeze_frame_data={
                "freeze_frame": sample.get("freeze_frame", []),
                "attacking_right": sample.get("attacking_right", True),
            },
            team_name=sample.get("metadata", {}).get("team_name"),
            sequence_summary=f"Event type: {sample.get('event_type') or 'Unknown'}",
            model_path=checkpoint_path,
            config=config,
        )
        record = {
            "event_id": sample["event_id"],
            "event_type": sample.get("event_type"),
            "freeze_frame_summary": _freeze_frame_summary(sample),
            "label_sources": sample.get("label_sources", {}),
            "gnn_prediction": comparison["gnn"],
            "heuristic_prediction": comparison["heuristic"],
            "final_tactical_labels": comparison["final_tactical_labels"],
            "tactical_description": comparison["tactical_description"],
            "used_fallback": comparison["used_fallback"],
            "disagreement_heads": comparison["disagreement_heads"],
            "category": _example_category(sample, comparison),
        }
        records.append(record)

    prioritized: list[dict[str, Any]] = []
    for category in ("high_confidence", "disagreement", "sparse", "fallback", "representative"):
        prioritized.extend([record for record in records if record["category"] == category])
    selected = prioritized[:10] if len(prioritized) >= 10 else prioritized

    (output_path / "examples.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    with (output_path / "examples.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "event_id",
                "event_type",
                "category",
                "used_fallback",
                "gnn_formation",
                "heuristic_formation",
                "final_formation",
                "disagreement_heads",
            ],
        )
        writer.writeheader()
        for record in selected:
            writer.writerow(
                {
                    "event_id": record["event_id"],
                    "event_type": record["event_type"],
                    "category": record["category"],
                    "used_fallback": record["used_fallback"],
                    "gnn_formation": record["gnn_prediction"].get("formation_approx", record["gnn_prediction"].get("formation")),
                    "heuristic_formation": record["heuristic_prediction"].get("formation_approx", record["heuristic_prediction"].get("formation")),
                    "final_formation": record["final_tactical_labels"].get("formation_approx"),
                    "disagreement_heads": ",".join(record["disagreement_heads"]),
                }
            )

    markdown_lines = [
        "# Tactical GNN Qualitative Examples",
        "",
        f"- Dataset path: `{data_path}`",
        f"- Usable samples: `{dataset_report.usable_samples}`",
        f"- Selected examples: `{len(selected)}`",
        "",
    ]
    for index, record in enumerate(selected, start=1):
        markdown_lines.extend(
            [
                f"## Example {index}: {record['event_id']}",
                f"- Event type: `{record['event_type']}`",
                f"- Category: `{record['category']}`",
                f"- Freeze-frame summary: `{record['freeze_frame_summary']}`",
                f"- Used fallback: `{record['used_fallback']}`",
                f"- GNN: `{record['gnn_prediction']}`",
                f"- Heuristic: `{record['heuristic_prediction']}`",
                f"- Final labels: `{record['final_tactical_labels']}`",
                f"- Tactical description: {record['tactical_description']}",
                "",
            ]
        )
    (output_path / "examples.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    LOGGER.info("Generated qualitative examples at %s", output_path)
    return {"selected_examples": len(selected), "output_dir": str(output_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate tactical GNN qualitative comparison examples.")
    parser.add_argument("--data", required=True, help="Dataset file or directory.")
    parser.add_argument("--output", required=True, help="Output directory for example artifacts.")
    parser.add_argument("--checkpoint", default=None, help="Optional model checkpoint for GNN predictions.")
    parser.add_argument("--disable-pseudo-labels", action="store_true", help="Use only true labels when building the dataset.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    generate_examples(
        args.data,
        args.output,
        checkpoint_path=args.checkpoint,
        allow_pseudo_labels=not args.disable_pseudo_labels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
