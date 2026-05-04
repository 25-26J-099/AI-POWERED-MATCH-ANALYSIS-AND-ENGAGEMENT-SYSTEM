from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.tactical_gnn.training import train
from app.tactical_gnn.utils import config_from_settings, set_seed


LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the repo-integrated tactical GNN on the normalized tactical dataset.",
    )
    parser.add_argument(
        "--data",
        default=str(BACKEND_ROOT / "data" / "tactical_gnn" / "gnn_synthetic_augmented.jsonl"),
        help="Path to the tactical dataset file or directory.",
    )
    parser.add_argument(
        "--output",
        default=str(BACKEND_ROOT / "checkpoints" / "tactical_gnn"),
        help="Directory for the trained checkpoint and metadata.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Optional training epoch override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto.")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed override.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument(
        "--disable-pseudo-labels",
        action="store_true",
        help="Restrict training to labels present in the dataset or lineup files.",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()

    config = config_from_settings()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.training_batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed

    set_seed(config.seed)
    result = train(
        args.data,
        args.output,
        config=config,
        allow_pseudo_labels=not args.disable_pseudo_labels,
        patience=args.patience,
    )

    summary = {
        "checkpoint_path": result["checkpoint_path"],
        "usable_samples": result["usable_samples"],
        "dropped_samples": result["dropped_samples"],
        "active_heads": result["active_heads"],
        "final_metrics": result["final_metrics"],
    }
    LOGGER.info("Training summary: %s", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
