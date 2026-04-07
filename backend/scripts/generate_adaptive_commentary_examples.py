from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.commentary.adaptive import build_level_comparison, validate_level_progression


DEFAULT_SAMPLE = {
    "team_name": "Blue FC",
    "sequence_summary": "Event type: Pass",
    "opposition_effect": "The opposition shape protects the central lane and slows central progression.",
    "support_context": "There is immediate support around the ball and a wider outlet is available.",
    "tactical_labels": {
        "formation": "4-3-3",
        "formation_approx": "4-3-3",
        "team_shape": "Balanced Shape",
        "attacking_structure": "Balanced Structure",
        "defensive_block": "Mid Block",
        "defensive_shape": "Compact Balanced Mid Block",
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Beginner / Intermediate / Expert tactical commentary examples.")
    parser.add_argument(
        "--input",
        default=None,
        help="Optional JSON file with team_name, tactical_labels, opposition_effect, support_context, and sequence_summary.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/adaptive_commentary_examples",
        help="Directory where the comparison files should be written.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    sample = DEFAULT_SAMPLE
    if args.input:
        sample = json.loads(Path(args.input).read_text(encoding="utf-8"))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_level_comparison(
        team_name=sample.get("team_name"),
        tactical_labels=sample.get("tactical_labels"),
        opposition_effect=sample.get("opposition_effect"),
        support_context=sample.get("support_context"),
        sequence_summary=sample.get("sequence_summary"),
        base_profile=sample.get("audience_profile"),
    )
    validation = validate_level_progression(comparison)

    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    lines = [
        "# Adaptive Commentary Examples",
        "",
        f"- Team: `{sample.get('team_name', 'The possession side')}`",
        f"- Validation passed: `{validation['all_passed']}`",
        "",
    ]
    for level in ("Beginner", "Intermediate", "Expert"):
        block = comparison[level]
        lines.append(f"## {level}")
        lines.append(f"- Text: {block['text']}")
        lines.append(f"- Metrics: `{json.dumps(block['metrics'])}`")
        lines.append("")
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "validation": validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
