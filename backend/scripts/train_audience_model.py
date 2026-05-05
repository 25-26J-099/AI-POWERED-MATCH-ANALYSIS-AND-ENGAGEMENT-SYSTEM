from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.commentary.adaptive import (  # noqa: E402
    AUDIENCE_MODEL_FEATURES,
    COMMENTARY_STYLES,
    COMMENTARY_VERBOSITY,
    build_audience_feature_vector,
    infer_audience_level_rules,
)


KNOWLEDGE_OPTIONS = ("unknown", "low", "moderate", "high")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the lightweight learned audience-modeling classifier used by adaptive commentary."
    )
    parser.add_argument(
        "--output",
        default="app/commentary/audience_model.json",
        help="Path to the output model bundle JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic sample generation.",
    )
    return parser


def _seed_examples() -> list[tuple[dict[str, object], str]]:
    examples: list[tuple[dict[str, object], str]] = []
    for educational_mode in (False, True):
        for verbosity in COMMENTARY_VERBOSITY:
            for style in COMMENTARY_STYLES:
                for football_knowledge in KNOWLEDGE_OPTIONS:
                    signals = {
                        "educational_mode": educational_mode,
                        "verbosity": verbosity,
                        "style": style,
                        "football_knowledge": football_knowledge,
                    }
                    label, _source = infer_audience_level_rules(signals)
                    weight = 1
                    if educational_mode:
                        weight += 2
                    if football_knowledge in {"low", "high"}:
                        weight += 2
                    if style in {"analytical", "coach"} and verbosity in {"low", "medium"}:
                        weight += 1
                    for _ in range(weight):
                        examples.append((signals.copy(), label))
    return examples


def _augment_examples(seed: int) -> list[tuple[dict[str, object], str]]:
    rng = random.Random(seed)
    examples: list[tuple[dict[str, object], str]] = []
    base_examples = _seed_examples()
    for signals, label in base_examples:
        examples.append((signals.copy(), label))
        for _ in range(6):
            noisy = signals.copy()
            if rng.random() < 0.2:
                noisy["verbosity"] = rng.choice(COMMENTARY_VERBOSITY)
            if rng.random() < 0.15:
                noisy["style"] = rng.choice(COMMENTARY_STYLES)
            if rng.random() < 0.15:
                noisy["football_knowledge"] = rng.choice(KNOWLEDGE_OPTIONS)
            if rng.random() < 0.1:
                noisy["educational_mode"] = not bool(noisy["educational_mode"])
            examples.append((noisy, infer_audience_level_rules(noisy)[0]))
    return examples


def _fit_classifier(examples: list[tuple[dict[str, object], str]]) -> tuple[LogisticRegression, dict[str, object]]:
    vectors = []
    labels = []
    for signals, label in examples:
        vector, _normalized = build_audience_feature_vector(signals)
        vectors.append(vector)
        labels.append(label)

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )
    classifier.fit(vectors, labels)
    predictions = classifier.predict(vectors)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    metrics = {
        "training_accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "sample_count": len(examples),
        "class_report": report,
    }
    return classifier, metrics


def main() -> int:
    args = build_arg_parser().parse_args()
    examples = _augment_examples(args.seed)
    classifier, metrics = _fit_classifier(examples)

    output_path = (BACKEND_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "model_type": "multinomial_logistic_regression",
        "feature_names": list(AUDIENCE_MODEL_FEATURES),
        "classes": list(classifier.classes_),
        "coefficients": classifier.coef_.tolist(),
        "intercepts": classifier.intercept_.tolist(),
        "min_confidence": 0.5,
        "metadata": {
            "seed": args.seed,
            "training_metrics": metrics,
            "training_source": "synthetic bootstrapped audience-preference samples",
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
