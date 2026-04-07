from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch

from app.tactical_gnn.graph_builder import build_graph_from_snapshot
from app.tactical_gnn.model import create_model
from app.tactical_gnn.schemas import LABEL_HEADS, TacticalGNNConfig, TacticalPredictionModel
from app.tactical_gnn.utils import config_from_settings, resolve_device, resolve_model_path

LOGGER = logging.getLogger(__name__)

HeuristicFallback = Callable[[dict[str, Any] | None, dict[str, Any] | None], dict[str, Any]]


def _default_prediction(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = TacticalPredictionModel().model_dump()
    if metadata is not None:
        payload["graph_metadata"] = {**payload["graph_metadata"], **metadata}
    return payload


def _normalize_fallback_result(fallback_result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    result = _default_prediction(metadata)
    result.update(
        {
            "model_used": "heuristic",
            "formation": fallback_result.get("formation", fallback_result.get("formation_approx", "Unclear")),
            "team_shape": fallback_result.get("team_shape", "Unknown"),
            "attacking_structure": fallback_result.get("attacking_structure", "Unknown"),
            "defensive_block": fallback_result.get("defensive_block", "Unknown"),
            "defensive_shape": fallback_result.get("defensive_shape", "Unknown"),
            "support_context": fallback_result.get("support_context", "Support context unavailable."),
            "opposition_effect": fallback_result.get("opposition_effect", "Opposition effect unavailable."),
        }
    )
    for head in LABEL_HEADS:
        result[f"{head}_confidence"] = float(fallback_result.get(f"{head}_confidence", 0.0))
    result["graph_metadata"] = {**result["graph_metadata"], **metadata}
    return result


def _fallback(
    event_data: dict[str, Any] | None,
    freeze_frame_data: dict[str, Any] | None,
    metadata: dict[str, Any],
    heuristic_fallback: HeuristicFallback | None,
) -> dict[str, Any]:
    if heuristic_fallback is None:
        return _default_prediction(metadata)
    return _normalize_fallback_result(heuristic_fallback(event_data, freeze_frame_data), metadata)


def _label_from_logits(logits: torch.Tensor, labels: list[str], confidence_threshold: float) -> tuple[str, float]:
    probabilities = torch.softmax(logits, dim=-1)
    confidence, index = torch.max(probabilities, dim=-1)
    label = labels[int(index.item())]
    if float(confidence.item()) < confidence_threshold:
        for candidate in ("Unknown", "Unclear"):
            if candidate in labels:
                label = candidate
                break
    return label, float(confidence.item())


@lru_cache(maxsize=4)
def _load_model_bundle(model_path: str, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_config = TacticalGNNConfig(**checkpoint.get("config", {}))
    label_maps = checkpoint.get("label_maps", checkpoint_config.label_maps)
    model = create_model(
        config=checkpoint_config,
        input_dim=int(checkpoint["input_dim"]),
        edge_dim=int(checkpoint["edge_dim"]),
        label_maps=label_maps,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, {
        "label_maps": label_maps,
        "config": checkpoint_config,
    }


def predict_tactical_snapshot(
    event_data: dict[str, Any] | None,
    freeze_frame_data: dict[str, Any] | None,
    model_path: str | None = None,
    config: TacticalGNNConfig | None = None,
    heuristic_fallback: HeuristicFallback | None = None,
) -> dict[str, Any]:
    runtime_config = config or config_from_settings()
    graph = build_graph_from_snapshot(event_data, freeze_frame_data, config=runtime_config)
    metadata = graph.metadata.model_dump()

    if not runtime_config.enabled:
        metadata["fallback_reason"] = "gnn tactical analysis disabled"
        return _fallback(event_data, freeze_frame_data, metadata, heuristic_fallback)
    if graph.metadata.insufficient_data:
        return _fallback(event_data, freeze_frame_data, metadata, heuristic_fallback)

    resolved_model_path = resolve_model_path(model_path, fallback=runtime_config.model_path)
    if not resolved_model_path or not Path(resolved_model_path).exists():
        metadata["fallback_reason"] = f"model checkpoint not found: {resolved_model_path or 'unset'}"
        return _fallback(event_data, freeze_frame_data, metadata, heuristic_fallback)

    try:
        device = resolve_device(runtime_config.device)
        model, bundle = _load_model_bundle(resolved_model_path, device)
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        graph.edge_attr = graph.edge_attr.to(device)
        graph.batch = graph.batch.to(device)
        graph.teammate_mask = graph.teammate_mask.to(device)
        with torch.no_grad():
            outputs = model(graph)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Falling back to heuristic tactical analysis: %s", exc)
        metadata["fallback_reason"] = str(exc)
        return _fallback(event_data, freeze_frame_data, metadata, heuristic_fallback)

    label_maps = bundle["label_maps"]
    checkpoint_config: TacticalGNNConfig = bundle["config"]
    prediction = _default_prediction(metadata)
    prediction["model_used"] = "gnn"

    for head_name in LABEL_HEADS:
        label, confidence = _label_from_logits(
            outputs[head_name][0].cpu(),
            label_maps[head_name],
            confidence_threshold=checkpoint_config.confidence_threshold,
        )
        prediction[head_name] = label
        prediction[f"{head_name}_confidence"] = confidence

    if prediction["defensive_shape"] not in {"Unknown", ""} and prediction["defensive_block"] not in {"Unknown", ""}:
        prediction["defensive_shape"] = f"{prediction['defensive_shape']} {prediction['defensive_block']}"
    return prediction
