from __future__ import annotations

from typing import Any

from app.commentary import tac_commentary as tac
from app.tactical_gnn.inference import predict_tactical_snapshot
from app.tactical_gnn.schemas import TacticalGNNConfig


def compare_tactical_predictions(
    *,
    event_id: str,
    event_data: dict[str, Any] | None,
    freeze_frame_data: dict[str, Any] | None,
    team_name: str | None = None,
    player_name: str | None = None,
    sequence_summary: str | None = None,
    model_path: str | None = None,
    config: TacticalGNNConfig | None = None,
) -> dict[str, Any]:
    heuristic_prediction = tac._normalize_prediction_for_commentary(  # noqa: SLF001 - internal reuse
        tac._heuristic_tactical_prediction(event_data, freeze_frame_data)  # noqa: SLF001 - internal reuse
    )
    gnn_prediction = tac._normalize_prediction_for_commentary(  # noqa: SLF001 - internal reuse
        predict_tactical_snapshot(
            event_data,
            freeze_frame_data,
            model_path=model_path,
            config=config,
            heuristic_fallback=tac._heuristic_tactical_prediction,  # noqa: SLF001 - internal reuse
        )
    )
    final_prediction = tac.get_tactical_analysis(
        event_data,
        freeze_frame_data,
        prefer_gnn=True,
        model_path=model_path,
        config=config,
    )
    tactical_description = tac.compose_tactical_description(
        team=team_name,
        player=player_name,
        sequence_summary=sequence_summary,
        gnn_pred=final_prediction,
        opposition_effect=final_prediction.get("opposition_effect"),
        support_context=final_prediction.get("support_context"),
    )

    disagreement_heads = [
        head
        for head in ("formation_approx", "team_shape", "attacking_structure", "defensive_block", "defensive_shape")
        if heuristic_prediction.get(head) != gnn_prediction.get(head)
    ]

    return {
        "event_id": event_id,
        "gnn": gnn_prediction,
        "heuristic": heuristic_prediction,
        "used_fallback": gnn_prediction.get("model_used") != "gnn",
        "final_tactical_labels": final_prediction,
        "tactical_description": tactical_description,
        "disagreement_heads": disagreement_heads,
    }
