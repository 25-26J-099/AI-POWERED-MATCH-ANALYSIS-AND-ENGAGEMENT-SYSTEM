from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_FORMATION_LABELS = (
    "4-3-3",
    "4-4-2",
    "4-2-3-1",
    "3-5-2",
    "3-4-3",
    "5-3-2",
    "4-1-4-1",
    "Unclear",
)
DEFAULT_TEAM_SHAPE_LABELS = (
    "Compact Shape",
    "Wide Shape",
    "Stretched Shape",
    "Vertical Shape",
    "Balanced Shape",
    "Unknown",
)
DEFAULT_ATTACKING_STRUCTURE_LABELS = (
    "Wide Structure",
    "Central Overload",
    "Vertical Support Structure",
    "Balanced Structure",
    "Unknown",
)
DEFAULT_DEFENSIVE_BLOCK_LABELS = (
    "High Press",
    "Mid Block",
    "Low Block",
    "Unknown",
)
DEFAULT_DEFENSIVE_SHAPE_LABELS = (
    "Compact Narrow",
    "Compact Balanced",
    "Compact Wide",
    "Spread Narrow",
    "Spread Balanced",
    "Spread Wide",
    "Unknown",
)
LABEL_HEADS = (
    "formation",
    "team_shape",
    "attacking_structure",
    "defensive_block",
    "defensive_shape",
)


def _coerce_location(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("location values must be numeric") from exc
    raise ValueError("location must contain at least two numeric coordinates")


class FreezeFramePlayer(BaseModel):
    model_config = ConfigDict(extra="allow")

    location: tuple[float, float]
    teammate: bool = False
    keeper: bool = False
    actor: bool = False
    player_id: str | int | None = None
    team_id: str | int | None = None

    @field_validator("location", mode="before")
    @classmethod
    def validate_location(cls, value: Any) -> tuple[float, float]:
        coerced = _coerce_location(value)
        if coerced is None:
            raise ValueError("location is required")
        return coerced


class TacticalLabelSet(BaseModel):
    formation: str | None = None
    team_shape: str | None = None
    attacking_structure: str | None = None
    defensive_block: str | None = None
    defensive_shape: str | None = None


class TacticalDatasetSample(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_id: str
    event_type: str | None = None
    event_location: tuple[float, float] | None = None
    attacking_right: bool = True
    freeze_frame: list[FreezeFramePlayer] = Field(default_factory=list)
    labels: TacticalLabelSet | None = None
    label_sources: dict[str, str] = Field(default_factory=dict)
    source_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_location", mode="before")
    @classmethod
    def validate_event_location(cls, value: Any) -> tuple[float, float] | None:
        return _coerce_location(value)


class GraphMetadataModel(BaseModel):
    num_nodes: int
    num_edges: int
    normalization_applied: bool
    missing_features: list[str] = Field(default_factory=list)
    edge_strategy: str = "knn"
    attacking_right: bool = True
    insufficient_data: bool = False
    fallback_reason: str | None = None


class TacticalPredictionModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_used: str = "heuristic"
    formation: str = "Unclear"
    formation_confidence: float = 0.0
    team_shape: str = "Unknown"
    team_shape_confidence: float = 0.0
    attacking_structure: str = "Unknown"
    attacking_structure_confidence: float = 0.0
    defensive_block: str = "Unknown"
    defensive_block_confidence: float = 0.0
    defensive_shape: str = "Unknown"
    defensive_shape_confidence: float = 0.0
    support_context: str = "Support context unavailable."
    opposition_effect: str = "Opposition effect unavailable."
    graph_metadata: GraphMetadataModel = Field(
        default_factory=lambda: GraphMetadataModel(
            num_nodes=0,
            num_edges=0,
            normalization_applied=False,
            missing_features=[],
        )
    )


@dataclass(slots=True)
class TacticalGNNConfig:
    enabled: bool = True
    model_path: str | None = None
    device: str = "cpu"
    edge_strategy: str = "knn"
    k_neighbors: int = 4
    radius: float = 18.0
    min_players: int = 6
    hidden_dim: int = 96
    num_layers: int = 3
    dropout: float = 0.2
    confidence_threshold: float = 0.4
    use_heuristic_fallback: bool = True
    formation_labels: tuple[str, ...] = DEFAULT_FORMATION_LABELS
    team_shape_labels: tuple[str, ...] = DEFAULT_TEAM_SHAPE_LABELS
    attacking_structure_labels: tuple[str, ...] = DEFAULT_ATTACKING_STRUCTURE_LABELS
    defensive_block_labels: tuple[str, ...] = DEFAULT_DEFENSIVE_BLOCK_LABELS
    defensive_shape_labels: tuple[str, ...] = DEFAULT_DEFENSIVE_SHAPE_LABELS
    pitch_length: float = 120.0
    pitch_width: float = 80.0
    local_density_radius: float = 12.0
    training_batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    val_split: float = 0.2
    seed: int = 7
    label_maps: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label_maps:
            self.label_maps = {
                "formation": list(self.formation_labels),
                "team_shape": list(self.team_shape_labels),
                "attacking_structure": list(self.attacking_structure_labels),
                "defensive_block": list(self.defensive_block_labels),
                "defensive_shape": list(self.defensive_shape_labels),
            }
        self.edge_strategy = str(self.edge_strategy or "knn").lower()
        self.k_neighbors = max(1, int(self.k_neighbors))
        self.radius = float(self.radius)
        self.min_players = max(2, int(self.min_players))
        self.hidden_dim = max(16, int(self.hidden_dim))
        self.num_layers = max(1, int(self.num_layers))
        self.dropout = float(self.dropout)
        self.confidence_threshold = float(self.confidence_threshold)
        self.training_batch_size = max(1, int(self.training_batch_size))
        self.epochs = max(1, int(self.epochs))
        self.val_split = min(max(float(self.val_split), 0.0), 0.5)
        self.seed = int(self.seed)

    def num_classes_for_head(self, head: str) -> int:
        return len(self.label_maps[head])


def validate_tactical_sample(payload: dict[str, Any]) -> TacticalDatasetSample:
    return TacticalDatasetSample.model_validate(payload)
