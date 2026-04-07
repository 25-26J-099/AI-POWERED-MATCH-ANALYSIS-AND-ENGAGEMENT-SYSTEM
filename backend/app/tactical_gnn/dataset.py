from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.tactical_gnn.schemas import LABEL_HEADS, TacticalDatasetSample, validate_tactical_sample
from app.tactical_gnn.utils import extract_freeze_frame_players, infer_attacking_direction_right

LOGGER = logging.getLogger(__name__)

REAL_EXPORT_PRIORITY = ("events", "freeze_frames", "statsbomb", "generic")

LABEL_NORMALIZATION_MAP: dict[str, dict[str, str]] = {
    "formation": {
        "unknown": "Unclear",
        "unclear": "Unclear",
    },
    "team_shape": {
        "compact": "Compact Shape",
        "wide": "Wide Shape",
        "stretched": "Stretched Shape",
        "vertical": "Vertical Shape",
        "balanced": "Balanced Shape",
        "unknown": "Unknown",
    },
    "attacking_structure": {
        "wide_structure": "Wide Structure",
        "central_overload": "Central Overload",
        "vertical_support": "Vertical Support Structure",
        "balanced_structure": "Balanced Structure",
        "rest_defense_stable": "Balanced Structure",
        "unknown": "Unknown",
    },
    "defensive_block": {
        "high_press": "High Press",
        "mid_block": "Mid Block",
        "low_block": "Low Block",
        "unknown": "Unknown",
    },
    "defensive_shape": {
        "compact_narrow": "Compact Narrow",
        "compact_balanced": "Compact Balanced",
        "compact_wide": "Compact Wide",
        "spread_narrow": "Spread Narrow",
        "spread_balanced": "Spread Balanced",
        "spread_wide": "Spread Wide",
        "back_five_compact": "Compact Balanced",
        "disorganized": "Unknown",
        "unknown": "Unknown",
    },
}


@dataclass(slots=True)
class DatasetPreparationReport:
    input_path: str
    source_files: list[str] = field(default_factory=list)
    detected_formats: list[str] = field(default_factory=list)
    total_records: int = 0
    usable_samples: int = 0
    dropped_samples: int = 0
    dropped_reasons: dict[str, int] = field(default_factory=dict)
    label_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    label_source_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    active_heads: list[str] = field(default_factory=list)
    schema_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _event_type_name(event: dict[str, Any]) -> str | None:
    event_type = event.get("type")
    if isinstance(event_type, dict):
        return event_type.get("name")
    if isinstance(event_type, str):
        return event_type
    return event.get("event_type") or event.get("type_name")


def _location_from_payload(payload: dict[str, Any]) -> list[float] | None:
    for key in ("event_location", "location", "position"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return [float(value[0]), float(value[1])]
            except (TypeError, ValueError):
                return None
    x_value = payload.get("x")
    y_value = payload.get("y")
    if x_value is not None and y_value is not None:
        try:
            return [float(x_value), float(y_value)]
        except (TypeError, ValueError):
            return None
    return None


def _normalize_player_entry(player: dict[str, Any], actor_team_id: Any, actor_id: Any) -> dict[str, Any] | None:
    location = player.get("location")
    if not isinstance(location, (list, tuple)) or len(location) < 2:
        return None
    try:
        normalized = {
            "location": [float(location[0]), float(location[1])],
            "player_id": player.get("player_id") or player.get("id"),
            "team_id": player.get("team_id"),
            "keeper": bool(player.get("keeper", False)),
            "actor": bool(player.get("actor", False)),
        }
    except (TypeError, ValueError):
        return None

    teammate = player.get("teammate")
    if teammate is None and actor_team_id is not None and player.get("team_id") is not None:
        teammate = player.get("team_id") == actor_team_id
    normalized["teammate"] = bool(teammate)
    if not normalized["actor"] and actor_id is not None and normalized["player_id"] is not None:
        normalized["actor"] = normalized["player_id"] == actor_id
    return normalized


def _extract_freeze_frame_list(record: dict[str, Any]) -> list[dict[str, Any]]:
    freeze_frame = record.get("freeze_frame")
    if isinstance(freeze_frame, dict):
        return freeze_frame.get("players", [])
    if isinstance(freeze_frame, list):
        return freeze_frame
    return []


def _coerce_label(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        return raw_value.get("name") or raw_value.get("label")
    text = str(raw_value).strip()
    return text or None


def _canonicalize_label_key(raw_value: str) -> str:
    return raw_value.strip().lower().replace("-", "_").replace(" ", "_")


def normalize_label_value(head: str, raw_value: Any) -> str | None:
    label = _coerce_label(raw_value)
    if not label:
        return None
    if label in {"Unknown", "Unclear"}:
        return label

    normalized = LABEL_NORMALIZATION_MAP.get(head, {}).get(_canonicalize_label_key(label))
    if normalized:
        return normalized

    # Keep already-compatible commentary labels untouched.
    return label


def _extract_label_sources(record: dict[str, Any]) -> dict[str, str]:
    raw_sources = record.get("label_sources")
    if not isinstance(raw_sources, dict):
        raw_sources = record.get("label_source")
    if isinstance(raw_sources, dict):
        return {str(head): str(source) for head, source in raw_sources.items() if source is not None}
    if raw_sources is None:
        return {}
    return {head: str(raw_sources) for head in LABEL_HEADS}


def _direct_label_candidates(record: dict[str, Any], lineups: dict[str, str] | None = None) -> tuple[dict[str, str | None], dict[str, str], dict[str, str | None]]:
    labels = {head: None for head in LABEL_HEADS}
    sources = {head: "missing" for head in LABEL_HEADS}
    raw_labels = {head: None for head in LABEL_HEADS}
    nested_labels = record.get("labels") if isinstance(record.get("labels"), dict) else {}
    tactical_labels = record.get("tactical_labels") if isinstance(record.get("tactical_labels"), dict) else {}
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    metadata_labels = metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}
    explicit_sources = _extract_label_sources(record)

    for head in LABEL_HEADS:
        candidate = (
            nested_labels.get(head)
            or tactical_labels.get(head)
            or metadata_labels.get(head)
            or record.get(head)
        )
        if head == "formation" and candidate is None:
            candidate = record.get("formation_approx")
        raw_label = _coerce_label(candidate)
        candidate_label = normalize_label_value(head, raw_label)
        if candidate_label:
            labels[head] = candidate_label
            raw_labels[head] = raw_label
            sources[head] = explicit_sources.get(head, "ground_truth")

    if labels["formation"] is None and lineups:
        team_id = record.get("team_id")
        team_name = record.get("team_name") or record.get("team")
        for key in (str(team_id), str(team_name) if team_name is not None else None):
            if key and key in lineups:
                labels["formation"] = normalize_label_value("formation", lineups[key]) or lineups[key]
                sources["formation"] = "ground_truth"
                raw_labels["formation"] = lineups[key]
                break
    return labels, sources, raw_labels


def _derive_pseudo_labels(event_payload: dict[str, Any], freeze_frame_payload: dict[str, Any]) -> dict[str, str]:
    from app.commentary.tac_commentary import analyze_tactical_snapshot, normalize_freeze_frame, normalize_location

    attacking_right = infer_attacking_direction_right(freeze_frame_payload)
    normalized_event = dict(event_payload)
    event_location = _location_from_payload(normalized_event)
    if event_location is not None:
        normalized_event["location"] = normalize_location(event_location, attacking_right)
    normalized_frame = normalize_freeze_frame(freeze_frame_payload, attacking_right)
    labels = analyze_tactical_snapshot(normalized_event, normalized_frame)
    return {
        "formation": normalize_label_value("formation", labels.get("formation_approx", "Unclear")) or "Unclear",
        "team_shape": normalize_label_value("team_shape", labels.get("team_shape", "Unknown")) or "Unknown",
        "attacking_structure": normalize_label_value(
            "attacking_structure",
            labels.get("attacking_structure", "Unknown"),
        ) or "Unknown",
        "defensive_block": normalize_label_value("defensive_block", labels.get("defensive_block", "Unknown")) or "Unknown",
        "defensive_shape": normalize_label_value("defensive_shape", labels.get("defensive_shape", "Unknown")) or "Unknown",
    }


def _candidate_files(dataset_path: Path) -> list[tuple[str, Path]]:
    if dataset_path.is_file():
        lowered = dataset_path.name.lower()
        if lowered.endswith("statsbomb_events.json"):
            detected_format = "statsbomb"
        elif "freeze_frames" in lowered:
            detected_format = "freeze_frames"
        elif lowered.endswith("events.json") or lowered.endswith("_events.json"):
            detected_format = "events"
        else:
            detected_format = "generic"
        return [(detected_format, dataset_path)]

    events_files: list[Path] = []
    freeze_files: list[Path] = []
    statsbomb_files: list[Path] = []
    generic_files: list[Path] = []
    for file_path in dataset_path.rglob("*"):
        if not file_path.is_file():
            continue
        lowered = file_path.name.lower()
        if file_path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            continue
        if lowered.endswith("statsbomb_events.json"):
            statsbomb_files.append(file_path)
        elif "freeze_frames" in lowered:
            freeze_files.append(file_path)
        elif lowered.endswith("events.json") or lowered.endswith("_events.json"):
            events_files.append(file_path)
        else:
            generic_files.append(file_path)

    discovered: list[tuple[str, Path]] = []
    if events_files:
        discovered.extend(("events", path) for path in sorted(events_files))
    elif freeze_files:
        discovered.extend(("freeze_frames", path) for path in sorted(freeze_files))
    elif statsbomb_files:
        discovered.extend(("statsbomb", path) for path in sorted(statsbomb_files))
    else:
        discovered.extend(("generic", path) for path in sorted(generic_files))
    return discovered


def _load_lineup_lookup(dataset_root: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for file_path in dataset_root.rglob("lineups*.json"):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        lineup_items = payload if isinstance(payload, list) else payload.get("lineups", [])
        if not isinstance(lineup_items, list):
            continue
        for item in lineup_items:
            if not isinstance(item, dict):
                continue
            formation = _coerce_label(item.get("formation"))
            if not formation:
                continue
            for key in (item.get("team_id"), item.get("team"), item.get("team_name"), item.get("name")):
                if key is not None:
                    lookup[str(key)] = formation
    return lookup


def _iter_generic_records(file_path: Path) -> list[dict[str, Any]]:
    if file_path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as csv_file:
            records = list(csv.DictReader(csv_file))
    else:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "samples" in payload:
            records = payload["samples"]
        elif isinstance(payload, dict) and "events" in payload:
            records = payload["events"]
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}")
    return [record for record in records if isinstance(record, dict)]


def _build_sample_from_record(
    record: dict[str, Any],
    source_path: Path,
    allow_pseudo_labels: bool,
    lineups: dict[str, str] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    freeze_players_raw = _extract_freeze_frame_list(record)
    actor_team_id = record.get("team_id")
    actor_id = record.get("player_id")
    normalized_players = [
        player
        for player in (
            _normalize_player_entry(player, actor_team_id=actor_team_id, actor_id=actor_id)
            for player in freeze_players_raw
            if isinstance(player, dict)
        )
        if player is not None
    ]
    if not normalized_players:
        return None, "missing_freeze_frame"

    event_id = record.get("event_id") or record.get("event_uuid") or record.get("id") or record.get("frame")
    if event_id is None:
        return None, "missing_event_id"

    event_location = _location_from_payload(record)
    if event_location is None and record.get("raw_data") and isinstance(record["raw_data"], dict):
        event_location = _location_from_payload(record["raw_data"])

    event_payload = {
        "location": event_location,
        "event_type": _event_type_name(record),
        "team_id": actor_team_id,
        "player_id": actor_id,
    }
    freeze_payload = {"freeze_frame": normalized_players}
    attacking_right = bool(record["attacking_right"]) if "attacking_right" in record else infer_attacking_direction_right(freeze_payload)
    labels, label_sources, raw_labels = _direct_label_candidates(record, lineups=lineups)
    if allow_pseudo_labels:
        pseudo_labels = _derive_pseudo_labels(event_payload, freeze_payload)
        for head, value in pseudo_labels.items():
            if labels[head] is None:
                labels[head] = value
                label_sources[head] = "pseudo_heuristic"

    sample = {
        "event_id": str(event_id),
        "event_type": _event_type_name(record),
        "event_location": event_location,
        "attacking_right": attacking_right,
        "freeze_frame": normalized_players,
        "labels": labels,
        "label_sources": label_sources,
        "source_path": str(source_path),
        "metadata": {
            "team_id": actor_team_id,
            "player_id": actor_id,
            "team_name": record.get("team_name") or record.get("team"),
            "record_format": "repo_export",
            "raw_labels": {head: value for head, value in raw_labels.items() if value is not None},
            "label_source": record.get("label_source"),
        },
    }
    try:
        validated = validate_tactical_sample(sample).model_dump(mode="json")
    except Exception as exc:
        return None, f"validation_error:{exc.__class__.__name__}"
    return validated, None


def prepare_tactical_dataset(
    dataset_path: str | Path,
    *,
    allow_pseudo_labels: bool = True,
) -> tuple[list[dict[str, Any]], DatasetPreparationReport]:
    path = Path(dataset_path)
    report = DatasetPreparationReport(input_path=str(path))
    if not path.exists():
        raise FileNotFoundError(f"dataset path does not exist: {path}")

    discovered_files = _candidate_files(path)
    if not discovered_files:
        raise FileNotFoundError(f"no supported tactical dataset files found under: {path}")

    lineups = _load_lineup_lookup(path if path.is_dir() else path.parent)
    if lineups:
        report.schema_notes.append("Detected lineup files with formation labels.")

    samples: list[dict[str, Any]] = []
    drop_counter: Counter[str] = Counter()
    label_counter = {head: Counter() for head in LABEL_HEADS}
    label_source_counter = {head: Counter() for head in LABEL_HEADS}
    seen_keys: set[tuple[str, str]] = set()

    for detected_format, file_path in discovered_files:
        report.source_files.append(str(file_path))
        if detected_format not in report.detected_formats:
            report.detected_formats.append(detected_format)
        records = _iter_generic_records(file_path)
        report.total_records += len(records)
        if detected_format == "events":
            report.schema_notes.append(f"{file_path.name}: detected repo events export with embedded freeze_frame.")
        elif detected_format == "freeze_frames":
            report.schema_notes.append(f"{file_path.name}: detected freeze frame sidecar export.")
        elif detected_format == "statsbomb":
            report.schema_notes.append(f"{file_path.name}: detected StatsBomb-style event export.")

        for record in records:
            sample, drop_reason = _build_sample_from_record(
                record,
                source_path=file_path,
                allow_pseudo_labels=allow_pseudo_labels,
                lineups=lineups,
            )
            if sample is None:
                drop_counter[drop_reason or "unknown"] += 1
                continue
            dedupe_key = (sample["event_id"], str(file_path))
            if dedupe_key in seen_keys:
                drop_counter["duplicate_event"] += 1
                continue
            seen_keys.add(dedupe_key)
            samples.append(sample)
            for head in LABEL_HEADS:
                label = sample.get("labels", {}).get(head)
                source = sample.get("label_sources", {}).get(head, "missing")
                if label:
                    label_counter[head][label] += 1
                label_source_counter[head][source] += 1

    report.usable_samples = len(samples)
    report.dropped_samples = int(sum(drop_counter.values()))
    report.dropped_reasons = dict(drop_counter)
    report.label_counts = {head: dict(counter) for head, counter in label_counter.items()}
    report.label_source_counts = {head: dict(counter) for head, counter in label_source_counter.items()}
    report.active_heads = [
        head
        for head in LABEL_HEADS
        if any(source != "missing" for source in report.label_source_counts.get(head, {}))
    ]
    return samples, report


def summarize_detected_schema(dataset_path: str | Path, allow_pseudo_labels: bool = True) -> dict[str, Any]:
    _, report = prepare_tactical_dataset(dataset_path, allow_pseudo_labels=allow_pseudo_labels)
    return report.to_dict()
