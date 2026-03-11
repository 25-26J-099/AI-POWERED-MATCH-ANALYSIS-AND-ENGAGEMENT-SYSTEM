"""Mappings for player style cluster labels."""

from __future__ import annotations

STYLE_CLUSTER_LABELS = {
    0: "Ball Playing Defender",
    1: "Central Midfielder",
    2: "Box to Box Midfielder",
    3: "Attacking Playmaker",
    4: "Pressing Midfielder",
    5: "Wide Winger",
    6: "Ball Winning Midfielder",
    7: "Support Midfielder",
    8: "Low Involvement Player",
    9: "Second Striker",
    10: "Deep Playmaker",
    11: "High Press Forward",
    12: "Poacher",
    13: "Target Striker",
}


def get_style_cluster_label(cluster_id: int | None) -> str | None:
    if cluster_id is None:
        return None
    return STYLE_CLUSTER_LABELS.get(int(cluster_id), f"Cluster {cluster_id}")
