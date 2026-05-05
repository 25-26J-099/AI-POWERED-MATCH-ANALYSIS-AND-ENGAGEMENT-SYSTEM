import numpy as np

from app.config.pipeline_config import PipelineConfig
from app.event_detection.team_assigner import TeamAssigner


def test_team_color_metadata_maps_detected_clusters_to_user_team_names():
    assigner = TeamAssigner(PipelineConfig())
    assigner.is_fitted = True
    assigner.team_colors = {
        0: np.array([15.0, 220.0, 220.0]),
        1: np.array([110.0, 220.0, 220.0]),
    }

    metadata = assigner.get_team_color_metadata({0: "Colombo Lions", 1: "Kandy Blues"})

    assert metadata[0]["detected_label"] == "Team 1"
    assert metadata[0]["team_name"] == "Colombo Lions"
    assert metadata[0]["color_name"] == "Orange"
    assert metadata[0]["hex"].startswith("#")
    assert metadata[1]["detected_label"] == "Team 2"
    assert metadata[1]["team_name"] == "Kandy Blues"
    assert metadata[1]["color_name"] == "Blue"
