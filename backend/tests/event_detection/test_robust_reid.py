import numpy as np

from app.config.pipeline_config import PipelineConfig
from app.event_detection.robust_reid import RobustReIDSystem


class DummyTrack:
    def __init__(self, det_id, bbox, team_id=0, jersey_number=None, jersey_stability=None):
        self.track_id = det_id
        self.bbox = bbox
        self.confidence = 0.9
        self.class_id = 0
        self.class_name = "player"
        self.team_id = team_id
        self.jersey_number = jersey_number
        self.jersey_confidence = 0.9 if jersey_number is not None else 0.0
        self.jersey_stability = jersey_stability or ("stable" if jersey_number is not None else "unknown")
        self.is_ball = False
        self.is_referee = False
        self.frames_tracked = 1
        self.frames_lost = 0
        self.is_active = True
        self.position_history = [self.center]

    @property
    def center(self):
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


class FakeReIDModel:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.backend = "fake"

    def crop_player(self, _frame, bbox):
        return tuple(int(v) for v in bbox)

    def extract_embedding(self, crop):
        return self.embeddings.get(crop)

    def compute_similarity(self, emb1, emb2):
        emb1 = np.asarray(emb1, dtype=np.float32)
        emb2 = np.asarray(emb2, dtype=np.float32)
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(emb1, emb2))


def _track(det_id, bbox, team_id=0, jersey_number=None, jersey_stability=None):
    return DummyTrack(det_id, bbox, team_id=team_id, jersey_number=jersey_number, jersey_stability=jersey_stability)


def _config():
    config = PipelineConfig()
    config.reid.use_enhanced_temporal = False
    config.reid.use_adaptive_thresholds = False
    config.reid.similarity_threshold = 0.78
    config.reid.combined_threshold = 0.55
    config.reid.spatial_threshold = 200.0
    return config


def test_process_frame_reuses_stable_identity_for_same_detection_id():
    config = _config()
    system = RobustReIDSystem(config)
    system.reid_model = FakeReIDModel(
        {
            (10, 10, 40, 90): np.array([1.0, 0.0, 0.0], dtype=np.float32),
            (12, 12, 42, 92): np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    stable_tracks_1 = system.process_frame(frame, {101: _track(101, (10, 10, 40, 90))}, frame_idx=1)
    stable_tracks_2 = system.process_frame(frame, {101: _track(101, (12, 12, 42, 92))}, frame_idx=2)

    assert list(stable_tracks_1.keys()) == [1]
    assert list(stable_tracks_2.keys()) == [1]
    assert stable_tracks_2[1].track_id == 1


def test_process_frame_matches_reappearance_to_existing_identity():
    config = _config()
    system = RobustReIDSystem(config)
    system.reid_model = FakeReIDModel(
        {
            (10, 10, 40, 90): np.array([1.0, 0.0, 0.0], dtype=np.float32),
            (14, 12, 44, 92): np.array([0.99, 0.01, 0.0], dtype=np.float32),
        }
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    system.process_frame(frame, {101: _track(101, (10, 10, 40, 90))}, frame_idx=1)
    system.process_frame(frame, {}, frame_idx=2)
    stable_tracks = system.process_frame(frame, {202: _track(202, (14, 12, 44, 92))}, frame_idx=3)

    assert list(stable_tracks.keys()) == [1]
    assert stable_tracks[1].track_id == 1


def test_process_frame_creates_new_identity_when_similarity_is_low():
    config = _config()
    system = RobustReIDSystem(config)
    system.reid_model = FakeReIDModel(
        {
            (10, 10, 40, 90): np.array([1.0, 0.0, 0.0], dtype=np.float32),
            (80, 10, 110, 90): np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    stable_tracks_1 = system.process_frame(frame, {101: _track(101, (10, 10, 40, 90), team_id=0)}, frame_idx=1)
    stable_tracks_2 = system.process_frame(frame, {202: _track(202, (80, 10, 110, 90), team_id=0)}, frame_idx=2)

    assert list(stable_tracks_1.keys()) == [1]
    assert list(stable_tracks_2.keys()) == [2]


def test_team_mismatch_blocks_cross_team_reassignment():
    config = _config()
    system = RobustReIDSystem(config)
    system.reid_model = FakeReIDModel(
        {
            (10, 10, 40, 90): np.array([1.0, 0.0, 0.0], dtype=np.float32),
            (14, 12, 44, 92): np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    system.process_frame(frame, {101: _track(101, (10, 10, 40, 90), team_id=0)}, frame_idx=1)
    system.process_frame(frame, {}, frame_idx=2)
    stable_tracks = system.process_frame(frame, {202: _track(202, (14, 12, 44, 92), team_id=1)}, frame_idx=3)

    assert list(stable_tracks.keys()) == [2]


def test_process_frame_preserves_stable_track_mapping_shape():
    config = _config()
    system = RobustReIDSystem(config)
    system.reid_model = FakeReIDModel({(10, 10, 40, 90): np.array([1.0, 0.0, 0.0], dtype=np.float32)})
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    stable_tracks = system.process_frame(frame, {77: _track(77, (10, 10, 40, 90))}, frame_idx=1)

    assert 1 in stable_tracks
    assert stable_tracks[1].track_id == 1


def test_robust_reid_exposes_backend_status_metadata():
    config = _config()
    system = RobustReIDSystem(config)

    status = system.get_backend_status()

    assert "backend" in status
    assert "resolved_config_path" in status
    assert "resolved_weights_path" in status
    assert "fallback_reason" in status


def test_jersey_match_can_rescue_medium_similarity_match():
    config = _config()
    config.reid.similarity_threshold = 0.85
    system = RobustReIDSystem(config)
    system.reid_model = FakeReIDModel(
        {
            (10, 10, 40, 90): np.array([1.0, 0.0, 0.0], dtype=np.float32),
            (12, 10, 42, 90): np.array([0.82, 0.18, 0.0], dtype=np.float32),
        }
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    system.process_frame(frame, {101: _track(101, (10, 10, 40, 90), jersey_number="10")}, frame_idx=1)
    system.identities[1].jersey_stability = "stable"
    system.process_frame(frame, {}, frame_idx=2)
    stable_tracks = system.process_frame(
        frame,
        {202: _track(202, (12, 10, 42, 90), jersey_number="10")},
        frame_idx=3,
    )

    assert list(stable_tracks.keys()) == [1]


def test_stable_jersey_conflict_forces_new_identity():
    config = _config()
    system = RobustReIDSystem(config)
    system.reid_model = FakeReIDModel(
        {
            (10, 10, 40, 90): np.array([1.0, 0.0, 0.0], dtype=np.float32),
            (12, 10, 42, 90): np.array([0.98, 0.02, 0.0], dtype=np.float32),
        }
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    system.process_frame(frame, {101: _track(101, (10, 10, 40, 90), jersey_number="10")}, frame_idx=1)
    system.identities[1].jersey_stability = "stable"
    system.process_frame(frame, {}, frame_idx=2)
    stable_tracks = system.process_frame(
        frame,
        {202: _track(202, (12, 10, 42, 90), jersey_number="11", jersey_stability="stable")},
        frame_idx=3,
    )

    assert list(stable_tracks.keys()) == [2]
