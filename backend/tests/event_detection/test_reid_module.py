import numpy as np
import pytest

from app.config.pipeline_config import PipelineConfig
from app.event_detection.reid_module import ReIDModel


def test_reid_model_backend_resolution_prefers_first_available(monkeypatch):
    config = PipelineConfig()
    config.reid.backend_priority = ("fastreid", "torchreid", "handcrafted")

    def fake_fastreid(self):
        return False

    def fake_torchreid(self):
        self.backend = "torchreid"
        return True

    monkeypatch.setattr(ReIDModel, "_init_fastreid_backend", fake_fastreid)
    monkeypatch.setattr(ReIDModel, "_init_torchreid_backend", fake_torchreid)

    model = ReIDModel(config=config.reid)

    assert model.backend == "torchreid"


def test_crop_player_rejects_invalid_boxes():
    config = PipelineConfig()
    model = ReIDModel(config=config.reid)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    assert model.crop_player(frame, (10, 10, 12, 20)) is None
    assert model.crop_player(frame, (-5, -5, 4, 8)) is None


def test_handcrafted_embedding_is_normalized():
    config = PipelineConfig()
    config.reid.backend_priority = ("handcrafted",)
    model = ReIDModel(config=config.reid)
    crop = np.full((96, 48, 3), 120, dtype=np.uint8)

    embedding = model.extract_embedding(crop)

    assert embedding is not None
    assert embedding.ndim == 1
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_similarity_is_symmetric_and_bounded():
    config = PipelineConfig()
    model = ReIDModel(config=config.reid)
    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.5, 0.5, 0.0], dtype=np.float32)

    sim12 = model.compute_similarity(emb1, emb2)
    sim21 = model.compute_similarity(emb2, emb1)

    assert np.isclose(sim12, sim21, atol=1e-6)
    assert -1.0 <= sim12 <= 1.0


def test_pipeline_config_uses_settings_driven_fastreid_defaults():
    config = PipelineConfig()

    assert config.reid.backend_priority[0] == "fastreid"
    assert config.reid.fastreid_config_path.endswith("models/reid/fastreid/configs/football_vit.yml")
    assert config.reid.model_path.endswith("models/reid/fastreid/weights/football_vit.pth")


def test_get_backend_status_reports_missing_fastreid_assets():
    config = PipelineConfig()
    config.reid.backend_priority = ("fastreid", "handcrafted")
    config.reid.fastreid_enabled = True
    config.reid.fastreid_config_path = "./models/reid/fastreid/configs/missing.yml"
    config.reid.model_path = "./models/reid/fastreid/weights/missing.pth"

    model = ReIDModel(config=config.reid)
    status = model.get_backend_status()

    assert status["backend"] == "handcrafted"
    assert "missing" in status["fallback_reason"].lower()
    assert status["resolved_config_path"].endswith("missing.yml")
    assert status["resolved_weights_path"].endswith("missing.pth")


def test_strict_fastreid_raises_when_backend_is_unavailable():
    config = PipelineConfig()
    config.reid.backend_priority = ("fastreid", "handcrafted")
    config.reid.strict_fastreid = True
    config.reid.fastreid_enabled = True
    config.reid.fastreid_config_path = "./models/reid/fastreid/configs/missing.yml"
    config.reid.model_path = "./models/reid/fastreid/weights/missing.pth"

    with pytest.raises(RuntimeError):
        ReIDModel(config=config.reid)


def test_reid_model_can_select_fastreid_when_assets_and_package_exist(monkeypatch):
    config = PipelineConfig()
    config.reid.backend_priority = ("fastreid", "handcrafted")

    monkeypatch.setattr("app.event_detection.reid_module.importlib.util.find_spec", lambda name: object())

    def fake_init(self):
        self.backend = "fastreid"
        self.backend_reason = "FastReID initialized successfully"
        self._register_backend_attempt("fastreid", True, self.backend_reason)
        return True

    monkeypatch.setattr(ReIDModel, "_init_fastreid_backend", fake_init)
    model = ReIDModel(config=config.reid)

    assert model.backend == "fastreid"
