import numpy as np

from app.config.pipeline_config import PipelineConfig
from app.event_detection.jersey_ocr import JerseyOCR, OCRResult, ProcessedVariant


class FakeReader:
    def __init__(self, results):
        self.results = results

    def readtext(self, image, detail=1, paragraph=False, allowlist=None):
        del image, detail, paragraph, allowlist
        return self.results


def _mock_reader(monkeypatch, results):
    def fake_init(self):
        self.reader = FakeReader(results)
    monkeypatch.setattr(JerseyOCR, "_initialize_reader", fake_init)


def test_crop_selector_returns_multiple_ranked_candidates(monkeypatch):
    _mock_reader(monkeypatch, [])
    ocr = JerseyOCR(PipelineConfig())
    frame = np.zeros((160, 120, 3), dtype=np.uint8)
    frame[10:70, 30:90] = 255

    candidates = ocr.crop_jersey_candidates(frame, (20, 10, 100, 140))

    assert len(candidates) >= 2
    assert candidates[0].quality_score >= candidates[-1].quality_score


def test_build_ocr_variants_generates_heavy_preprocessing_paths(monkeypatch):
    _mock_reader(monkeypatch, [])
    ocr = JerseyOCR(PipelineConfig())
    crop = np.full((40, 20, 3), 180, dtype=np.uint8)
    crop[10:30, 8:12] = 20

    variants = ocr.build_ocr_variants(crop, crop_name="upper_torso")
    names = {variant.name for variant in variants}

    assert len(variants) >= 4
    assert "unsharp_gray" in names
    assert "adaptive_bin" in names


def test_extract_number_filters_non_numeric_and_low_confidence(monkeypatch):
    _mock_reader(
        monkeypatch,
        [
            (None, "AB", 0.95),
            (None, "109", 0.99),
            (None, "12", 0.72),
            (None, "8", 0.40),
        ],
    )
    ocr = JerseyOCR(PipelineConfig())

    result = ocr.extract_number(np.zeros((20, 20, 3), dtype=np.uint8))

    assert result.number == "12"
    assert result.confidence >= 0.71


def test_ensemble_fusion_prefers_best_supported_number(monkeypatch):
    _mock_reader(monkeypatch, [])
    ocr = JerseyOCR(PipelineConfig())

    variants = [
        ProcessedVariant(np.full((10, 10), 10, dtype=np.uint8), "adaptive_bin", "upper_torso", 0.8, 0.82, 90.0),
        ProcessedVariant(np.full((10, 10), 20, dtype=np.uint8), "unsharp_gray", "upper_torso", 0.78, 0.82, 85.0),
        ProcessedVariant(np.full((10, 10), 30, dtype=np.uint8), "clahe_gray", "center_strip", 0.76, 0.74, 70.0),
    ]

    def fake_run(self, image):
        marker = int(np.mean(image))
        if marker == 10:
            return [OCRResult(number="10", confidence=0.81)]
        if marker == 20:
            return [OCRResult(number="10", confidence=0.79)]
        if marker == 30:
            return [OCRResult(number="18", confidence=0.78)]
        return []

    monkeypatch.setattr(JerseyOCR, "_run_easyocr", fake_run)

    result = ocr.aggregate_variant_results(variants)

    assert result.number == "10"
    assert result.support_count == 2
    assert len(result.variant_sources) == 2


def test_temporal_fusion_locks_and_resists_replacement(monkeypatch):
    _mock_reader(monkeypatch, [])
    ocr = JerseyOCR(PipelineConfig())

    result1 = OCRResult(number="10", confidence=0.85, support_count=2, quality_score=0.8)
    result2 = OCRResult(number="10", confidence=0.79, support_count=2, quality_score=0.78)
    result3 = OCRResult(number="11", confidence=0.92, support_count=2, quality_score=0.88)

    stable_number, _, state = ocr.update_track(7, result1, frame_idx=1)
    stable_number, _, state = ocr.update_track(7, result2, frame_idx=2)
    replacement, _, replacement_state = ocr.update_track(7, result3, frame_idx=3)

    assert stable_number == "10"
    assert state in {"candidate", "stable"}
    assert replacement == "10"
    assert replacement_state == "stable"


def test_process_tracks_updates_track_attributes(monkeypatch):
    _mock_reader(monkeypatch, [(None, "9", 0.91)])
    config = PipelineConfig()
    config.ocr.update_interval = 1
    ocr = JerseyOCR(config)

    class DummyTrack:
        def __init__(self):
            self.bbox = (10, 10, 40, 90)
            self.frames_tracked = 3
            self.is_ball = False
            self.is_referee = False
            self.jersey_number = None
            self.jersey_confidence = 0.0
            self.jersey_stability = "unknown"

    frame = np.full((120, 160, 3), 200, dtype=np.uint8)
    frame[15:50, 18:30] = 40
    tracks = {5: DummyTrack()}

    results = ocr.process_tracks(frame, tracks, frame_idx=5)

    assert results[5] == "9"
    assert tracks[5].jersey_number == "9"
    assert tracks[5].jersey_confidence >= 0.90
    assert tracks[5].jersey_stability in {"candidate", "stable"}
