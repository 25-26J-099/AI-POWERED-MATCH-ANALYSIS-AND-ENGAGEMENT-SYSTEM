"""Accuracy-first jersey number OCR for tracked football players."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None

logger = logging.getLogger(__name__)
BACKEND_ROOT = Path(__file__).resolve().parents[2]
_NUMBER_PATTERN = re.compile(r"\d{1,2}")


@dataclass
class CropCandidate:
    image: np.ndarray
    name: str
    quality_score: float
    blur_score: float
    contrast_score: float
    edge_score: float
    occupancy_score: float


@dataclass
class ProcessedVariant:
    image: np.ndarray
    name: str
    crop_name: str
    quality_score: float
    crop_quality_score: float
    blur_score: float


@dataclass
class OCRResult:
    number: Optional[str]
    confidence: float = 0.0
    support_count: int = 0
    variant_sources: tuple[str, ...] = ()
    quality_score: float = 0.0


@dataclass
class TrackObservation:
    number: str
    confidence: float
    frame_idx: int
    quality_score: float
    support_count: int


class JerseyOCR:
    """Independent OCR module optimized for blurred, small football jersey numbers."""

    def __init__(self, config):
        self.cfg = config.ocr if hasattr(config, "ocr") else config
        self.backend = getattr(self.cfg, "backend", "easyocr")
        self.confidence_threshold = float(getattr(self.cfg, "confidence_threshold", 0.5))
        self.update_interval = max(1, int(getattr(self.cfg, "update_interval", 5)))
        self.history_size = max(1, int(getattr(self.cfg, "history_size", 12)))
        self.top_region_ratio = float(getattr(self.cfg, "top_region_ratio", 0.55))
        self.resize_scale = float(getattr(self.cfg, "resize_scale", 4.0))
        self.min_crop_height = int(getattr(self.cfg, "min_crop_height", 24))
        self.min_crop_width = int(getattr(self.cfg, "min_crop_width", 12))
        self.use_thresholding = bool(getattr(self.cfg, "use_thresholding", True))
        self.center_strip_ratio = float(getattr(self.cfg, "center_strip_ratio", 0.58))
        self.expanded_region_ratio = float(getattr(self.cfg, "expanded_region_ratio", 0.7))
        self.horizontal_inset_ratio = float(getattr(self.cfg, "horizontal_inset_ratio", 0.18))
        self.horizontal_expand_ratio = float(getattr(self.cfg, "horizontal_expand_ratio", 0.1))
        self.max_crop_candidates = max(1, int(getattr(self.cfg, "max_crop_candidates", 3)))
        self.max_variants_per_crop = max(1, int(getattr(self.cfg, "max_variants_per_crop", 6)))
        self.clahe_clip_limit = float(getattr(self.cfg, "clahe_clip_limit", 3.0))
        self.clahe_grid_size = int(getattr(self.cfg, "clahe_grid_size", 8))
        self.blur_threshold = float(getattr(self.cfg, "blur_threshold", 55.0))
        self.min_quality_score = float(getattr(self.cfg, "min_quality_score", 0.2))
        self.adaptive_block_size = int(getattr(self.cfg, "adaptive_block_size", 31))
        if self.adaptive_block_size % 2 == 0:
            self.adaptive_block_size += 1
        self.adaptive_c = int(getattr(self.cfg, "adaptive_c", 11))
        self.morphology_kernel_size = max(1, int(getattr(self.cfg, "morphology_kernel_size", 3)))
        self.denoise_strength = int(getattr(self.cfg, "denoise_strength", 7))
        self.aggregation_margin = float(getattr(self.cfg, "aggregation_margin", 0.12))
        self.min_support_count = max(1, int(getattr(self.cfg, "min_support_count", 2)))
        self.stable_lock_threshold = float(getattr(self.cfg, "stable_lock_threshold", 0.72))
        self.replacement_threshold = float(getattr(self.cfg, "replacement_threshold", 1.25))
        self.decay_per_frame = float(getattr(self.cfg, "decay_per_frame", 0.92))
        self.enable = bool(getattr(self.cfg, "enable", True))
        self.reader = None

        self.track_observations: Dict[int, deque[TrackObservation]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self.track_numbers: Dict[int, str] = {}
        self.track_confidences: Dict[int, float] = {}
        self.track_states: Dict[int, str] = defaultdict(lambda: "unknown")
        self.track_quality: Dict[int, float] = {}
        self.track_last_signature: Dict[int, float] = {}
        self.track_last_bbox_area: Dict[int, float] = {}

        if self.enable:
            self._initialize_reader()

    def _initialize_reader(self) -> None:
        if self.backend == "crnn":
            logger.info("[OCR] CRNN backend requested but not implemented; falling back to EasyOCR")
            self.backend = "easyocr"

        if easyocr is None:
            logger.warning("[OCR] EasyOCR not available; jersey OCR disabled")
            self.enable = False
            return

        use_gpu = bool(getattr(self.cfg, "use_gpu", True))
        gpu_enabled = bool(use_gpu and torch is not None and torch.cuda.is_available())
        model_dir = (BACKEND_ROOT / "model_cache" / "easyocr").resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.reader = easyocr.Reader(
                ["en"],
                gpu=gpu_enabled,
                verbose=False,
                model_storage_directory=str(model_dir),
                user_network_directory=str(model_dir),
            )
            logger.info("[OCR] Jersey OCR initialized with backend=%s gpu=%s", self.backend, gpu_enabled)
        except Exception as exc:  # noqa: BLE001
            self.reader = None
            self.enable = False
            logger.warning("[OCR] EasyOCR initialization failed; disabling jersey OCR: %s", exc)

    def _quality_metrics(self, image: np.ndarray) -> tuple[float, float, float, float, float]:
        if image is None or image.size == 0 or cv2 is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blur_score = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        contrast_score = float(np.std(gray)) / 64.0
        edges = cv2.Canny(gray, 60, 180)
        edge_score = float(np.count_nonzero(edges)) / max(edges.size, 1)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        occupancy_score = float(np.count_nonzero(binary)) / max(binary.size, 1)
        text_scale_score = min(1.0, gray.shape[0] / 96.0) * (0.6 + min(blur_score / max(self.blur_threshold, 1.0), 1.0) * 0.4)

        blur_component = min(1.0, blur_score / max(self.blur_threshold, 1.0))
        contrast_component = min(1.0, contrast_score)
        edge_component = min(1.0, edge_score * 4.0)
        occupancy_component = 1.0 - min(abs(occupancy_score - 0.45), 0.45) / 0.45
        quality_score = (
            0.28 * blur_component
            + 0.22 * contrast_component
            + 0.24 * edge_component
            + 0.14 * occupancy_component
            + 0.12 * text_scale_score
        )
        return (quality_score, blur_score, contrast_score, edge_score, occupancy_score)

    def crop_jersey_candidates(
        self, frame: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> list[CropCandidate]:
        if frame is None or frame.size == 0 or cv2 is None:
            return []

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        if x2 - x1 < self.min_crop_width or y2 - y1 < self.min_crop_height:
            return []

        box_h = y2 - y1
        box_w = x2 - x1
        top_h = max(self.min_crop_height, int(box_h * self.top_region_ratio))
        expanded_h = max(top_h, int(box_h * self.expanded_region_ratio))
        inset = int(box_w * self.horizontal_inset_ratio)
        expand = int(box_w * self.horizontal_expand_ratio)

        specs = [
            ("upper_torso", x1, y1, x2, min(y2, y1 + top_h)),
            (
                "center_strip",
                min(max(0, x1 + inset), width - 1),
                y1,
                max(min(width, x2 - inset), min(max(0, x1 + inset), width - 1) + self.min_crop_width),
                min(y2, y1 + top_h),
            ),
            (
                "expanded_chest",
                max(0, x1 - expand),
                y1,
                min(width, x2 + expand),
                min(y2, y1 + expanded_h),
            ),
        ]

        candidates: list[CropCandidate] = []
        for name, cx1, cy1, cx2, cy2 in specs:
            if cx2 - cx1 < self.min_crop_width or cy2 - cy1 < self.min_crop_height:
                continue
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            quality_score, blur_score, contrast_score, edge_score, occupancy_score = self._quality_metrics(crop)
            if quality_score < self.min_quality_score * 0.6:
                continue
            candidates.append(
                CropCandidate(
                    image=crop.copy(),
                    name=name,
                    quality_score=quality_score,
                    blur_score=blur_score,
                    contrast_score=contrast_score,
                    edge_score=edge_score,
                    occupancy_score=occupancy_score,
                )
            )

        candidates.sort(key=lambda candidate: candidate.quality_score, reverse=True)
        return candidates[: self.max_crop_candidates]

    def crop_jersey_region(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        candidates = self.crop_jersey_candidates(frame, bbox)
        return candidates[0].image if candidates else None

    def _resize(self, image: np.ndarray, scale: float, interpolation) -> np.ndarray:
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)

    def preprocess(self, image: np.ndarray) -> Optional[np.ndarray]:
        variants = self.build_ocr_variants(image, crop_name="single")
        return variants[0].image if variants else None

    def build_ocr_variants(self, image: np.ndarray, crop_name: str = "crop") -> list[ProcessedVariant]:
        if image is None or image.size == 0 or cv2 is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_grid_size, self.clahe_grid_size),
        )
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, self.denoise_strength, 7, 21)
        resized = self._resize(denoised, self.resize_scale, cv2.INTER_LANCZOS4)

        gaussian = cv2.GaussianBlur(resized, (0, 0), sigmaX=1.2)
        unsharp = cv2.addWeighted(resized, 1.8, gaussian, -0.8, 0)
        laplacian = cv2.Laplacian(resized, cv2.CV_16S, ksize=3)
        laplacian = cv2.convertScaleAbs(laplacian)
        lap_sharp = cv2.addWeighted(resized, 1.0, laplacian, 0.35, 0)
        bilateral = cv2.bilateralFilter(resized, 9, 35, 35)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morphology_kernel_size, self.morphology_kernel_size),
        )
        variants_raw = []
        variants_raw.append(("clahe_gray", resized))
        variants_raw.append(("unsharp_gray", unsharp))
        variants_raw.append(("laplacian_gray", lap_sharp))
        variants_raw.append(("bilateral_gray", bilateral))

        if self.use_thresholding:
            adaptive = cv2.adaptiveThreshold(
                unsharp,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.adaptive_block_size,
                self.adaptive_c,
            )
            adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

            _, otsu = cv2.threshold(lap_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)

            variants_raw.append(("adaptive_bin", adaptive))
            variants_raw.append(("otsu_bin", otsu))

        variants: list[ProcessedVariant] = []
        for name, variant_img in variants_raw[: self.max_variants_per_crop]:
            quality_score, blur_score, _, _, _ = self._quality_metrics(variant_img)
            variants.append(
                ProcessedVariant(
                    image=variant_img,
                    name=name,
                    crop_name=crop_name,
                    quality_score=quality_score,
                    crop_quality_score=quality_score if crop_name == "single" else 0.0,
                    blur_score=blur_score,
                )
            )
        return variants

    def _normalize_candidate(self, text: str, confidence: float) -> OCRResult:
        if confidence < self.confidence_threshold:
            return OCRResult(number=None, confidence=0.0)

        cleaned = text.strip()
        if not _NUMBER_PATTERN.fullmatch(cleaned):
            return OCRResult(number=None, confidence=0.0)

        try:
            numeric_value = int(cleaned)
        except ValueError:
            return OCRResult(number=None, confidence=0.0)

        if not 0 <= numeric_value <= 99:
            return OCRResult(number=None, confidence=0.0)

        return OCRResult(number=f"{numeric_value:d}", confidence=float(confidence))

    def _run_easyocr(self, image: np.ndarray) -> list[OCRResult]:
        if self.reader is None:
            return []
        try:
            raw_results = self.reader.readtext(image, detail=1, paragraph=False, allowlist="0123456789")
        except Exception as exc:  # noqa: BLE001
            logger.debug("[OCR] EasyOCR failed: %s", exc)
            return []

        results: list[OCRResult] = []
        for _, text, confidence in raw_results:
            candidate = self._normalize_candidate(str(text), float(confidence))
            if candidate.number is not None:
                results.append(candidate)
        return results

    def extract_number(self, image: np.ndarray) -> OCRResult:
        variants = self.build_ocr_variants(image, crop_name="single")
        return self.aggregate_variant_results(variants)

    def aggregate_variant_results(self, variants: list[ProcessedVariant]) -> OCRResult:
        if not self.enable or self.reader is None or not variants:
            return OCRResult(number=None, confidence=0.0)

        votes: Dict[str, dict[str, object]] = {}
        for variant in variants:
            variant_results = self._run_easyocr(variant.image)
            for candidate in variant_results:
                weight = (
                    candidate.confidence * 0.55
                    + min(1.0, variant.quality_score) * 0.25
                    + min(1.0, variant.crop_quality_score or variant.quality_score) * 0.20
                )
                entry = votes.setdefault(
                    candidate.number,
                    {"score": 0.0, "confidence_sum": 0.0, "sources": [], "support_count": 0, "quality": 0.0},
                )
                entry["score"] += float(weight)
                entry["confidence_sum"] += candidate.confidence
                entry["support_count"] += 1
                entry["quality"] = max(float(entry["quality"]), float(variant.quality_score))
                sources = entry["sources"]
                if isinstance(sources, list):
                    sources.append(f"{variant.crop_name}:{variant.name}")

        if not votes:
            return OCRResult(number=None, confidence=0.0)

        ranked = sorted(votes.items(), key=lambda item: (float(item[1]["score"]), int(item[1]["support_count"])), reverse=True)
        best_number, best_meta = ranked[0]
        best_score = float(best_meta["score"])
        second_score = float(ranked[1][1]["score"]) if len(ranked) > 1 else 0.0
        support_count = int(best_meta["support_count"])
        aggregated_confidence = min(1.0, float(best_meta["confidence_sum"]) / max(support_count, 1))

        if support_count < self.min_support_count and best_score < self.stable_lock_threshold:
            return OCRResult(number=None, confidence=aggregated_confidence)
        if second_score > 0.0 and (best_score - second_score) < self.aggregation_margin:
            return OCRResult(number=None, confidence=aggregated_confidence)

        sources = tuple(best_meta["sources"]) if isinstance(best_meta["sources"], list) else ()
        return OCRResult(
            number=best_number,
            confidence=aggregated_confidence,
            support_count=support_count,
            variant_sources=sources,
            quality_score=float(best_meta["quality"]),
        )

    def update_track(self, track_id: int, result: OCRResult, frame_idx: int) -> tuple[Optional[str], float, str]:
        if result.number is not None:
            self.track_observations[track_id].append(
                TrackObservation(
                    number=result.number,
                    confidence=result.confidence,
                    frame_idx=frame_idx,
                    quality_score=result.quality_score,
                    support_count=result.support_count,
                )
            )

        observations = list(self.track_observations[track_id])
        if not observations:
            return (self.track_numbers.get(track_id), self.track_confidences.get(track_id, 0.0), self.track_states[track_id])

        now = frame_idx
        scores: Dict[str, float] = defaultdict(float)
        confidences: Dict[str, list[float]] = defaultdict(list)
        for obs in observations:
            age = max(0, now - obs.frame_idx)
            decay = self.decay_per_frame ** age
            weight = decay * (0.55 * obs.confidence + 0.30 * obs.quality_score + 0.15 * min(1.0, obs.support_count / 3.0))
            scores[obs.number] += weight
            confidences[obs.number].append(obs.confidence)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_number, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = float(np.mean(confidences[best_number])) if confidences[best_number] else 0.0
        existing = self.track_numbers.get(track_id)
        existing_score = scores.get(existing, 0.0) if existing else 0.0

        state = "candidate"
        if best_score >= self.stable_lock_threshold and len(confidences[best_number]) >= self.min_support_count:
            state = "stable"

        if existing and existing != best_number and self.track_states.get(track_id) == "stable":
            if best_score < existing_score * self.replacement_threshold:
                best_number = existing
                confidence = self.track_confidences.get(track_id, confidence)
                state = "stable"

        self.track_numbers[track_id] = best_number
        self.track_confidences[track_id] = confidence
        self.track_states[track_id] = state
        self.track_quality[track_id] = result.quality_score
        return (best_number, confidence, state)

    def get_track_number(self, track_id: int) -> Optional[str]:
        return self.track_numbers.get(track_id)

    def _crop_signature(self, image: np.ndarray) -> float:
        if image is None or image.size == 0:
            return 0.0
        resized = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
        return float(np.mean(resized))

    def process_tracks(self, frame: np.ndarray, tracks: Dict[int, object], frame_idx: int) -> Dict[int, Optional[str]]:
        if not self.enable:
            return {}

        results: Dict[int, Optional[str]] = {}
        for track_id, track in tracks.items():
            if getattr(track, "is_ball", False) or getattr(track, "is_referee", False):
                continue

            bbox = getattr(track, "bbox", None)
            if bbox is None:
                continue

            current_area = float(max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
            existing_number = self.track_numbers.get(track_id)
            existing_state = self.track_states.get(track_id, "unknown")
            area_changed = False
            if track_id in self.track_last_bbox_area and self.track_last_bbox_area[track_id] > 0:
                ratio = current_area / self.track_last_bbox_area[track_id]
                area_changed = ratio > 1.25 or ratio < 0.8

            should_refresh = (
                existing_number is None
                or existing_state != "stable"
                or getattr(track, "frames_tracked", 0) <= 2
                or area_changed
                or frame_idx % self.update_interval == 0
            )

            candidates = self.crop_jersey_candidates(frame, bbox) if should_refresh else []
            if candidates:
                signature = self._crop_signature(candidates[0].image)
                if abs(signature - self.track_last_signature.get(track_id, -999.0)) < 0.5 and existing_state == "stable":
                    should_refresh = False
                else:
                    self.track_last_signature[track_id] = signature
                    self.track_last_bbox_area[track_id] = current_area

            if not should_refresh:
                track.jersey_number = existing_number
                track.jersey_confidence = self.track_confidences.get(track_id, 0.0)
                track.jersey_stability = existing_state
                results[track_id] = existing_number
                continue

            variants: list[ProcessedVariant] = []
            for candidate in candidates:
                candidate_variants = self.build_ocr_variants(candidate.image, crop_name=candidate.name)
                for variant in candidate_variants:
                    variant.crop_quality_score = candidate.quality_score
                    variants.append(variant)

            ocr_result = self.aggregate_variant_results(variants)
            stable_number, stable_confidence, state = self.update_track(track_id, ocr_result, frame_idx)

            track.jersey_number = stable_number
            track.jersey_confidence = stable_confidence
            track.jersey_stability = state
            results[track_id] = stable_number

        return results
