"""
Module 1: Video Preprocessing and Enhancement.

Handles:
- AI-powered video stabilization using optical flow
- Super-resolution using OpenCV DNN (ESPCN model)
- General quality enhancement (denoising, contrast)

Designed for single-camera, low-quality football footage.
"""
import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple

from app.config.settings import settings
from app.services.model_loader import download_hf_asset

logger = logging.getLogger(__name__)


class VideoStabilizer:
    """
    Video stabilization using optical flow-based motion estimation.
    Compensates for camera shake common in amateur football recordings.
    """

    def __init__(self, smoothing_radius: int = 30, border_crop: int = 20):
        self.smoothing_radius = smoothing_radius
        self.border_crop = border_crop
        self.transforms = []
        self.prev_gray = None

    def _moving_average(self, curve: np.ndarray, radius: int) -> np.ndarray:
        """Apply moving average smoothing to trajectory."""
        window_size = 2 * radius + 1
        kernel = np.ones(window_size) / window_size
        # Pad edges
        padded = np.pad(curve, (radius, radius), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='same')
        return smoothed[radius:-radius]

    def estimate_motion(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Estimate inter-frame motion transforms for all frames."""
        self.transforms = []

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # Detect good features to track
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=30,
                blockSize=3,
            )

            if prev_pts is not None and len(prev_pts) > 0:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, prev_pts, None
                )

                # Filter valid points
                valid = status.ravel() == 1
                if np.sum(valid) >= 3:
                    prev_valid = prev_pts[valid]
                    curr_valid = curr_pts[valid]

                    # Estimate affine transform
                    m, _ = cv2.estimateAffinePartial2D(prev_valid, curr_valid)
                    if m is not None:
                        dx = m[0, 2]
                        dy = m[1, 2]
                        da = np.arctan2(m[1, 0], m[0, 0])
                        self.transforms.append([dx, dy, da])
                    else:
                        self.transforms.append([0, 0, 0])
                else:
                    self.transforms.append([0, 0, 0])
            else:
                self.transforms.append([0, 0, 0])

            prev_gray = curr_gray

        self.transforms = np.array(self.transforms)
        return self.transforms

    def smooth_trajectory(self) -> np.ndarray:
        """Compute smoothed trajectory from motion transforms."""
        # Compute cumulative trajectory
        trajectory = np.cumsum(self.transforms, axis=0)

        # Smooth each component
        smoothed = np.zeros_like(trajectory)
        for i in range(3):
            smoothed[:, i] = self._moving_average(
                trajectory[:, i], self.smoothing_radius
            )

        # Compute difference to get smoothing corrections
        difference = smoothed - trajectory
        return difference

    def stabilize_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply stabilization to frames."""
        if len(frames) < 2:
            return frames

        logger.info(f"Stabilizing {len(frames)} frames...")
        self.estimate_motion(frames)
        smooth_corrections = self.smooth_trajectory()

        h, w = frames[0].shape[:2]
        stabilized = [frames[0]]  # First frame unchanged

        for i in range(len(self.transforms)):
            dx = self.transforms[i, 0] + smooth_corrections[i, 0]
            dy = self.transforms[i, 1] + smooth_corrections[i, 1]
            da = self.transforms[i, 2] + smooth_corrections[i, 2]

            # Build transformation matrix
            m = np.zeros((2, 3), np.float64)
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy

            stabilized_frame = cv2.warpAffine(frames[i + 1], m, (w, h))

            # Crop borders to remove black edges
            if self.border_crop > 0:
                c = self.border_crop
                stabilized_frame = stabilized_frame[c:h - c, c:w - c]
                stabilized_frame = cv2.resize(stabilized_frame, (w, h))

            stabilized.append(stabilized_frame)

        logger.info("Stabilization complete")
        return stabilized


class SuperResolutionEnhancer:
    """
    Lightweight super-resolution using OpenCV DNN module.
    Uses ESPCN for speed-quality tradeoff suitable for real-time.
    """

    SUPPORTED_MODELS = {
        "espcn": {
            "prefix": "ESPCN",
            "scales": [2, 3, 4],
        },
        "edsr": {
            "prefix": "EDSR",
            "scales": [2, 3, 4],
        },
        "fsrcnn": {
            "prefix": "FSRCNN",
            "scales": [2, 3, 4],
        },
        "lapsrn": {
            "prefix": "LapSRN",
            "scales": [2, 4, 8],
        },
    }

    def __init__(self, model_name: str = "espcn", scale: int = 2):
        self.model_name = model_name.lower()
        self.scale = scale
        self.sr = None
        self._initialized = False

    def initialize(self) -> bool:
        """Try to initialize the super-resolution model."""
        try:
            # Try different OpenCV super-resolution API versions
            sr = None
            try:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
            except AttributeError:
                try:
                    sr = cv2.dnn_superres.DnnSuperResImpl.create()
                except (AttributeError, Exception):
                    pass

            if sr is None:
                logger.info("OpenCV super-resolution not available, using bicubic fallback")
                return False

            self.sr = sr
            model_info = self.SUPPORTED_MODELS.get(self.model_name)
            if model_info is None:
                logger.warning(f"Unknown SR model: {self.model_name}, using bicubic upscale")
                return False

            model_file = f"{model_info['prefix']}_x{self.scale}.pb"

            import os
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models", model_file
            )

            if os.path.exists(model_path):
                self.sr.readModel(model_path)
                self.sr.setModel(self.model_name, self.scale)
                self._initialized = True
                logger.info(f"SR model loaded: {model_file}")
                return True

            hf_model_path = None
            if self.model_name == "espcn" and self.scale == 2:
                hf_model_path = download_hf_asset(
                    settings.HF_FOOTBALL_MODELS_REPO,
                    settings.HF_ESPCN_MODEL_FILE,
                )

            if hf_model_path:
                self.sr.readModel(hf_model_path)
                self.sr.setModel(self.model_name, self.scale)
                self._initialized = True
                logger.info("SR model loaded from HuggingFace: %s", hf_model_path)
                return True

            logger.info(
                f"SR model file not found locally or on HuggingFace for {model_file}. "
                f"Using bicubic interpolation as fallback."
            )
            return False

        except Exception as e:
            logger.info(f"SR initialization: {e}. Using bicubic fallback.")
            return False

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply super-resolution or bicubic upscaling."""
        if self._initialized and self.sr is not None:
            try:
                return self.sr.upsample(frame)
            except Exception:
                pass

        # Fallback: bicubic upscaling
        h, w = frame.shape[:2]
        return cv2.resize(
            frame, (w * self.scale, h * self.scale),
            interpolation=cv2.INTER_CUBIC
        )


class VideoPreprocessor:
    """
    Combined video preprocessing pipeline.
    Applies stabilization + super-resolution + quality enhancement.
    """

    def __init__(self, config):
        self.cfg = config.preprocessing
        self.stabilizer = VideoStabilizer(
            smoothing_radius=self.cfg.stabilization_smoothing_radius,
            border_crop=self.cfg.stabilization_border_crop,
        )
        self.sr_enhancer = SuperResolutionEnhancer(
            model_name=self.cfg.sr_model_name,
            scale=self.cfg.sr_scale_factor,
        )
        if self.cfg.enable_super_resolution:
            self.sr_enhancer.initialize()

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply per-frame quality enhancements."""
        # Denoise
        enhanced = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)

        # Adaptive contrast enhancement (CLAHE)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def _apply_super_resolution_preserving_size(self, frame: np.ndarray) -> np.ndarray:
        """Sharpen with SR while keeping downstream dimensions unchanged."""
        if not self.cfg.enable_super_resolution:
            return frame
        try:
            original_h, original_w = frame.shape[:2]
            enhanced = self.sr_enhancer.enhance(frame)
            if enhanced.shape[:2] != (original_h, original_w):
                enhanced = cv2.resize(
                    enhanced,
                    (original_w, original_h),
                    interpolation=cv2.INTER_AREA,
                )
            return enhanced
        except Exception:
            return frame

    def process_frames(
        self,
        frames: List[np.ndarray],
        enhance_quality: bool = True,
    ) -> List[np.ndarray]:
        """Full preprocessing pipeline for a batch of frames."""
        result = frames

        # Step 1: Stabilization
        if self.cfg.enable_stabilization and len(result) > 1:
            try:
                result = self.stabilizer.stabilize_frames(result)
            except Exception as e:
                logger.warning(f"Stabilization failed: {e}")

        # Step 2: Quality enhancement (per-frame)
        if enhance_quality:
            enhanced = []
            for frame in result:
                try:
                    quality_frame = self.enhance_frame(frame)
                    enhanced.append(self._apply_super_resolution_preserving_size(quality_frame))
                except Exception:
                    enhanced.append(frame)
            result = enhanced

        logger.info(f"Preprocessed {len(result)} frames")
        return result

    def process_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """Lightweight per-frame preprocessing (no stabilization)."""
        # Only CLAHE for speed
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return self._apply_super_resolution_preserving_size(enhanced)
