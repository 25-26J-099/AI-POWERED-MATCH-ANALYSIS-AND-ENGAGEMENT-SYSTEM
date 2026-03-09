"""
Video I/O utilities for reading and writing video files.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List, Generator
import logging

logger = logging.getLogger(__name__)


class VideoReader:
    """Efficient video reader with frame skipping and resizing support."""

    def __init__(
        self,
        video_path: str,
        frame_skip: int = 1,
        max_width: int = 0,
        max_height: int = 0,
    ):
        self.video_path = video_path
        self.frame_skip = max(1, frame_skip)
        self.max_width = max_width
        self.max_height = max_height

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        # Calculate output dimensions
        self.width, self.height = self._calc_output_dims()

        logger.info(
            f"Video: {video_path} | {self.original_width}x{self.original_height} "
            f"-> {self.width}x{self.height} | {self.fps:.1f} FPS | "
            f"{self.total_frames} frames | {self.duration:.1f}s"
        )

    def _calc_output_dims(self) -> Tuple[int, int]:
        w, h = self.original_width, self.original_height
        if self.max_width > 0 and w > self.max_width:
            scale = self.max_width / w
            w = self.max_width
            h = int(h * scale)
        if self.max_height > 0 and h > self.max_height:
            scale = self.max_height / h
            h = self.max_height
            w = int(w * scale)
        return w, h

    def read_frames(self, max_frames: int = 0) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, frame) tuples."""
        frame_idx = 0
        count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % self.frame_skip == 0:
                if self.width != self.original_width or self.height != self.original_height:
                    frame = cv2.resize(frame, (self.width, self.height))
                yield frame_idx, frame
                count += 1
                if max_frames > 0 and count >= max_frames:
                    break

            frame_idx += 1

    def read_all_frames(self, max_frames: int = 0) -> List[np.ndarray]:
        """Read all frames into memory."""
        frames = []
        for _, frame in self.read_frames(max_frames):
            frames.append(frame)
        return frames

    def read_batch(self, batch_size: int = 16) -> Generator[List[Tuple[int, np.ndarray]], None, None]:
        """Yield batches of (frame_index, frame) tuples."""
        batch = []
        for item in self.read_frames():
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        self.cap.release()

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


class VideoWriter:
    """Video writer with codec auto-selection."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 30.0,
        codec: str = "mp4v",
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {output_path}")
        self.frame_count = 0
        logger.info(f"VideoWriter: {output_path} | {width}x{height} @ {fps:.1f} FPS")

    def write(self, frame: np.ndarray):
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)
        self.frame_count += 1

    def release(self):
        self.writer.release()
        logger.info(f"VideoWriter: wrote {self.frame_count} frames to {self.output_path}")

    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.release()