"""Paired, frame-indexed video input."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

class VideoPair:
    """Read matching frame indices; stop at the shorter stream, never loop one eye."""

    def __init__(self, left: Path, right: Path, starts: tuple[int, int]):
        self.left = cv2.VideoCapture(str(left))
        self.right = cv2.VideoCapture(str(right))
        self.starts = starts
        try:
            for cap, path in ((self.left, left), (self.right, right)):
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open video: {path}")
            rates = [cap.get(cv2.CAP_PROP_FPS) for cap in (self.left, self.right)]
            if not all(np.isfinite(f) and f > 0 for f in rates):
                raise RuntimeError("Video FPS is missing; re-encode both videos at a fixed FPS.")
            if abs(rates[0] - rates[1]) > max(rates) * 0.001:
                raise RuntimeError(f"Different FPS {rates}. Re-encode to the same constant FPS first.")
            self.fps = rates[0]
            counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                      for cap in (self.left, self.right)]
            if any(count <= start for count, start in zip(counts, starts)):
                raise RuntimeError("Start frame is outside the video or frame count is unavailable.")
            self.total = min(count - start for count, start in zip(counts, starts))
            sizes = [(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                     for cap in (self.left, self.right)]
            if sizes[0] != sizes[1]:
                raise RuntimeError(f"Left/right video sizes differ: {sizes}")
            self.size = sizes[0]
            self.rewind()
            print(f"[VIDEO] {self.size}, {self.fps:.3f} FPS, {self.total} paired frames")
        except Exception:
            self.release()
            raise

    def rewind(self):
        for cap, start in zip((self.left, self.right), self.starts):
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
                raise RuntimeError(f"Cannot seek video to frame {start}")
        self.index = 0

    def read(self):
        if self.index >= self.total:
            return None
        ok_l, left = self.left.read()
        ok_r, right = self.right.read()
        if not ok_l or not ok_r:
            return None
        index = self.index
        self.index += 1
        return index, left, right

    def release(self):
        self.left.release()
        self.right.release()
