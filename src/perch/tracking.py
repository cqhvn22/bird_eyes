"""Conservative same-class instance association across frames."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import cv2
import numpy as np
from .measurement import assess

@dataclass
class Track:
    identity: int
    class_id: int
    mask: np.ndarray
    last_frame: int
    history: deque = field(default_factory=lambda: deque(maxlen=3))
    bounds: tuple | None = None
    area: int = 0


class MaskTracks:
    """Conservative same-class mask IoU association; never pool distinct objects."""

    def __init__(self, history=7, min_samples=3):
        self.tracks: list[Track] = []
        self.next_id = 1
        self.history, self.min_samples = history, min_samples

    def reset_measurements(self):
        for track in self.tracks:
            track.history.clear()

    def update(self, detections, frame_index):
        self.tracks = [t for t in self.tracks if frame_index - t.last_frame <= 3]
        candidates = []
        geometry = [(cv2.boundingRect(det["mask"]), cv2.countNonZero(det["mask"])) for det in detections]
        for track in self.tracks:
            if track.bounds is None:
                track.bounds, track.area = cv2.boundingRect(track.mask), cv2.countNonZero(track.mask)
        for i, det in enumerate(detections):
            (x, y, w, h), area = geometry[i]
            for j, track in enumerate(self.tracks):
                if det["class_id"] != track.class_id:
                    continue
                tx, ty, tw, th = track.bounds
                x0, y0, x1, y1 = max(x, tx), max(y, ty), min(x+w, tx+tw), min(y+h, ty+th)
                if x1 <= x0 or y1 <= y0 or det["mask"].shape != track.mask.shape:
                    continue
                intersection = cv2.countNonZero(cv2.bitwise_and(det["mask"][y0:y1, x0:x1],
                                                              track.mask[y0:y1, x0:x1]))
                union = area + track.area - intersection
                iou = intersection / max(1, union)
                if iou >= 0.25:
                    candidates.append((iou, i, j))
        matched, used_tracks = {}, set()
        for _, i, j in sorted(candidates, reverse=True):
            if i not in matched and j not in used_tracks:
                matched[i] = self.tracks[j]
                used_tracks.add(j)
        for i, det in enumerate(detections):
            track = matched.get(i)
            if track is None:
                track = Track(self.next_id, det["class_id"], det["mask"], frame_index,
                              deque(maxlen=self.history))
                self.next_id += 1
                self.tracks.append(track)
            track.mask, track.last_frame = det["mask"], frame_index
            track.bounds, track.area = geometry[i]
            values = det["measurement"].values()
            track.history.append((frame_index, values))
            recent = [v for frame, v in track.history
                      if frame_index - frame < self.history and v is not None]
            # A previous measurement must never hide a failed current measurement.
            stable = (tuple(float(x) for x in np.median(recent, axis=0))
                      if values is not None and len(recent) >= self.min_samples else None)
            det.update(track_id=track.identity, stable=stable, samples=len(recent))
            det["verdict"], det["reason"] = assess(stable)
            if values is None:
                det["reason"] = det["measurement"].reason
