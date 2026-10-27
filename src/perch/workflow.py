"""LIVE, three-pair measurement burst and frozen result state machine."""
from __future__ import annotations

import numpy as np
from .measurement import assess

def select_target(detections, preferred_id=None):
    for det in detections:
        if det["track_id"] == preferred_id:
            return det
    return max(detections, key=lambda d: np.count_nonzero(d["mask"]) * d["confidence"],
               default=None)


class PerchBurst:
    """T-triggered N-frame median for one target, with no pre-trigger samples."""

    def __init__(self, target_id=None, requested=3, min_samples=3):
        self.target_id, self.requested, self.min_samples = target_id, requested, min_samples
        self.n_frames, self.samples, self.last_seen = 0, [], False

    @property
    def done(self):
        return self.n_frames >= self.requested

    def add(self, detections):
        if self.done:
            return
        if self.target_id is None:
            first = select_target(detections)
            self.target_id = first["track_id"] if first else None
        target = next((d for d in detections if d["track_id"] == self.target_id), None)
        self.last_seen = target is not None
        if target is not None and target["measurement"].values() is not None:
            self.samples.append(target["measurement"].values())
        self.n_frames += 1

    def result(self):
        values = tuple(float(v) for v in np.median(self.samples, axis=0)) if self.samples else None
        verdict, reason = assess(values)
        if not self.done:
            verdict, reason = None, f"Incomplete burst: {self.n_frames}/{self.requested} pairs"
        elif len(self.samples) < self.min_samples:
            verdict, reason = None, f"Insufficient depth: {len(self.samples)}/{self.min_samples} valid pairs"
        elif not self.last_seen:
            verdict, reason = None, "Target lost at end of burst; align and trigger again"
        return dict(target_id=self.target_id, n_frames=self.n_frames, requested=self.requested,
                    valid_frames=len(self.samples), values=values, verdict=verdict, reason=reason)


class PerchWorkflow:
    """LIVE -> T -> CAPTURING -> RESULT (frozen); R returns to LIVE."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state, self.paused, self.target_id = "LIVE", False, None
        self.burst, self.result = None, None

    def trigger(self, n_frames=3, min_samples=3):
        if self.state == "CAPTURING":
            return False
        self.burst = PerchBurst(self.target_id, n_frames, min_samples)
        self.result = None
        self.state, self.paused = "CAPTURING", False
        return True

    def observe(self, detections):
        for det in detections:
            det.update(stable=None, verdict=None,
                       reason="Press T to measure" if self.state == "LIVE" else "Capturing burst")
        if self.state == "CAPTURING":
            self.burst.add(detections)
            self.target_id = self.burst.target_id
            if self.burst.done:
                self.finish(detections)

    def finish(self, detections):
        self.result = self.burst.result()
        self.state, self.paused = "RESULT", True
        for det in detections:
            if det["track_id"] == self.result["target_id"]:
                det.update(stable=self.result["values"], verdict=self.result["verdict"],
                           reason=self.result["reason"])
