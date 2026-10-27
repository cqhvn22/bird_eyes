"""Fresh stereo pair adapter using the original recorder connection sequence."""
from __future__ import annotations

import time
import cv2

class PicameraPairSource:
    """Latest fresh camera pair; never count one sequence twice in a burst.

    Use the same StereoCamera constructor/open/start/read/release calls as
    record_stereo_video.py. Preserve its raw frame arrays by default so the
    trained model sees the same channel order as the recorded videos.
    """
    total = "LIVE"

    def __init__(self, args, camera_factory=None):
        if camera_factory is None:
            from .camera import StereoCamera
            camera_factory = StereoCamera
        self.camera = camera_factory(left_camera_num=args.left_camera, right_camera_num=args.right_camera,
                                     capture_width=args.width, capture_height=args.height,
                                     framerate=args.camera_fps, flip_method=args.flip_method)
        self.size, self.fps = (args.width, args.height), args.camera_fps
        self.index, self.last_sequence = 0, None
        self.timeout, self.color = args.camera_timeout, args.camera_color
        self.max_gap, self.last_gap_ms = args.max_pair_gap_ms, None
        self.sequence, self.rejected = None, 0
        try:
            if not self.camera.open() or not self.camera.start():
                raise RuntimeError("Cannot start both Picamera2 cameras")
        except Exception:
            self.release()
            raise

    def discard_buffered(self):
        """Discard a pair already retrieved before T/reset, then wait for a new one."""
        ok, frame = self.camera.read(copy_frames=False)
        if ok and frame is not None:
            self.last_sequence = frame.sequence

    def read(self):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            ok, frame = self.camera.read(last_sequence=self.last_sequence, copy_frames=False)
            if not ok or frame is None or (self.last_sequence is not None and frame.sequence <= self.last_sequence):
                time.sleep(0.003)
                continue
            self.last_sequence = frame.sequence
            if self.max_gap > 0 and frame.delta_ms > self.max_gap:
                self.rejected += 1
                continue
            left, right = frame.left, frame.right
            if self.color == "rgb":
                left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
                right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
            self.last_gap_ms, self.sequence = float(frame.delta_ms), int(frame.sequence)
            index = self.index
            self.index += 1
            return index, left, right
        raise TimeoutError(f"No fresh accepted camera pair for {self.timeout:g}s; "
                           f"check cameras/gap limit (rejected={self.rejected})")

    def release(self):
        self.camera.release()
