"""Default paths and shared runtime settings."""
from __future__ import annotations

import os
from pathlib import Path

# PERCH_HOME supports installations outside an editable source checkout.
ROOT = Path(os.environ.get("PERCH_HOME", Path(__file__).resolve().parents[2])).expanduser().resolve()
ASSETS = ROOT / "assets"
VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_DIR = ROOT / "outputs"
CALIB_FILE = ASSETS / "calibration" / "stereo_calibration.npz"
MODEL_FILE = ASSETS / "models" / "perch_pi5_960_fp16_ncnn_model"
MODEL_PATHS = {"fast": ASSETS / "models" / "perch_pi5_640_fp16_ncnn_model", "detail": MODEL_FILE}
CHECKPOINT = ASSETS / "models" / "best-seg.pt"  # Optional provenance check; not needed for inference.
LEFT_CAM_NUM, RIGHT_CAM_NUM = 0, 1
CAPTURE_WIDTH, CAPTURE_HEIGHT, CAMERA_FPS = 1920, 1080, 30
DEPTH_DOWNSCALE, CALIB_UNIT = 0.5, "mm"
YOLO_CONF, N_BURST, MIN_VALID_FRAMES = 0.35, 3, 3
NCNN_THREADS, OPENCV_THREADS = 3, 1
SNAPSHOT_DIR = OUTPUT_DIR / "camera"
UNIT_TO_M = {"mm": 0.001, "cm": 0.01, "m": 1.0}
