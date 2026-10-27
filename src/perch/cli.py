"""Camera/video command-line interfaces and argument validation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import cv2
from .settings import (CALIB_FILE, LEFT_CAM_NUM, RIGHT_CAM_NUM, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAMERA_FPS, CALIB_UNIT, DEPTH_DOWNSCALE, NCNN_THREADS, OPENCV_THREADS, YOLO_CONF, N_BURST, MIN_VALID_FRAMES, SNAPSHOT_DIR, VIDEO_DIR, OUTPUT_DIR)

def build_parser(*, default_source="camera", source_choices=("camera", "video")):
    p = argparse.ArgumentParser(description="Trained NCNN stereo perch detector", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", choices=source_choices, default=default_source)
    p.add_argument("--profile", choices=("fast", "detail"), default="detail",
                   help="detail matches perch_video_seg.py imgsz=960; fast uses the trained 640 export")
    p.add_argument("--model", type=Path, help="Custom NCNN model directory; overrides profile")
    p.add_argument("--calib", type=Path, default=CALIB_FILE)
    p.add_argument("--left-camera", "--left-sensor", dest="left_camera", type=int, default=LEFT_CAM_NUM)
    p.add_argument("--right-camera", "--right-sensor", dest="right_camera", type=int, default=RIGHT_CAM_NUM)
    p.add_argument("--width", type=int, default=CAPTURE_WIDTH)
    p.add_argument("--height", type=int, default=CAPTURE_HEIGHT)
    p.add_argument("--camera-fps", "--fps", dest="camera_fps", type=int, default=CAMERA_FPS)
    p.add_argument("--flip-method", type=int, choices=range(8), default=0)
    p.add_argument("--camera-timeout", type=float, default=3.0)
    p.add_argument("--camera-color", choices=("rgb", "bgr"), default="bgr",
                   help="bgr: preserve recorder's frame arrays (default); rgb: explicitly swap R/B")
    p.add_argument("--max-pair-gap-ms", type=float, default=0,
                   help="Optional software retrieval-gap rejection; 0 disables (not exposure timestamps)")
    p.add_argument("--threads", type=int, default=NCNN_THREADS)
    p.add_argument("--cv-threads", type=int, default=OPENCV_THREADS)
    p.add_argument("--fp32-arithmetic", action="store_true", help="Use FP32 arithmetic with FP16 stored weights")
    p.add_argument("--unit", choices=("mm", "cm", "m"), default=CALIB_UNIT)
    p.add_argument("--downscale", type=float, default=DEPTH_DOWNSCALE)
    p.add_argument("--conf", type=float, default=YOLO_CONF)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--classes", type=int, nargs="+")
    p.add_argument("--seg-space", choices=("original", "rectified"), default="original")
    p.add_argument("--burst-frames", type=int, default=N_BURST)
    p.add_argument("--min-samples", type=int, default=MIN_VALID_FRAMES)
    p.add_argument("--align-angle", type=float, default=5)
    p.add_argument("--align-center", type=float, default=0.12)
    p.add_argument("--no-guidance", dest="guidance", action="store_false", default=True,
                   help="Disable the animated 2D drone-view alignment overlay (G toggles it)")
    p.add_argument("--min-mask-area", type=int, default=16)
    p.add_argument("--min-depth-pixels", type=int, default=20)
    p.add_argument("--min-depth-fraction", type=float, default=0.15)
    p.add_argument("--max-depth", type=float, default=5)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--trigger-frame", type=int, help="Auto T after N processed pairs (zero based)")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    p.add_argument("--save-last", action="store_true")
    p.add_argument("--output", type=Path, help="Save 2x2 preview as MP4")
    p.add_argument("--left", type=Path, default=VIDEO_DIR / "l1.mp4", help="Video debug only")
    p.add_argument("--right", type=Path, default=VIDEO_DIR / "r1.mp4", help="Video debug only")
    p.add_argument("--left-start", type=int, default=0)
    p.add_argument("--right-start", type=int, default=0)
    p.add_argument("--loop", action="store_true", help="Restart both videos together at EOF (video source only)")
    return p


def validate_args(args, parser):
    if not args.calib.is_file():
        parser.error(f"Calibration missing: {args.calib}")
    if min(args.width, args.height, args.camera_fps, args.threads, args.cv_threads) < 1:
        parser.error("Resolution, FPS and threads must be positive")
    if args.left_camera == args.right_camera and args.source == "camera":
        parser.error("Left and right must use different camera indices")
    if not (0 < args.downscale <= 1 and 0 < args.conf <= 1 and 0 < args.iou <= 1):
        parser.error("downscale/conf/iou must be in (0, 1]")
    if not (0 < args.align_angle < 90 and 0 < args.align_center < .5):
        parser.error("Invalid alignment tolerances")
    if not 1 <= args.min_samples <= args.burst_frames:
        parser.error("Require 1 <= min-samples <= burst-frames")
    if min(args.min_mask_area, args.min_depth_pixels) < 1 or not 0 < args.min_depth_fraction <= 1:
        parser.error("Invalid mask/depth sample thresholds")
    if args.camera_timeout <= 0 or args.max_depth <= 0 or args.max_pair_gap_ms < 0:
        parser.error("Invalid camera timeout, depth maximum or pair-gap threshold")
    if min(args.left_start, args.right_start) < 0 or (args.trigger_frame is not None and args.trigger_frame < 0):
        parser.error("Frame indices must be nonnegative")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("max-frames must be positive")
    if args.headless and args.source == "camera" and args.max_frames is None and args.trigger_frame is None:
        parser.error("Headless camera mode needs max-frames or trigger-frame")
    if args.loop and args.source != "video":
        parser.error("loop requires the video source")
    if args.headless and args.loop and args.max_frames is None:
        parser.error("headless + loop needs max-frames")
    if args.output:
        if args.output.suffix.lower() != ".mp4":
            parser.error("output must be .mp4")
        if args.output.resolve() in {args.left.resolve(), args.right.resolve(), args.calib.resolve()}:
            parser.error("output cannot overwrite an input")


def build_video_parser():
    parser = build_parser(default_source="video", source_choices=("video",))
    parser.description = "Trained NCNN perch segmentation from two synchronized videos"
    parser.set_defaults(left=VIDEO_DIR / "left_20261026_172822.mp4",
                        right=VIDEO_DIR / "right_20261026_172822.mp4",
                        snapshot_dir=OUTPUT_DIR / "video",
                        window_title="Stereo Perch - trained NCNN videos - T to measure",
                        imgsz=960, device="cpu")

    class ProfileSize(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            namespace.imgsz = value
            namespace.profile = "fast" if value == 640 else "detail"

    parser.add_argument("--imgsz", type=int, choices=(640, 960), action=ProfileSize,
                        help="Compatibility alias: 640=fast, 960=detail")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser


def _execute(argv, parser):
    from .app import run
    try:
        run(argv, parser=parser)
        return 0
    except (RuntimeError, ValueError, KeyError, OSError, ImportError, cv2.error) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


def camera_main(argv=None):
    """Run live stereo cameras; segmentation is left-eye only."""
    return _execute(argv, build_parser(source_choices=("camera",)))


def video_main(argv=None):
    """Run two videos through the same application and numerical pipeline."""
    return _execute(argv, build_video_parser())
