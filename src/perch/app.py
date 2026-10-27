"""Shared application controller for camera and video entrypoints."""
from __future__ import annotations

import json
import sys
import time
import cv2
from .backend import load_trained_model
from .cli import build_parser, validate_args
from .settings import MODEL_PATHS
from .sources.picamera import PicameraPairSource
from .sources.video import VideoPair
from .depth import SegmentDepth
from .tracking import MaskTracks
from .workflow import PerchWorkflow
from .pipeline import process_stereo_pair, refresh_view
from .snapshots import records, save_current

def run(argv=None, *, parser=None, model_factory=None, workflow_factory=PerchWorkflow,
         model_paths=None):
    parser = parser or build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    cv2.setNumThreads(args.cv_threads)
    model_paths = model_paths or MODEL_PATHS
    model_path = args.model or model_paths[args.profile]
    model = model_factory(model_path, args) if model_factory else load_trained_model(model_path, args)
    args.imgsz, args.device = max(model.imgsz), "cpu"
    estimator = SegmentDepth(args)
    backend = getattr(model, "backend", "ncnn")
    print(f"[MODEL] {backend} {model_path} | {model.imgsz} | threads={args.threads}")
    source = writer = last = None
    workflow = workflow_factory()
    tracks = MaskTracks(args.burst_frames, args.min_samples)
    processed, auto_triggered, show_depth = 0, False, True
    previous_completion = None
    window = getattr(args, "window_title", "Pi 5 Perch - NCNN - T to measure")
    started = time.perf_counter()
    try:
        source = (PicameraPairSource(args) if args.source == "camera" else
                  VideoPair(args.left, args.right, (args.left_start, args.right_start)))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                     source.fps, (estimator.size[0]*2, estimator.size[1]*2))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create output: {args.output}")
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1280, 720)
        print("[CONTROLS] T=burst, R=live/reset, SPACE=pause, G=guidance, D=depth, S=save, Q=quit")
        print("[DETECTION] Left eye only; right eye is preview/stereo input")
        print("[DEPTH] LIVE=off; T=reference SGBM/WLS for the next burst only", flush=True)
        while True:
            tick = time.perf_counter()
            if not workflow.paused:
                pair = source.read()
                if pair is None:
                    if workflow.state == "CAPTURING":
                        workflow.finish(last["detections"] if last else [])
                        if last:
                            refresh_view(last, workflow, args, source, show_depth)
                        if not args.headless:
                            continue
                    if args.loop and isinstance(source, VideoPair) and source.index == source.total:
                        source.rewind()
                        tracks = MaskTracks(args.burst_frames, args.min_samples)
                        workflow.reset()
                        previous_completion = None
                        continue
                    break
                index, left, right = pair
                captured = time.perf_counter()
                last = process_stereo_pair(model, estimator, tracks, workflow, args, index, left, right,
                                           right_mask=False)
                completed = time.perf_counter()
                # Frame-completion intervals include the preceding capture/render/wait,
                # instead of reporting only the inner inference pipeline rate.
                last["fps"] = 1 / max(completed - (previous_completion or tick), 1e-6)
                previous_completion = completed
                processed += 1
                limit = args.max_frames is not None and processed >= args.max_frames
                if workflow.state == "CAPTURING" and (limit or (isinstance(source, VideoPair) and source.index == source.total)):
                    workflow.finish(last["detections"])
                refresh_view(last, workflow, args, source, show_depth,
                             render=not args.headless or writer is not None)
                render_finished = time.perf_counter()
                last["summary"].update(source=args.source, backend=backend, model=str(model_path),
                                       segmentation_eye="left",
                                       camera_sequence=getattr(source, "sequence", None),
                                       retrieval_gap_ms=getattr(source, "last_gap_ms", None))
                last["summary"]["timing_ms"] = dict(capture=(captured-tick)*1000,
                    pipeline=(completed-captured)*1000, render=(render_finished-completed)*1000)
                last["summary"].update(getattr(model, "runtime_metadata", {}))
                if writer is not None:
                    writer.write(last["view"])
                if processed == 1 or processed % 30 == 0:
                    print(json.dumps(dict(pair=index, state=workflow.state, inference_ms=model.speed,
                                          timing_ms=last["summary"]["timing_ms"], fps=last["fps"],
                                          detections=records(last["detections"])), allow_nan=False), flush=True)
                if workflow.state == "RESULT":
                    print("[RESULT] " + json.dumps(workflow.result, allow_nan=False), flush=True)
                if limit or (args.headless and workflow.state == "RESULT"):
                    break
            if args.headless:
                key = -1
            else:
                cv2.imshow(window, last["view"])
                delay = max(1, round((1 / source.fps - (time.perf_counter()-tick))*1000))
                key = cv2.waitKey(30 if workflow.paused else delay) & 0xff
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if key in (ord("q"), 27):
                break
            if args.trigger_frame is not None and not auto_triggered and last["index"] >= args.trigger_frame and workflow.state == "LIVE":
                key, auto_triggered = ord("t"), True
            if key == ord("t"):
                if workflow.trigger(args.burst_frames, args.min_samples):
                    tracks.reset_measurements()
                    if isinstance(source, PicameraPairSource):
                        source.discard_buffered()
                    print(f"[BURST] Target #{workflow.target_id}: next {args.burst_frames} fresh pairs", flush=True)
            elif key == ord("r") or (key == ord(" ") and workflow.state == "RESULT"):
                workflow.reset()
                tracks.reset_measurements()
                if isinstance(source, PicameraPairSource):
                    source.discard_buffered()
            elif key == ord(" "):
                workflow.paused = not workflow.paused
            elif key == ord("d"):
                show_depth = not show_depth
            elif key == ord("g"):
                args.guidance = not args.guidance
            elif key == ord("b") and isinstance(source, VideoPair):
                source.rewind()
                tracks = MaskTracks(args.burst_frames, args.min_samples)
                workflow.reset()
            elif key == ord("s"):
                save_current(args.snapshot_dir, last)
            # Key-driven state changes need one redraw; otherwise reuse the single
            # view composed for this frame, including while paused.
            if not args.headless and key in (ord("t"), ord("r"), ord(" "), ord("d"), ord("g"), ord("b")):
                refresh_view(last, workflow, args, source, show_depth)
                previous_completion = None
        if last is None:
            raise RuntimeError("No stereo pairs received")
        if args.save_last:
            refresh_view(last, workflow, args, source, show_depth)
            save_current(args.snapshot_dir, last)
        print(f"[DONE] {processed} pairs; {time.perf_counter()-started:.1f}s; torch imported={'torch' in sys.modules}")
    finally:
        if source is not None:
            source.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
