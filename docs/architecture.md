# Architecture

Two interfaces call the same controller: `perch-detector` for stereo cameras and
`perch-video-seg` for paired videos. Root Python launchers expose the same
interfaces when running directly from a source checkout.

| Module in src/perch | Responsibility |
|---|---|
| cli.py | Argument parsing, validation and user-facing errors |
| settings.py | Asset paths and runtime defaults |
| app.py | Shared event loop, keyboard controls and resource cleanup |
| sources/video.py | Paired video frames, offsets, EOF and rewind |
| sources/picamera.py | Fresh sequence filtering and camera adapter |
| sources/camera.py | Original Picamera2 capture implementation |
| backend.py | Trained-model verification and NCNN segmentation |
| depth.py | Reference SGBM/WLS and mask rectification |
| measurement.py | Mask cleanup, robust depth and projected dimensions |
| tracking.py | Same-class mask IoU association |
| workflow.py | LIVE / CAPTURING / RESULT and burst median |
| guidance.py | Image-plane centering and angle hints |
| rendering.py | Both eyes, animated guidance and result display |
| pipeline.py | Shared per-frame processing and view refresh |
| snapshots.py | PNG, NumPy depth and JSON output |

LIVE rectifies both eyes and segments only the left eye (`right_mask=False`).
It computes no stereo depth. T uses the next three fresh pairs, without
pre-trigger samples. CAPTURING calls `StereoDepth.process` on each pair.
A median of at least three valid samples assesses one tracked target.
RESULT freezes the preview. R returns to LIVE and clears measurements.

`StereoDepth` and `depth_to_color` retain the original implementation from
`Calibration/video_depth_estimation.py`. `SegmentDepth` inherits `process()`
unchanged. Camera construction, capture and channel passthrough retain the
recorder's behavior. Camera dependencies are loaded only in camera mode.
Calibration and NCNN model files retain their original bytes.

Only the NCNN FP16 exports of the trained best-seg.pt are needed at runtime.
Model manifests preserve the source-checkpoint and component hashes. Training,
exporting, recording, benchmarking and COCO/geometry experiments are outside
the standalone application's scope.

## Installation and paths

This is a src-layout package with pyproject.toml metadata and two console
entrypoints. Use `pip install -e .` for this portable source distribution.
Large assets stay outside the Python package; transfer the entire folder.

Defaults resolve relative to the checkout, independently of the current working
directory. Explicit relative CLI paths resolve from the current working directory.
For a non-editable wheel installation, set PERCH_HOME to the directory containing
assets/, data/ and outputs/. Models, calibration and videos are not embedded in
the wheel. The original project outside perch_standalone remains unchanged.
