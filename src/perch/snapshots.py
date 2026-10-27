"""JSON, masks, preview images and optional depth export."""
from __future__ import annotations

import json
import time
import cv2
import numpy as np

def records(detections):
    return [dict(track_id=d["track_id"], class_id=d["class_id"], name=d["name"],
                 confidence=d["confidence"], box=d["box"],
                 distance_m=d["measurement"].distance, length_m=d["measurement"].length,
                 width_m=d["measurement"].width,
                 valid_depth_fraction=d["measurement"].valid_fraction,
                 stable=d["stable"], samples=d["samples"], verdict=d["verdict"], reason=d["reason"],
                 **{key: d[key] for key in ("source", "score_kind", "shape", "shape_rectified", "cylinder_confirmed") if key in d})
            for d in detections]


def save_snapshot(directory, annotated, depth_color, depth, detections, index,
                  right=None, view=None, summary=None):
    directory.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000000:09d}_f{index:06d}"
    for suffix, img in (("seg", annotated), ("right", right), ("view", view), ("depth", depth_color)):
        if img is None:
            continue
        if not cv2.imwrite(str(directory / f"{tag}_{suffix}.png"), img):
            raise RuntimeError(f"Cannot save snapshot in {directory}")
    if depth is not None:
        np.save(directory / f"{tag}_depth_m.npy", depth)
    for det in detections:
        if not cv2.imwrite(str(directory / f"{tag}_mask_{det['track_id']}.png"), det["mask"] * 255):
            raise RuntimeError(f"Cannot save instance mask in {directory}")
    (directory / f"{tag}.json").write_text(
        json.dumps(dict(pair_index=index, workflow=summary, detections=records(detections)), indent=2, allow_nan=False),
        encoding="utf-8")
    print(f"[SAVE] {directory / tag}")


def save_current(directory, frame):
    save_snapshot(directory, frame["left_view"], frame["depth_view"], frame["depth"],
                  frame["detections"], frame["index"], right=frame["right_view"],
                  view=frame["view"], summary=frame["summary"])
