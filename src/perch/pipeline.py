"""Shared left-eye segmentation and T-triggered stereo processing."""
from __future__ import annotations

import time
import cv2
import numpy as np
from .settings import UNIT_TO_M
from .depth import depth_to_color
from .measurement import Measurement, clean_mask, measure_mask
from .workflow import select_target
from .guidance import alignment_hint
from .rendering import compose_view

def infer(model, image, depth, focal, args, mask_projector=None):
    prediction = model.predict(
        source=image, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        retina_masks=True, device=args.device, classes=args.classes,
        verbose=False)[0]
    boxes, masks = prediction.boxes, prediction.masks
    if boxes is None or len(boxes) == 0:
        return []
    if masks is None or len(masks.data) != len(boxes):
        raise RuntimeError("Detections have no matching segmentation masks; use a segmentation model.")
    # retina_masks guarantees native coordinates; refuse a silently misaligned mask.
    raw_masks = masks.data.cpu().numpy()
    if tuple(raw_masks.shape[1:]) != image.shape[:2]:
        raise RuntimeError("Mask/image size mismatch; update Ultralytics (retina_masks required).")
    detections = []
    candidate_info = getattr(prediction, "candidate_info", None)
    if candidate_info is not None and len(candidate_info) != len(raw_masks):
        raise RuntimeError("Candidate provenance and mask counts differ")
    for mask_index, (box, raw) in enumerate(zip(boxes.cpu().data.numpy(), raw_masks)):
        if mask_projector is not None:
            raw = mask_projector(raw)
        if depth is not None and raw.shape != depth.shape:
            raise RuntimeError("Projected mask and depth coordinates do not match.")
        mask = clean_mask(raw, args.min_mask_area)
        if mask is None:
            continue
        filter_candidate = getattr(model, "filter_projected_candidate", None)
        filter_mask = getattr(model, "filter_projected_mask", None)
        if filter_candidate:
            projected_info = filter_candidate(mask, candidate=candidate_info[mask_index] if candidate_info else None,
                                               class_id=int(box[5]))
        else:
            projected_info = filter_mask(mask) if filter_mask else {}
        if projected_info is None:
            continue
        class_id = int(box[5])
        if depth is None:
            measurement = Measurement(reason="Press T to measure")
            oriented = cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(mask)))
        else:
            measurement, _ = measure_mask(
                mask, depth, focal, args.min_depth_pixels, args.min_depth_fraction)
            oriented = cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(mask)))
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
        detections.append(dict(
            mask=mask, class_id=class_id, name=model.names[class_id], confidence=float(box[4]),
            box=(x, y, x + w, y + h), measurement=measurement, oriented=oriented))
        if candidate_info is not None:
            detections[-1].update(candidate_info[mask_index])
        detections[-1].update(projected_info)
    return detections


def process_stereo_pair(model, estimator, tracks, workflow, args, index, left, right,
                        right_mask=True):
    """Shared pipeline: compute reference stereo depth only during a T burst."""
    tick = time.perf_counter()
    if workflow.state == "CAPTURING":
        left_rect, _, depth_raw = estimator.process(left, right)
        reference_color = depth_to_color(depth_raw, args.max_depth, args.unit)
        depth = depth_raw * UNIT_TO_M[args.unit]
    else:
        left_rect = estimator.rectify_left(left)
        depth = reference_color = None
    right_rect = estimator.rectify_right(right)
    image = left if args.seg_space == "original" else left_rect
    projector = estimator.rectify_mask if args.seg_space == "original" else None
    detections = infer(model, image, depth, float(estimator.Q[2, 3]), args, projector)
    tracks.update(detections, index)
    right_detections = []
    if right_mask:
        image_r = right if args.seg_space == "original" else right_rect
        project_r = (lambda mask: estimator.rectify_mask(mask, "right")) if args.seg_space == "original" else None
        right_detections = infer(model, image_r, None, float(estimator.Q[2, 3]), args, project_r)
        for i, det in enumerate(right_detections):
            det.update(track_id=f"R{i+1}", stable=None, samples=0, verdict=None,
                       reason="Right preview; measurement uses left depth")
    if workflow.state == "LIVE":
        target = select_target(detections, workflow.target_id)
        workflow.target_id = target["track_id"] if target else None
    workflow.observe(detections)
    target = next((d for d in detections if d["track_id"] == workflow.target_id), None)
    guide = alignment_hint(target["mask"] if target else None, args.align_angle, args.align_center)
    return dict(left=left_rect, right=right_rect, depth_color=reference_color, depth=depth,
                detections=detections, right_detections=right_detections, guide=guide,
                index=index, fps=1 / max(time.perf_counter()-tick, 1e-6), animation_time=tick)


def refresh_view(frame, workflow, args, source, show_depth, render=True):
    if render:
        view, left_view, right_view, depth_view = compose_view(
            frame["left"], frame["right"], frame["depth_color"], frame["detections"],
            frame["right_detections"], frame["guide"], workflow.state, frame["index"],
            source.total, frame["fps"], workflow.target_id, workflow.burst, workflow.result,
            show_depth, workflow.paused, getattr(args, 'guidance', True), frame.get('animation_time'))
        frame.update(view=view, left_view=left_view, right_view=right_view, depth_view=depth_view)
    frame["summary"] = dict(frame.get("summary", {}), state=workflow.state, target_id=workflow.target_id,
                            alignment=frame["guide"], burst=workflow.result,
                            left_frame=args.left_start + frame["index"],
                            right_frame=args.right_start + frame["index"],
                            depth_computed=frame["depth"] is not None)
