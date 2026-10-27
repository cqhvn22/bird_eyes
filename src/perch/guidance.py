"""Image-plane alignment hints; no aircraft body control commands."""
from __future__ import annotations

import cv2
import numpy as np

def alignment_hint(mask, angle_tolerance=5.0, center_tolerance=0.12):
    """Image-plane roll/centering only; no depth, IMU or inferred 3D attitude."""
    if mask is None or not np.any(mask):
        return dict(angle_deg=None, offset=None, ready=False, axis=None,
                    clipped=False, angle_tolerance=angle_tolerance, center_tolerance=center_tolerance,
                    messages=["Find a perch visible in both eyes."])
    rect = cv2.minAreaRect(cv2.findNonZero(mask))
    corners = cv2.boxPoints(rect)
    edges = np.roll(corners, -1, axis=0) - corners
    vector = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    if vector[0] < 0:
        vector = -vector
    major, minor = sorted(rect[1], reverse=True)
    angle = float(np.degrees(np.arctan2(vector[1], vector[0]))) if major >= 1.5 * max(minor, 1) else None
    moments = cv2.moments(mask)
    center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    h, w = mask.shape
    dx, dy = float((center[0] - w / 2) / w), float((center[1] - h / 2) / h)
    clipped = bool(np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]))
    messages = []
    if angle is None:
        messages.append("Bar axis unclear: change viewpoint to see its length.")
    elif abs(angle) > angle_tolerance:
        direction = "CCW" if angle > 0 else "CW"
        messages.append(f"Level bar: image rotation {direction} by {abs(angle):.1f} deg.")
    else:
        messages.append(f"Roll aligned (within +/-{angle_tolerance:g} deg).")
    if abs(dx) > center_tolerance or abs(dy) > center_tolerance:
        directions = []
        if abs(dx) > center_tolerance:
            directions.append("RIGHT" if dx > 0 else "LEFT")
        if abs(dy) > center_tolerance:
            directions.append("DOWN" if dy > 0 else "UP")
        messages.append("Aim toward bar: " + " + ".join(directions) + ".")
    else:
        messages.append("Bar centered in view.")
    if clipped:
        messages.append("Move back / reframe: part of the bar is outside the image.")
    axis = np.stack((np.array(rect[0]) - vector / 2, np.array(rect[0]) + vector / 2))
    ready = angle is not None and abs(angle) <= angle_tolerance and abs(dx) <= center_tolerance and abs(dy) <= center_tolerance and not clipped
    return dict(angle_deg=angle, offset=(dx, dy), ready=bool(ready), axis=axis.tolist(), messages=messages,
                clipped=clipped, angle_tolerance=angle_tolerance, center_tolerance=center_tolerance)


def guidance_commands(guide):
    """Image/view corrections only. No yaw, pitch, thrust or drone-body commands."""
    offset, angle = guide.get("offset"), guide.get("angle_deg")
    if offset is None:
        return dict(aim_x=None, aim_y=None, image_rotation=None, status="SEARCH TARGET")
    tolerance = guide.get("center_tolerance", .12)
    dx, dy = offset
    horizontal = ("RIGHT" if dx > 0 else "LEFT") if abs(dx) > tolerance else None
    vertical = ("DOWN" if dy > 0 else "UP") if abs(dy) > tolerance else None
    rotation = (("CCW" if angle > 0 else "CW")
                if angle is not None and abs(angle) > guide.get("angle_tolerance", 5) else None)
    status = ("REFRAME TARGET" if guide.get("clipped") else "AXIS UNCLEAR" if angle is None else
              "VIEW ALIGNED" if guide.get("ready") else "ADJUST VIEW")
    return dict(aim_x=horizontal, aim_y=vertical, image_rotation=rotation, status=status)
