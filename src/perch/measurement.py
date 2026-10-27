"""Mask cleanup, robust depth measurements and perch suitability thresholds."""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

def clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray | None:
    """Remove tiny disconnected islands; keep holes and large occluded pieces.

    No convex hull, dilation or polygon fill: these would invent foreground.
    """
    binary = (mask > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    threshold = max(min_area, int(areas.max() * 0.01))
    keep = np.flatnonzero(areas >= threshold) + 1
    if not len(keep):
        return None
    return np.isin(labels, keep).astype(np.uint8)


@dataclass
class Measurement:
    distance: float | None = None
    length: float | None = None
    width: float | None = None
    valid_fraction: float = 0.0
    reason: str = "No reliable depth in mask"

    def values(self):
        if self.distance is None:
            return None
        return self.distance, self.length, self.width


def measure_mask(mask, depth_m, focal, min_pixels=20, min_fraction=0.15):
    """Estimate projected dimensions from an oriented mask, with robust masked depth.

    Border erosion only selects depth samples; it never changes the displayed
    mask or its measured outline. Dimensions assume one representative Z plane.
    """
    result = Measurement()
    distance_to_edge = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    margin = min(2.0, float(distance_to_edge.max()) * 0.25)
    core = (mask > 0) & (distance_to_edge > margin)
    valid = core & np.isfinite(depth_m) & (depth_m > 0)
    result.valid_fraction = float(valid.sum() / max(1, core.sum()))
    samples = depth_m[valid]
    if samples.size < min_pixels or result.valid_fraction < min_fraction:
        return result, None
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    # MAD alone can be zero for a split between two flat depth planes.
    q25, q75 = np.percentile(samples, (25, 75))
    if (q75 - q25) / median > 0.40 or 1.4826 * mad / median > 0.20:
        result.reason = "Depth spread too large"
        return result, None
    inliers = samples[np.abs(samples - median) <= max(0.005, 3 * 1.4826 * mad)]
    if inliers.size < min_pixels or inliers.size / max(1, core.sum()) < min_fraction:
        result.reason = "Too few depth inliers"
        return result, None
    z = float(np.median(inliers))
    points = cv2.findNonZero(mask)
    rect = cv2.minAreaRect(points)
    length_px, width_px = sorted(rect[1], reverse=True)
    if width_px < 1 or focal <= 0:
        result.reason = "Mask too thin to measure"
        return result, None
    result.distance = z
    result.length = length_px * z / focal
    result.width = width_px * z / focal
    result.reason = "Measured"
    return result, cv2.boxPoints(rect)


def assess(values):
    """Same distance/length/width thresholds as perch_detector.py, in metres."""
    if values is None:
        return None, "UNKNOWN: insufficient depth/history"
    distance, length, width = values
    reasons = []
    if not 0.20 <= distance <= 5.00:
        reasons.append("distance outside 0.20-5.00m")
    if not 0.10 <= length <= 2.00:
        reasons.append("length outside 0.10-2.00m")
    if not 0.02 <= width <= 0.20:
        reasons.append("width outside 0.02-0.20m")
    return (False, "; ".join(reasons)) if reasons else (True, "WITHIN SIZE/DIST LIMITS")
