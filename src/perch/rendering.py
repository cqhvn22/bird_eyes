"""Two-eye preview, animated guidance and measurement result display."""
from __future__ import annotations

import textwrap
import time
import cv2
import numpy as np
from .guidance import guidance_commands

OK, FAIL, UNKNOWN = (40, 220, 60), (40, 60, 240), (0, 200, 255)


def text(image, value, position, color=(240, 240, 240), scale=0.5):
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def _rotation_cue(image, center, radius, direction, phase, color, thickness):
    sign = 1 if direction == "CW" else -1  # OpenCV y grows downwards.
    start = -np.pi/2 + sign * phase * np.pi/3
    angles = start + sign * np.linspace(0, np.pi*.65, 24)
    points = np.column_stack((center[0]+radius*np.cos(angles), center[1]+radius*np.sin(angles)))
    points = np.rint(points).astype(np.int32)
    cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)
    cv2.arrowedLine(image, tuple(points[-5]), tuple(points[-1]), color, thickness,
                    cv2.LINE_AA, tipLength=.65)


def draw_guidance_overlay(image, guide, phase, state):
    """Lightweight reticle and moving aim chevrons in the left camera view."""
    h, w = image.shape[:2]
    scale = min(w/960, h/540)
    center = np.array([w/2, h/2])
    tolerance = guide.get("center_tolerance", .12)
    half = np.array([w*tolerance, h*tolerance])
    color = OK if guide.get("ready") else UNKNOWN
    thick = max(1, round(2*scale))
    for sx in (-1, 1):
        for sy in (-1, 1):
            corner = np.rint(center + half * (sx, sy)).astype(int)
            cv2.line(image, tuple(corner), tuple(corner - np.array([sx*14*scale, 0]).astype(int)), color, thick)
            cv2.line(image, tuple(corner), tuple(corner - np.array([0, sy*14*scale]).astype(int)), color, thick)
    if guide.get("offset") is not None:
        target = center + np.array(guide["offset"]) * (w, h)
        cv2.circle(image, tuple(np.rint(target).astype(int)), max(4, round((10+5*phase)*scale)), color, thick, cv2.LINE_AA)
        if not guide.get("ready"):
            vector = target-center
            if np.linalg.norm(vector) > 16*scale:
                for progress in ((phase+i/3) % 1 for i in range(3)):
                    start = center + vector * (.1+.65*progress)
                    end = start + vector * .09
                    cv2.arrowedLine(image, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)),
                                    color, thick, cv2.LINE_AA, tipLength=.75)
    commands = guidance_commands(guide)
    aim = " + ".join(value for value in (commands['aim_x'], commands['aim_y']) if value)
    caption = f"AIM {aim}" if aim else commands['status']
    if commands['image_rotation']:
        caption += f" | IMAGE {commands['image_rotation']}"
    if state == "RESULT":
        caption = "RESULT FROZEN - R FOR LIVE"
    text(image, caption, (max(6, round(12*scale)), h-max(8, round(14*scale))), color, .55*scale)


def draw_guidance_instrument(image, guide, phase, state, paused):
    """Use the idle depth tile for a readable animated drone-view instrument."""
    h, w = image.shape[:2]
    scale = min(w/960, h/540)
    center = (round(w*.30), round(h*.48))
    radius = max(12, round(min(w*.20, h*.30)))
    commands = guidance_commands(guide)
    color = OK if guide.get("ready") else UNKNOWN
    line_width = max(1, round(2*scale))
    cv2.circle(image, center, radius, (90, 100, 100), 1, cv2.LINE_AA)
    cv2.line(image, (center[0]-radius, center[1]), (center[0]+radius, center[1]), (55, 65, 65), 1)
    cv2.line(image, (center[0], center[1]-radius), (center[0], center[1]+radius), (55, 65, 65), 1)
    half = max(3, round(radius*2*guide.get("center_tolerance", .12)))
    cv2.rectangle(image, (center[0]-half, center[1]-half), (center[0]+half, center[1]+half), OK, 1)
    # Neutral drone symbol: it is a view-control aid, not an estimated attitude.
    arm = max(5, round(20*scale))
    for sx, sy in ((-1,-1), (-1,1), (1,-1), (1,1)):
        rotor = (center[0]+sx*arm, center[1]+sy*arm)
        cv2.line(image, center, rotor, (230, 230, 220), line_width)
        cv2.circle(image, rotor, max(3, round(8*scale)), (230, 230, 220), line_width, cv2.LINE_AA)
    cv2.circle(image, center, max(3, round(5*scale)), (230, 230, 220), -1)
    if guide.get("offset") is not None:
        vector = np.array(guide["offset"]) * radius * 2
        length = np.linalg.norm(vector)
        if length > radius*.9:
            vector *= radius*.9/length
        target = np.array(center) + vector
        cv2.circle(image, tuple(np.rint(target).astype(int)), max(4, round((8+4*phase)*scale)), color, line_width, cv2.LINE_AA)
        if length > half:
            for progress in ((phase+i/3) % 1 for i in range(3)):
                point = np.rint(np.array(center)+vector*(.25+.7*progress)).astype(int)
                cv2.circle(image, tuple(point), max(1, round(3*scale)), color, -1, cv2.LINE_AA)
    else:
        angle = phase*2*np.pi
        point = (round(center[0]+radius*.8*np.cos(angle)), round(center[1]+radius*.8*np.sin(angle)))
        cv2.circle(image, point, max(3, round(6*scale)), UNKNOWN, -1, cv2.LINE_AA)
    if commands['image_rotation']:
        _rotation_cue(image, center, radius+round(16*scale), commands['image_rotation'], phase, color, line_width)
    elif guide.get("ready"):
        cv2.circle(image, center, radius+round((6+8*phase)*scale), OK, line_width, cv2.LINE_AA)
    tx = round(w*.56)
    def label(value, y, size=.65, ink=color):
        text(image, value, (tx, round(y*h)), ink, size*scale)
    label("DRONE VIEW / 2D", .22, .66, (230, 230, 220))
    label(commands['status'], .30, .75)
    aim = " + ".join(value for value in (commands['aim_x'], commands['aim_y']) if value)
    label("AIM " + aim if aim else "HOLD AIM" if guide.get('offset') is not None else "FIND A PERCH", .41, .67)
    label("IMAGE " + commands['image_rotation'] if commands['image_rotation'] else
          "AXIS UNCLEAR" if guide.get('angle_deg') is None else "ROLL ALIGNED", .50, .62)
    if guide.get('angle_deg') is not None:
        label(f"Image angle {guide['angle_deg']:+.1f} deg", .58, .57)
    prompt = ("RESULT FROZEN" if state == 'RESULT' else "PAUSED" if paused else
              "KEEP VIEW - MEASURING" if state == 'CAPTURING' else
              "ALIGNED - PRESS T" if guide.get('ready') else "ALIGN, THEN PRESS T")
    text(image, prompt, (round(18*scale), round(h*.84)), color, .72*scale)
    text(image, "Image/view cues only; no drone-body axes or distance command.",
         (round(18*scale), h-max(10, round(22*scale))), (190, 195, 195), .47*scale)


def draw_eye(image, detections, label, target_id=None, guide=None):
    annotated = image.copy()
    for det in detections:
        color = OK if det["verdict"] is True else FAIL if det["verdict"] is False else UNKNOWN
        mask = det["mask"].astype(bool)
        annotated[mask] = (0.7 * annotated[mask] + 0.3 * np.array(color)).astype(np.uint8)
        contours, _ = cv2.findContours(det["mask"], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, color, 1, cv2.LINE_AA)
        if det["oriented"] is not None:
            cv2.polylines(annotated, [np.rint(det["oriented"]).astype(np.int32)], True, color, 1)
        x1, y1, _, _ = det["box"]
        selected = det["track_id"] == target_id
        score = (f"shape {det['confidence']:.2f}" if det.get("source") == "GEOMETRY"
                 else f"{det['confidence']:.0%}")
        caption = f"#{det['track_id']} {det['name']} {score}" + ("  TARGET" if selected else "")
        label_at = (max(5, min(x1, image.shape[1] - 320)), max(60, y1 - 10))
        text(annotated, caption, label_at, color)
        if det.get("source") and det["stable"] is not None:
            z, length, width = det["stable"]
            text(annotated, f"Z {z:.2f}m | L {length*100:.1f}cm | W {width*100:.1f}cm",
                 (label_at[0], min(image.shape[0]-8, label_at[1]+20)), color, scale=0.48)
    h, w = image.shape[:2]
    cv2.drawMarker(annotated, (w // 2, h // 2), (255, 220, 100), cv2.MARKER_CROSS, 26, 1)
    if guide and guide["axis"] is not None:
        axis = np.rint(guide["axis"]).astype(np.int32)
        cx, cy = np.rint(axis.mean(axis=0)).astype(int)
        cv2.line(annotated, tuple(axis[0]), tuple(axis[1]), (255, 255, 0), 2, cv2.LINE_AA)
        for x in range(max(0, cx - 100), min(w, cx + 100), 16):
            cv2.line(annotated, (x, cy), (min(w - 1, x + 8), cy), (255, 255, 255), 1)
    text(annotated, label, (12, 28), scale=0.65)
    return annotated


def compose_view(left, right, depth_color, detections, right_detections, guide,
                 state, index, total, fps, target_id, burst=None, result=None,
                 show_depth=True, paused=False, show_guidance=True, animation_time=None):
    """A stable 2x2 layout: left/right eyes, reference depth, guidance/result."""
    left_view = draw_eye(left, detections, "LEFT - rectified", target_id, guide)
    right_view = draw_eye(right, right_detections, "RIGHT - rectified")
    phase = ((animation_time if animation_time is not None else time.perf_counter()) / 1.8) % 1
    if paused or state == "RESULT":
        phase = .5
    if show_guidance:
        draw_guidance_overlay(left_view, guide, phase, state)
    depth_view = depth_color.copy() if show_depth and depth_color is not None else np.zeros_like(left)
    if show_depth and depth_color is not None:
        for det in detections:
            contours, _ = cv2.findContours(det["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(depth_view, contours, -1, UNKNOWN, 1)
    depth_label = ("DEPTH HIDDEN - press D" if not show_depth else
                   "DEPTH IDLE - press T to measure" if depth_color is None else
                   "DEPTH - Plasma (reference)")
    if show_guidance and (depth_color is None or not show_depth):
        draw_guidance_instrument(depth_view, guide, phase, state, paused)
        depth_label = "DRONE VIEW | depth IDLE (T)" if depth_color is None else "DRONE VIEW | depth hidden (D)"
    text(depth_view, depth_label, (12, 28))
    panel = np.full_like(left, (24, 23, 20))
    h, w = panel.shape[:2]
    scale = min(w / 960, h / 540)
    # Fixed native panels are sized from calibration; resize only the final window.
    def line(value, y, color=(235, 235, 235), size=0.58):
        text(panel, value, (int(18 * scale), int(y * scale)), color, size * scale)
    line(f"{state}" + (" | PAUSED" if paused else "") + f" | pair {index+1}/{total} | {fps:.1f} FPS", 30)
    controls = "T=measure burst   R=live/reset   SPACE=pause"
    line(controls + ("   B=restart" if total != "LIVE" else ""), 60, size=0.49)
    line("G=guide   D=depth   S=snapshot   Q/ESC=quit", 84, size=0.49)
    line(f"ALIGNMENT - target #{target_id}" if target_id is not None else "ALIGNMENT - no target", 122, UNKNOWN)
    if guide["angle_deg"] is not None:
        line(f"Image roll error: {guide['angle_deg']:+.1f} deg", 149)
    for i, message in enumerate(guide["messages"]):
        line(message, 178 + i * 26, OK if guide["ready"] else UNKNOWN, 0.51)
    if state == "CAPTURING":
        line(f"MEASURING: {burst.n_frames}/{burst.requested} stereo pairs", 308, UNKNOWN, 0.72)
        line("Median is computed separately for the selected target.", 340, size=0.51)
    elif result is not None:
        verdict = result["verdict"]
        color = OK if verdict is True else FAIL if verdict is False else UNKNOWN
        title = "CAN LAND (size/distance)" if verdict is True else "CANNOT LAND" if verdict is False else "INSUFFICIENT DATA"
        if result.get("cylinder_confirmed") is False and verdict is not None:
            title = "SIZE/DIST PASS (candidate)" if verdict else "SIZE/DIST FAIL (candidate)"
        line(title, 306, color, 0.78)
        for i, message in enumerate(textwrap.wrap(result["reason"], width=76)):
            line(message, 336 + i * 24, color, 0.49)
        if result["values"] is not None:
            z, length, width = result["values"]
            line(f"Distance {z:.3f} m | Length {length*100:.1f} cm | Width {width*100:.1f} cm", 419, size=0.54)
        line(f"Burst: {result['n_frames']}/{result['requested']} pairs | valid: {result['valid_frames']}", 451, size=0.51)
        if "objects" in result:
            line(f"Candidates measured: {len(result['objects'])} | all results in JSON (S)", 478, size=0.49)
    else:
        line("NOT MEASURED - align, then press T", 306, UNKNOWN, 0.67)
        line("Only the next burst is used for the landing assessment.", 341, size=0.51)
    footer = ("2D candidate; cylinder shape NOT confirmed in 3D."
              if result is not None and result.get("cylinder_confirmed") is False else
              "2D view alignment; projected dimensions at representative depth.")
    line(footer, 503, size=0.45)
    return cv2.vconcat((cv2.hconcat((left_view, right_view)), cv2.hconcat((depth_view, panel)))), left_view, right_view, depth_view
