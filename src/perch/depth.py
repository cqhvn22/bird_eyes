"""Reference SGBM/WLS depth and rectified mask projection; numerical algorithm preserved."""
from __future__ import annotations

import cv2
import numpy as np

class StereoDepth:
    """SGBM stereo matcher with WLS filter – same parameters as the live
    pipeline so depth values are directly comparable."""

    def __init__(self, calibration_file: str, downscale: float = 0.5) -> None:
        data = np.load(calibration_file)

        self.left_map_x  = data["left_map_x"]
        self.left_map_y  = data["left_map_y"]
        self.right_map_x = data["right_map_x"]
        self.right_map_y = data["right_map_y"]
        self.Q           = data["Q"].copy()

        # Resolution the calibration was computed at
        self._calib_w = int(self.left_map_x.shape[1])
        self._calib_h = int(self.left_map_x.shape[0])
        self._video_w: int | None = None  # set on first process() call
        self._video_h: int | None = None

        print(f"Calibration map size : {self._calib_w}x{self._calib_h}")

        self.downscale = downscale

        block_size = 7
        num_disp   = 128 if downscale >= 1.0 else 96

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8  * block_size ** 2,
            P2=32 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        if not hasattr(cv2, "ximgproc"):
            raise RuntimeError(
                "cv2.ximgproc is required for stereo WLS filtering. "
                "Install the OpenCV contrib package with: "
                "pip install opencv-contrib-python"
            )

        if not hasattr(cv2.ximgproc, "createRightMatcher"):
            raise RuntimeError(
                "This OpenCV build is missing the ximgproc stereo helpers "
                "(createRightMatcher / createDisparityWLSFilter). "
                "Install opencv-contrib-python and re-run the script."
            )

        self.right_matcher = cv2.ximgproc.createRightMatcher(self.matcher)
        self.wls_filter    = cv2.ximgproc.createDisparityWLSFilter(
            matcher_left=self.matcher
        )
        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.5)

    def process(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a stereo pair and return rectified image, disparity, and depth.

        If the input frames differ from the calibration resolution (including
        a different aspect ratio), they are resized to the calibration size
        before remapping. This is the only geometrically correct approach when
        the aspect ratios do not match (e.g. 640x480 video vs 960x540 calib).

        Returns:
            left_rectified  – BGR image at calibration resolution
            disparity       – float32 disparity map
            depth           – float32 depth map in the calibration unit (mm)
        """
        h, w = left.shape[:2]

        # Resize to calibration resolution if needed (handles both same-ratio
        # scale differences and different-aspect-ratio recordings).
        if w != self._calib_w or h != self._calib_h:
            if self._video_w != w or self._video_h != h:
                # Log once per unique video resolution
                print(f"Video size           : {w}x{h}")
                print(f"Calibration size     : {self._calib_w}x{self._calib_h}")
                if abs(w / h - self._calib_w / self._calib_h) > 0.01:
                    print("WARNING: Different aspect ratios detected "
                          f"({w/h:.3f} vs {self._calib_w/self._calib_h:.3f}). "
                          "Frames will be stretched to calibration resolution.\n"
                          "For accurate depth, re-record videos at "
                          f"{self._calib_w}x{self._calib_h} or re-calibrate at "
                          f"{w}x{h}.\n")
                else:
                    print(f"Resizing frames {w}x{h} -> "
                          f"{self._calib_w}x{self._calib_h} (same aspect ratio).\n")
                self._video_w = w
                self._video_h = h

            left  = cv2.resize(left,  (self._calib_w, self._calib_h),
                                interpolation=cv2.INTER_LINEAR)
            right = cv2.resize(right, (self._calib_w, self._calib_h),
                                interpolation=cv2.INTER_LINEAR)

        left_rectified  = cv2.remap(left,  self.left_map_x,  self.left_map_y,  cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)

        left_gray  = cv2.cvtColor(left_rectified,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        if self.downscale != 1.0:
            small_left  = cv2.resize(left_gray,  None,
                                     fx=self.downscale, fy=self.downscale,
                                     interpolation=cv2.INTER_AREA)
            small_right = cv2.resize(right_gray, None,
                                     fx=self.downscale, fy=self.downscale,
                                     interpolation=cv2.INTER_AREA)
        else:
            small_left, small_right = left_gray, right_gray

        disp_left  = self.matcher.compute(small_left, small_right)
        disp_right = self.right_matcher.compute(small_right, small_left)

        filtered = self.wls_filter.filter(
            disp_left, small_left, disparity_map_right=disp_right
        )

        disparity = filtered.astype(np.float32) / 16.0

        if self.downscale != 1.0:
            disparity = cv2.resize(
                disparity,
                (left_gray.shape[1], left_gray.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ) / self.downscale

        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        depth = points_3d[:, :, 2]

        valid = np.isfinite(depth) & (disparity > 0) & (depth > 0)
        depth[~valid] = 0

        return left_rectified, disparity, depth


def depth_to_color(depth: np.ndarray,
                   max_depth_m: float = 5.0,
                   calibration_unit: str = "mm") -> np.ndarray:
    """Render a float32 depth map as a colour image."""
    depth_m = depth.copy()
    if calibration_unit == "mm":
        depth_m /= 1000.0
    elif calibration_unit == "cm":
        depth_m /= 100.0

    valid   = (depth_m > 0) & np.isfinite(depth_m)
    display = np.zeros(depth_m.shape, dtype=np.uint8)

    if np.any(valid):
        normalized       = np.clip(depth_m / max_depth_m, 0, 1)
        display          = (normalized * 255).astype(np.uint8)
        display[~valid]  = 0

    return cv2.applyColorMap(display, cv2.COLORMAP_PLASMA)


class SegmentDepth(StereoDepth):
    """Use the reference StereoDepth unchanged; add only mask projection.

    process() is inherited directly, returning rectified left, disparity and
    depth in calibration units, with the reference's zero-valued invalid pixels.
    """

    def __init__(self, args, video_size=None):
        super().__init__(str(args.calib), downscale=args.downscale)
        self.size = (self.left_map_x.shape[1], self.left_map_x.shape[0])
        self.maps = [self.left_map_x, self.left_map_y, self.right_map_x, self.right_map_y]

    def rectify_mask(self, mask, eye="left"):
        """Project a mask into its eye's rectified image; depth stays unchanged."""
        resized = cv2.resize(mask.astype(np.uint8), self.size, interpolation=cv2.INTER_NEAREST)
        offset = 0 if eye == "left" else 2
        rectified = cv2.remap(resized, self.maps[offset], self.maps[offset + 1], cv2.INTER_NEAREST)
        return rectified

    def rectify_left(self, left):
        """Left-eye preview without stereo matching or depth reconstruction."""
        if (left.shape[1], left.shape[0]) != self.size:
            left = cv2.resize(left, self.size, interpolation=cv2.INTER_LINEAR)
        return cv2.remap(left, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)

    def rectify_right(self, right):
        """Right-eye preview only, using the reference's resize/remap operations."""
        if (right.shape[1], right.shape[0]) != self.size:
            right = cv2.resize(right, self.size, interpolation=cv2.INTER_LINEAR)
        return cv2.remap(right, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)
