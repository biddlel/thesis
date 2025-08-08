#!/usr/bin/env python3
"""
capture_test_undistort.py
Save one undistorted frame from the Pi Camera Module 3 Wide.

• Requires cm3wide_calib.npz in the same directory.
• Uses Picamera2 for capture, OpenCV for undistort & I/O.
"""

import time, cv2, numpy as np
from pathlib import Path
from picamera2 import Picamera2

# ─── configuration ───────────────────────────────────────────────
UNDIST_OUTFILE      = Path("test_capture_undist.jpg")
DIST_OUTFILE      = Path("test_capture_dist.jpg")
CAP_SIZE     = (1280, 720)        # (width, height)
ROTATE180    = True               # flip image
SWAP_RB      = True               # Picamera2 gives RGB; cv2.imwrite expects BGR
CALIB_FILE   = Path("cm3wide_calib.npz")
# ─────────────────────────────────────────────────────────────────

# ─── load calibration file ───────────────────────────────────────
if not CALIB_FILE.exists():
    raise FileNotFoundError(
        f"{CALIB_FILE} missing – run calibrate_picam_cv.py first")

calib = np.load(CALIB_FILE)
CAM_MTX, DIST_COEFFS = calib["mtx"], calib["dist"]

# Build undistort rectification maps for chosen output size
def build_remap(size):
    w, h = size
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(
        CAM_MTX, DIST_COEFFS, (w, h), alpha=0.5)  # alpha=0 = crop black border
    return cv2.initUndistortRectifyMap(
        CAM_MTX, DIST_COEFFS, None, new_mtx, (w, h), cv2.CV_16SC2)

remap_x, remap_y = build_remap(CAP_SIZE)

# ─── Picamera2 setup ─────────────────────────────────────────────
picam = Picamera2()
picam.configure(picam.create_still_configuration({"size": CAP_SIZE}))
picam.start()
time.sleep(0.3)                       # warm-up / auto-exposure settle

# ─── capture frame ───────────────────────────────────────────────
frame_rgb = picam.capture_array("main")   # RGB
picam.close()

# ─── undistort & post-process ───────────────────────────────────
frame_undist = cv2.remap(frame_rgb, remap_x, remap_y, cv2.INTER_LINEAR)


if SWAP_RB:
    frame_dist = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    frame_undist = cv2.cvtColor(frame_undist, cv2.COLOR_RGB2BGR)
if ROTATE180:
    frame_dist = cv2.rotate(frame_dist, cv2.ROTATE_180)
    frame_undist = cv2.rotate(frame_undist, cv2.ROTATE_180)

cv2.imwrite(str(UNDIST_OUTFILE), frame_undist)
cv2.imwrite(str(DIST_OUTFILE), frame_dist)

print(f"✅ Saved undistorted image → {UNDIST_OUTFILE.resolve()}")