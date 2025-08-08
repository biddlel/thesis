#!/usr/bin/env python3
"""
calibrate_picam_cv.py
Head-less calibration for Raspberry Pi Camera Module 3 Wide
using Picamera2 for capture and OpenCV for processing.

• Checkerboard: 9 × 7 squares  →  8 × 6 interior corners
• Square size: 25 mm   (change SQUARE_SIZE_MM if needed)
• Run:  python3 calibrate_picam_cv.py --frames 25
"""

import argparse, sys, time, cv2, numpy as np
from pathlib import Path
from picamera2 import Picamera2

# ───────── checkerboard spec ─────────
CHESS_ROWS, CHESS_COLS = 6, 8          # interior corners (height, width)
SQUARE_SIZE_MM         = 25            # each square’s edge
# ──────────────────────────────────────

# CLI options
ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=20,
                help="Good checkerboard views to collect (default 20)")
ap.add_argument("--outfile", default="cm3wide_calib.npz",
                help="Calibration output file (default cm3wide_calib.npz)")
ap.add_argument("--size", default="1280x720",
                help="Capture resolution WxH (default 1280x720)")
ap.add_argument("--rotate180", action="store_true",
                help="Rotate each frame 180° before processing")
args = ap.parse_args()

w_cap, h_cap = map(int, args.size.lower().split("x"))

# Build the object-point template (Z=0 plane)
objp = np.zeros((CHESS_ROWS * CHESS_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESS_COLS, 0:CHESS_ROWS].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints, imgpoints = [], []

# ──────────────────────────────────────
# Picamera2 setup
# ──────────────────────────────────────
picam = Picamera2()
cfg = picam.create_still_configuration({"size": (w_cap, h_cap)})
picam.configure(cfg)
picam.start()
time.sleep(0.3)  # small warm-up

print(f"Collecting {args.frames} good checkerboard views … "
      "(Ctrl-C to abort)")

try:
    while len(objpoints) < args.frames:
        frame_rgb = picam.capture_array("main")   # RGB order

        if args.rotate180:
            frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)

        # OpenCV works in BGR by convention
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray, (CHESS_COLS, CHESS_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if found:
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"  ✔  view {len(objpoints):>2}/{args.frames}")
            time.sleep(0.4)   # give you time to move the board
        else:
            time.sleep(0.05)

except KeyboardInterrupt:
    print("\nInterrupted — continuing with what we have.")

picam.close()

if len(objpoints) < 10:
    sys.exit(f"Need at least 10 good views, got {len(objpoints)}")

print("Calibrating …")
rms, mtx, dist, *_ = cv2.calibrateCamera(
    objpoints, imgpoints, (w_cap, h_cap), None, None)

np.savez(args.outfile, mtx=mtx, dist=dist)

print(f"Done. RMS reprojection error: {rms:.3f} px")
print(f"Saved → {Path(args.outfile).resolve()}")