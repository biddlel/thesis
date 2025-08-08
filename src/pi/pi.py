#!/usr/bin/env python3
"""
doa_yolo_picam.py
────────────────────────────────────────────────────────────────
• Reads 'Angle: XXX, Count: YYY' from a Teensy over USB-Serial
• Captures a still with Picamera2, undistorts using calibration
  from cm3wide_calib.npz, and runs YOLO (Ultralytics).
• Correlates the acoustic DOA with detected objects.

Dependencies:
    pip3 install pyserial ultralytics opencv-python picamera2
"""

# ─── Imports ──────────────────────────────────────────────────
import re, sys, time, os, termios, tty, select
from pathlib import Path
import numpy as np
import cv2
import serial
from picamera2 import Picamera2
from ultralytics import YOLO
from math import atan, degrees, radians, tan

# ─── USER CONFIG ────────────────────────────────────────────
SERIAL_PORT       = "/dev/ttyAMA0"
BAUD_RATE         = 115200

CAP_SIZE          = (1280, 720)  # (width, height)
HFOV_DEG          = 102.0        # Cam Module 3 Wide

MIN_MODE_COUNT    = 7           # Ignore DOA bins with fewer hits
TOP_DRAW          = 4            # Draw up to this many wedges
ANGLE_TOLERANCE   = 20.0         # ±° for DOA↔object match

ANGLE_OFFSET      = 270

ROTATE180         = True         # Flip image if cam is upside-down
SWAP_RB           = True         # Picamera2 gives RGB; cv2 wants BGR

MODEL_PATH        = "models/yolo11s_ncnn_model"  # or yolov11.pt
SAVE_DIR          = Path("captures")

WEDGE_ALPHA_MAIN  = 0.30         # primary wedge opacity
WEDGE_ALPHA_OTH   = 0.15         # other wedges opacity
# ────────────────────────────────────────────────────────────

SAVE_DIR.mkdir(exist_ok=True)

# --- helper maths -------------------------------------------------
def pixel_to_angle(x_px: float, img_w: int) -> float:
    cx = img_w / 2
    fx = cx / tan(radians(HFOV_DEG / 2))
    return degrees(atan((x_px - cx) / fx))

def add_wedge(img, ang_deg, tol, colour, alpha):
    h, w = img.shape[:2]
    overlay = img.copy()
    r = int(min(w, h) * 0.8)
    cv2.ellipse(overlay, (w//2, h//2), (r, r), 0,
                -(ang_deg + tol), -(ang_deg - tol),
                colour, thickness=-1)
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

# --- initialise camera, model, serial ------------------------------
picam = Picamera2()
picam.configure(picam.create_still_configuration({"size": CAP_SIZE}))
picam.start(); time.sleep(0.3)

model = YOLO(MODEL_PATH, task="detect")
ser   = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
pat_modes = re.compile(r"Angles:\s+(.*)")

# user-quit (q ↵) preparation
fd = sys.stdin.fileno()
old_tty = termios.tcgetattr(fd); tty.setcbreak(fd)

try:
    while True:
        # quit on 'q'
        if select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.read(1).lower() == 'q':
                print("🛑  Quitting."); break

        line = ser.readline().decode(errors="ignore").strip()
        m = pat_modes.match(line)
        if not m:
            continue

        # ---------- parse & filter modes ----------
        modes = []
        for tok in m.group(1).split():
            if '°:' in tok:
                try:
                    ang_s, cnt_s = tok.split('°:')
                    cnt = int(cnt_s)
                    if cnt >= MIN_MODE_COUNT:
                        angle = (int(ang_s) + ANGLE_OFFSET) % 360
                        modes.append((angle, cnt))
                except ValueError:
                    pass

        if not modes:
            continue
        modes.sort(key=lambda x: x[1], reverse=True)
        primary_ang, primary_cnt = modes[0]
        print(f"Primary DOA {primary_ang}°  ({primary_cnt} hits)")

        # ---------- capture frame ----------
        frame = picam.capture_array("main")
        if ROTATE180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        if SWAP_RB:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ---------- YOLO ----------
        res = model(frame, task="detect", verbose=False)[0]
        if len(res.boxes):
            best, best_err = None, 1e9
            h, w = frame.shape[:2]
            for box, cls_id, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                x1, _, x2, _ = box
                az = pixel_to_angle((x1 + x2) / 2, w)
                err = min(abs((az - ang) % 360) for ang,_ in modes[:TOP_DRAW])
                if err < best_err:
                    best, best_err = (box, int(cls_id), float(conf), az), err
        else:
            best = None

        ts   = time.strftime("%Y%m%d_%H%M%S")
        out  = SAVE_DIR / f"{ts}.jpg"

        if best and best_err <= ANGLE_TOLERANCE:
            (x1,y1,x2,y2), cls, conf, az_cam = best
            x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{model.names[cls]} {conf:.2f}",
                        (x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)
            # cv2.putText(frame,f"Match {az_cam:.1f}°",(10,frame.shape[0]-10),
            #             cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            out = SAVE_DIR / f"{ts}_match.jpg"
        else:
            cv2.putText(frame,f"No match {primary_ang:.1f}°",
                        (10,frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            for i, (ang, _) in enumerate(modes[:TOP_DRAW]):
                alpha = WEDGE_ALPHA_MAIN if i == 0 else WEDGE_ALPHA_OTH
                colour = (0,0,255) if i == 0 else (0,0,128)
                frame = add_wedge(frame, ang, ANGLE_TOLERANCE, colour, alpha)

            out = SAVE_DIR / f"{ts}_nomatch.jpg"

        cv2.imwrite(str(out), frame)
        print("Saved", out.name)
        time.sleep(0.05)

finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_tty)
    picam.close()
    ser.close()