#!/usr/bin/env python3

import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time

# ================= CONFIG =================


SERIAL_1 = "215322071654" # cam1
SERIAL_2 = "213622077408" # cam2

WIDTH, HEIGHT, FPS = 640, 480, 15

# 🔥 Your board: 9x7 squares → 8x6 internal corners
CHECKERBOARD = (8, 6)
SQUARE_SIZE = 0.029  # meters

MAX_SAMPLES = 25
COOLDOWN = 1.0

SAVE_FILE = "stereo_calib12_.npz"

# ==========================================

def create_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)
    return pipeline

def get_intrinsics(pipeline):
    profile = pipeline.get_active_profile()
    stream = profile.get_stream(rs.stream.color)
    intr = stream.as_video_stream_profile().get_intrinsics()

    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ], dtype=np.float32)

    D = np.array(intr.coeffs, dtype=np.float32)

    return K, D

# 🔥 CRITICAL FIX: enforce same corner ordering
def canonicalize_corners(corners, pattern_size):
    corners = corners.reshape(pattern_size[1], pattern_size[0], 2)

    # Ensure top row is actually top
    if corners[0, 0, 1] > corners[-1, 0, 1]:
        corners = corners[::-1, :, :]

    # Ensure left-to-right
    if corners[0, 0, 0] > corners[0, -1, 0]:
        corners = corners[:, ::-1, :]

    return corners.reshape(-1, 1, 2)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints1 = []
imgpoints2 = []

pipe1 = create_pipeline(SERIAL_1)
pipe2 = create_pipeline(SERIAL_2)

# Warmup
for _ in range(30):
    pipe1.wait_for_frames()
    pipe2.wait_for_frames()

K1, D1 = get_intrinsics(pipe1)
K2, D2 = get_intrinsics(pipe2)

print("Auto capture started...")

last_capture = 0
prev_corners1 = None

criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

while True:

    f1 = pipe1.wait_for_frames()
    f2 = pipe2.wait_for_frames()

    img1 = np.asanyarray(f1.get_color_frame().get_data())
    img2 = np.asanyarray(f2.get_color_frame().get_data())

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, flags)
    ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, flags)

    vis1 = img1.copy()
    vis2 = img2.copy()

    if ret1:
        corners1 = cv2.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(vis1, CHECKERBOARD, corners1, ret1)

    if ret2:
        corners2 = cv2.cornerSubPix(gray2, corners2, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(vis2, CHECKERBOARD, corners2, ret2)

    cv2.imshow("Cam1", vis1)
    cv2.imshow("Cam2", vis2)

    now = time.time()

    # ✅ AUTO CAPTURE
    if ret1 and ret2 and (now - last_capture > COOLDOWN):

        # Enforce consistent ordering
        corners1 = canonicalize_corners(corners1, CHECKERBOARD)
        corners2 = canonicalize_corners(corners2, CHECKERBOARD)

        # Reject mismatched sizes
        if corners1.shape != corners2.shape:
            continue

        # Reject duplicates (low motion)
        if prev_corners1 is not None:
            movement = np.linalg.norm(corners1 - prev_corners1)
            if movement < 5:
                continue

        print(f"Captured {len(objpoints)}")

        objpoints.append(objp)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)

        prev_corners1 = corners1.copy()
        last_capture = now

    if len(objpoints) >= MAX_SAMPLES:
        print("Enough samples collected.")
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

pipe1.stop()
pipe2.stop()
cv2.destroyAllWindows()

# ================= CALIBRATION =================

print("Running stereo calibration...")

flags = (
    cv2.CALIB_FIX_INTRINSIC |
    cv2.CALIB_USE_INTRINSIC_GUESS |
    cv2.CALIB_ZERO_TANGENT_DIST
)

ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints1,
    imgpoints2,
    K1, D1,
    K2, D2,
    gray1.shape[::-1],
    criteria=(cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 100, 1e-5),
    flags=flags
)

print("\nR:\n", R)
print("\nT:\n", T)

# ================= SAVE =================

np.savez(
    SAVE_FILE,
    R=R, T=T,
    K1=K1, K2=K2,
    D1=D1, D2=D2,
    objpoints=objpoints,
    imgpoints1=imgpoints1,
    imgpoints2=imgpoints2
)

print(f"\nSaved to {SAVE_FILE}")