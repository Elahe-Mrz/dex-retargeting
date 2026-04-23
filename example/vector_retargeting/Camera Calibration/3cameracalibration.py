#!/usr/bin/env python3

import cv2
import numpy as np
import pyrealsense2 as rs
import time

# ================= CONFIG =================

SERIAL_1 = "215322071654"
SERIAL_2 = "215222078301"
SERIAL_3 = "213622077408"

WIDTH, HEIGHT, FPS = 640, 480, 15

CHECKERBOARD = (8, 6)
SQUARE_SIZE = 0.029

MAX_SAMPLES = 25
SAVE_FILE = "multi_cam_calib.npz"

SYNC_THRESHOLD_MS = 30       # realistic for RealSense
MOTION_THRESHOLD = 2.0
REPROJ_THRESHOLD = 1.5

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

def canonicalize_corners(corners, pattern_size):
    corners = corners.reshape(pattern_size[1], pattern_size[0], 2)

    if corners[0, 0, 1] > corners[-1, 0, 1]:
        corners = corners[::-1, :, :]

    if corners[0, 0, 0] > corners[0, -1, 0]:
        corners = corners[:, ::-1, :]

    return corners.reshape(-1, 1, 2)

# ================= SETUP =================

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints1, imgpoints2, imgpoints3 = [], [], []

pipe1 = create_pipeline(SERIAL_1)
pipe2 = create_pipeline(SERIAL_2)
pipe3 = create_pipeline(SERIAL_3)

# Warmup
for _ in range(30):
    pipe1.wait_for_frames()
    pipe2.wait_for_frames()
    pipe3.wait_for_frames()

K1, D1 = get_intrinsics(pipe1)
K2, D2 = get_intrinsics(pipe2)
K3, D3 = get_intrinsics(pipe3)

criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

print("Streaming started...")

prev_corners1 = None
good_samples = 0

# ================= MAIN LOOP =================

while True:

    # frames1 = pipe1.poll_for_frames()
    # frames2 = pipe2.poll_for_frames()
    # frames3 = pipe3.poll_for_frames()
    frames1 = pipe1.wait_for_frames()
    frames2 = pipe2.wait_for_frames()
    frames3 = pipe3.wait_for_frames()

    if not frames1 or not frames2 or not frames3:
        continue

    f1 = frames1.get_color_frame()
    f2 = frames2.get_color_frame()
    f3 = frames3.get_color_frame()

    if not f1 or not f2 or not f3:
        continue

    img1 = np.asanyarray(f1.get_data())
    img2 = np.asanyarray(f2.get_data())
    img3 = np.asanyarray(f3.get_data())

    # ================= DISPLAY ALWAYS =================
    vis1, vis2, vis3 = img1.copy(), img2.copy(), img3.copy()

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, flags)
    ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, flags)
    ret3, corners3 = cv2.findChessboardCorners(gray3, CHECKERBOARD, flags)

    if ret1:
        corners1 = cv2.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(vis1, CHECKERBOARD, corners1, ret1)

    if ret2:
        corners2 = cv2.cornerSubPix(gray2, corners2, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(vis2, CHECKERBOARD, corners2, ret2)

    if ret3:
        corners3 = cv2.cornerSubPix(gray3, corners3, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(vis3, CHECKERBOARD, corners3, ret3)

    cv2.imshow("Cam1", vis1)
    cv2.imshow("Cam2", vis2)
    cv2.imshow("Cam3", vis3)

    # ================= SYNC CHECK =================
    t1 = frames1.get_timestamp()
    t2 = frames2.get_timestamp()
    t3 = frames3.get_timestamp()

    sync_diff = max(t1, t2, t3) - min(t1, t2, t3)

    # Debug (optional)
    # print(f"sync diff: {sync_diff:.2f} ms")

    if sync_diff > SYNC_THRESHOLD_MS:
        continue

    # ================= REQUIRE ALL DETECTIONS =================
    if not (ret1 and ret2 and ret3):
        continue

    corners1 = canonicalize_corners(corners1, CHECKERBOARD)
    corners2 = canonicalize_corners(corners2, CHECKERBOARD)
    corners3 = canonicalize_corners(corners3, CHECKERBOARD)

    if not (corners1.shape == corners2.shape == corners3.shape):
        continue

    # ================= MOTION FILTER =================
    if prev_corners1 is not None:
        movement = np.linalg.norm(corners1 - prev_corners1)
        if movement > MOTION_THRESHOLD:
            continue

    # ================= PNP QUALITY CHECK =================
    ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners1, K1, D1)

    if not ret_pnp:
        continue

    proj, _ = cv2.projectPoints(objp, rvec, tvec, K1, D1)
    proj = proj.reshape(-1, 2)

    reproj_error = np.mean(
        np.linalg.norm(corners1.reshape(-1,2) - proj, axis=1)
    )

    if reproj_error > REPROJ_THRESHOLD:
        continue

    # ================= ACCEPT SAMPLE =================
    print(f"[OK] Sample {good_samples} | sync={sync_diff:.2f}ms | reproj={reproj_error:.2f}px")

    objpoints.append(objp)
    imgpoints1.append(corners1)
    imgpoints2.append(corners2)
    imgpoints3.append(corners3)

    prev_corners1 = corners1.copy()
    good_samples += 1

    if good_samples >= MAX_SAMPLES:
        print("Collected enough samples.")
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ================= CLEANUP =================

pipe1.stop()
pipe2.stop()
pipe3.stop()
cv2.destroyAllWindows()

# ================= CALIBRATION =================

flags = (
    cv2.CALIB_FIX_INTRINSIC |
    cv2.CALIB_USE_INTRINSIC_GUESS |
    cv2.CALIB_ZERO_TANGENT_DIST
)

print("Calibrating Cam1 ↔ Cam2...")
_, _, _, _, _, R12, T12, _, _ = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2,
    K1, D1, K2, D2,
    gray1.shape[::-1],
    flags=flags
)

print("Calibrating Cam1 ↔ Cam3...")
_, _, _, _, _, R13, T13, _, _ = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints3,
    K1, D1, K3, D3,
    gray1.shape[::-1],
    flags=flags
)

# ================= SAVE =================

np.savez(
    SAVE_FILE,
    R12=R12, T12=T12,
    R13=R13, T13=T13,
    K1=K1, K2=K2, K3=K3,
    D1=D1, D2=D2, D3=D3,
    objpoints=objpoints,
    imgpoints1=imgpoints1,
    imgpoints2=imgpoints2,
    imgpoints3=imgpoints3
)

print("Saved calibration.")