#!/usr/bin/env python3

import cv2
import numpy as np
import pyrealsense2 as rs
import time

# ================= CONFIG =================

SERIAL_1 = "215322071654"
SERIAL_2 = "215222078301"


WIDTH, HEIGHT, FPS = 640, 480, 30

CHECKERBOARD = (8, 6)
SQUARE_SIZE = 0.029

MAX_SAMPLES = 20
COOLDOWN = 1.0

SAVE_FILE = "stereo_fixed_intrinsics.npz"

# ================= LOAD YOUR INTRINSICS =================

# 👉 REPLACE THESE

K1 = np.array([[605.9, 0, 328.7],
               [0, 605.4, 244.3],
               [0, 0, 1]], dtype=np.float32)
D1 = np.zeros(5)

K2 = np.array([[578.6, 0, 287.0],
               [0, 574.7, 243.9],
               [0, 0, 1]], dtype=np.float32)
D2 = np.array([0.048, -0.034, -0.001, -0.026, -0.437], dtype=np.float32)

# ==========================================

def create_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)
    return pipeline

def canonicalize(corners):
    # enforce consistent orientation
    pts = corners.reshape(CHECKERBOARD[1], CHECKERBOARD[0], 2)

    if pts[0,0,1] > pts[-1,0,1]:
        pts = pts[::-1,:,:]

    if pts[0,0,0] > pts[0,-1,0]:
        pts = pts[:,::-1,:]

    return pts.reshape(-1,1,2)

# object points
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints1 = []
imgpoints2 = []

pipe1 = create_pipeline(SERIAL_1)
pipe2 = create_pipeline(SERIAL_2)

# warmup
for _ in range(30):
    pipe1.wait_for_frames()
    pipe2.wait_for_frames()

print("Capturing stereo pairs...")

last_capture = 0
prev = None

criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

while True:

    f1 = pipe1.wait_for_frames()
    f2 = pipe2.wait_for_frames()

    img1 = np.asanyarray(f1.get_color_frame().get_data())
    img2 = np.asanyarray(f2.get_color_frame().get_data())

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    ret1, c1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, flags)
    ret2, c2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, flags)

    vis1 = img1.copy()
    vis2 = img2.copy()

    if ret1:
        c1 = cv2.cornerSubPix(gray1, c1, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(vis1, CHECKERBOARD, c1, ret1)

    if ret2:
        c2 = cv2.cornerSubPix(gray2, c2, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(vis2, CHECKERBOARD, c2, ret2)

    cv2.imshow("cam1", vis1)
    cv2.imshow("cam2", vis2)

    if ret1 and ret2:

        # ---------- STRICT FILTERS ----------

        x1,y1,w1,h1 = cv2.boundingRect(c1)
        x2,y2,w2,h2 = cv2.boundingRect(c2)

        # size
        if w1 < 150 or w2 < 150:
            continue

        # edges
        if x1<20 or y1<20 or x2<20 or y2<20:
            continue

        # canonical ordering
        c1 = canonicalize(c1)
        c2 = canonicalize(c2)

        # duplicate rejection
        if prev is not None:
            if np.linalg.norm(c1 - prev) < 10:
                continue

        now = time.time()
        if now - last_capture > COOLDOWN:

            print(f"Captured {len(objpoints)}")

            objpoints.append(objp)
            imgpoints1.append(c1)
            imgpoints2.append(c2)

            prev = c1.copy()
            last_capture = now

    if len(objpoints) >= MAX_SAMPLES:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

pipe1.stop()
pipe2.stop()
cv2.destroyAllWindows()

# ================= STEREO CALIB =================

print("\nRunning stereo calibration...")

flags = cv2.CALIB_FIX_INTRINSIC

ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints1,
    imgpoints2,
    K1, D1,
    K2, D2,
    (WIDTH, HEIGHT),
    flags=flags,
    criteria=(cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 100, 1e-5)
)

print("\nR:\n", R)
print("\nT:\n", T)

np.savez(
    SAVE_FILE,
    R=R, T=T,
    K1=K1, K2=K2,
    D1=D1, D2=D2,
    objpoints=objpoints,
    imgpoints1=imgpoints1,
    imgpoints2=imgpoints2
)

print("\nSaved to", SAVE_FILE)