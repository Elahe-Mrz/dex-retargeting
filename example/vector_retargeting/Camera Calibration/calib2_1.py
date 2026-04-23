#!/usr/bin/env python3

import cv2
import numpy as np
import pyrealsense2 as rs
import time

# ================= CONFIG =================

SERIAL_1 = "215322071654" # cam1
SERIAL_2 = "213622077408" # cam2

WIDTH, HEIGHT, FPS = 640, 480, 30

CHECKERBOARD = (8, 6)
SQUARE_SIZE = 0.029

MAX_SAMPLES = 20
COOLDOWN = 1.0

SAVE_FILE = "stereo_fixed_intrinsics_cam12_n.npz"

# ==========================================

def create_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)
    return pipeline, profile

def get_color_intrinsics(profile):
    """
    Read intrinsics from the ACTIVE RealSense color stream profile.
    This is the correct way when using pyrealsense2 directly.
    """
    color_stream = profile.get_stream(rs.stream.color)
    vsp = color_stream.as_video_stream_profile()
    intr = vsp.get_intrinsics()

    K = np.array([
        [intr.fx, 0.0,    intr.ppx],
        [0.0,    intr.fy, intr.ppy],
        [0.0,    0.0,     1.0]
    ], dtype=np.float64)

    # RealSense gives 5 distortion coeffs for Modified Brown-Conrady models typically.
    D = np.array(intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)

    return K, D, intr

def canonicalize(corners):
    pts = corners.reshape(CHECKERBOARD[1], CHECKERBOARD[0], 2)

    # top-to-bottom consistency
    if pts[0, 0, 1] > pts[-1, 0, 1]:
        pts = pts[::-1, :, :]

    # left-to-right consistency
    if pts[0, 0, 0] > pts[0, -1, 0]:
        pts = pts[:, ::-1, :]

    return pts.reshape(-1, 1, 2)

# ================= OBJECT POINTS =================

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints1 = []
imgpoints2 = []

# ================= START CAMERAS =================

pipe1, profile1 = create_pipeline(SERIAL_1)
pipe2, profile2 = create_pipeline(SERIAL_2)

# warmup
for _ in range(30):
    pipe1.wait_for_frames()
    pipe2.wait_for_frames()

# Read intrinsics from the ACTIVE stream profiles
K1, D1, intr1 = get_color_intrinsics(profile1)
K2, D2, intr2 = get_color_intrinsics(profile2)

print("\n=== Camera 1 Intrinsics ===")
print(f"Serial: {SERIAL_1}")
print(f"Resolution: {intr1.width} x {intr1.height}")
print(f"fx={intr1.fx:.6f}, fy={intr1.fy:.6f}, ppx={intr1.ppx:.6f}, ppy={intr1.ppy:.6f}")
print("K1:\n", K1)
print("D1:\n", D1.ravel())

print("\n=== Camera 2 Intrinsics ===")
print(f"Serial: {SERIAL_2}")
print(f"Resolution: {intr2.width} x {intr2.height}")
print(f"fx={intr2.fx:.6f}, fy={intr2.fy:.6f}, ppx={intr2.ppx:.6f}, ppy={intr2.ppy:.6f}")
print("K2:\n", K2)
print("D2:\n", D2.ravel())

print("\nCapturing stereo pairs...")

last_capture = 0.0
prev = None

criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

# ================= CAPTURE LOOP =================

while True:
    frames1 = pipe1.wait_for_frames()
    frames2 = pipe2.wait_for_frames()

    color1 = frames1.get_color_frame()
    color2 = frames2.get_color_frame()

    if not color1 or not color2:
        continue

    img1 = np.asanyarray(color1.get_data())
    img2 = np.asanyarray(color2.get_data())

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ret1, c1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, find_flags)
    ret2, c2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, find_flags)

    vis1 = img1.copy()
    vis2 = img2.copy()

    ready_to_capture = False

    if ret1:
        c1 = cv2.cornerSubPix(gray1, c1, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(vis1, CHECKERBOARD, c1, ret1)

    if ret2:
        c2 = cv2.cornerSubPix(gray2, c2, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(vis2, CHECKERBOARD, c2, ret2)

    if ret1 and ret2:
        x1, y1, w1, h1 = cv2.boundingRect(c1)
        x2, y2, w2, h2 = cv2.boundingRect(c2)

        # Ensure board is large enough and not too close to edges
        if w1 >= 150 and w2 >= 150 and x1 >= 20 and y1 >= 20 and x2 >= 20 and y2 >= 20:
            c1 = canonicalize(c1)
            c2 = canonicalize(c2)
            ready_to_capture = True
            cv2.putText(vis1, "Press 'c' to Capture", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(vis1, "Board too far or on edge", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    cv2.putText(vis1, f"Samples: {len(objpoints)}/{MAX_SAMPLES}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("cam1", vis1)
    cv2.imshow("cam2", vis2)

    # Single waitKey call to prevent missed keystrokes
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('c') and ready_to_capture:
        now = time.time()
        if now - last_capture > COOLDOWN:
            print(f"Captured {len(objpoints) + 1}/{MAX_SAMPLES}")
            objpoints.append(objp.copy())
            imgpoints1.append(c1.copy())
            imgpoints2.append(c2.copy())
            last_capture = now

    if len(objpoints) >= MAX_SAMPLES:
        print("Enough samples collected!")
        break

pipe1.stop()
pipe2.stop()
cv2.destroyAllWindows()

# ================= STEREO CALIB =================

if len(objpoints) < 5:
    raise RuntimeError(f"Not enough valid stereo pairs captured: {len(objpoints)}")

print("\nRunning stereo calibration...")

stereo_flags = cv2.CALIB_FIX_INTRINSIC

ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints1,
    imgpoints2,
    K1, D1,
    K2, D2,
    (WIDTH, HEIGHT),
    flags=stereo_flags,
    criteria=(cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 100, 1e-5)
)

print(f"\nStereo RMS error: {ret:.6f}")
print("\nR:\n", R)
print("\nT:\n", T)

np.savez(
    SAVE_FILE,
    R=R,
    T=T,
    E=E,
    F=F,
    rms=ret,
    K1=K1,
    K2=K2,
    D1=D1,
    D2=D2,
    serial_1=SERIAL_1,
    serial_2=SERIAL_2,
    width=WIDTH,
    height=HEIGHT,
    fps=FPS,
    objpoints=np.array(objpoints, dtype=object),
    imgpoints1=np.array(imgpoints1, dtype=object),
    imgpoints2=np.array(imgpoints2, dtype=object),
)

print("\nSaved to", SAVE_FILE)