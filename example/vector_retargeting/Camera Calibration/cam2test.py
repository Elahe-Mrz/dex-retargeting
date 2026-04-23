#!/usr/bin/env python3

import cv2
import numpy as np
import pyrealsense2 as rs
import time

# ================= CONFIG =================

SERIAL = "215222078301"   # <-- your camera 2 serial

WIDTH, HEIGHT, FPS = 640, 480, 15

CHECKERBOARD = (8, 6)   # 9x7 squares → 8x6 corners
SQUARE_SIZE = 0.025     # meters

MAX_SAMPLES = 20
COOLDOWN = 0.8

SAVE_FILE = "cam2_intrinsics.npz"

# ==========================================

def create_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)
    return pipeline

# Prepare object points
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

pipe = create_pipeline(SERIAL)

# Warmup
for _ in range(30):
    pipe.wait_for_frames()

print("Starting capture...")

last_capture = 0
prev_corners = None

criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

while True:

    frames = pipe.wait_for_frames()
    frame = frames.get_color_frame()

    if not frame:
        continue

    img = np.asanyarray(frame.get_data())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

    vis = img.copy()

    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners, ret)

        # ---------- QUALITY FILTERS ----------

        # 1. bounding box size
        x, y, w, h = cv2.boundingRect(corners)
        if w < 150 or h < 150:
            cv2.putText(vis, "Too small", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Cam2", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # 2. near image border
        if x < 20 or y < 20 or (x+w) > (WIDTH-20) or (y+h) > (HEIGHT-20):
            cv2.putText(vis, "Too close to edge", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Cam2", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # 3. duplicate pose rejection
        if prev_corners is not None:
            movement = np.linalg.norm(corners - prev_corners)
            if movement < 10:
                cv2.putText(vis, "Duplicate", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Cam2", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

        # ---------- AUTO CAPTURE ----------

        now = time.time()
        if now - last_capture > COOLDOWN:

            print(f"Captured {len(objpoints)}")

            objpoints.append(objp)
            imgpoints.append(corners)

            prev_corners = corners.copy()
            last_capture = now

    cv2.imshow("Cam2", vis)

    if len(objpoints) >= MAX_SAMPLES:
        print("Enough samples collected.")
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

pipe.stop()
cv2.destroyAllWindows()

# ================= CALIBRATION =================

print("\nRunning calibration...")

ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    (WIDTH, HEIGHT),
    None,
    None
)

print("\nK:\n", K)
print("\nD:\n", D.ravel())

# ================= ERROR =================

total_error = 0
total_points = 0

for i in range(len(objpoints)):
    proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    proj = proj.reshape(-1, 2)
    imgp = imgpoints[i].reshape(-1, 2)

    err = np.mean(np.linalg.norm(imgp - proj, axis=1))
    print(f"Frame {i}: error = {err:.3f} px")

    total_error += err * len(objpoints[i])
    total_points += len(objpoints[i])

mean_error = total_error / total_points

print("\n===== FINAL RESULT =====")
print(f"Mean reprojection error: {mean_error:.3f} px")

if mean_error < 1:
    print("Quality: GOOD")
elif mean_error < 2:
    print("Quality: OK")
else:
    print("Quality: BAD")

# ================= SAVE =================

np.savez(SAVE_FILE, K=K, D=D)

print(f"\nSaved intrinsics to {SAVE_FILE}")