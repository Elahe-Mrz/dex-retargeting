#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import time
import matplotlib.pyplot as plt

# ============================================================
# ------------------------ CONFIG -----------------------------
# ============================================================

SERIAL_1 = "215322071654"
SERIAL_2 = "215222078301"
SERIAL_3 = "213622077408"

WIDTH = 424
HEIGHT = 240
FPS = 15

BASELINE = 0.08  # placeholder

USE_SMOOTHING = True
ALPHA = 0.4

# ============================================================
# ---------------- REALSENSE SETUP ----------------------------
# ============================================================

def create_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)
    return pipeline

def get_frame(pipeline):
    try:
        frames = pipeline.wait_for_frames(timeout_ms=500)
    except RuntimeError:
        return None, None

    color_frame = frames.get_color_frame()
    if not color_frame:
        return None, None

    img = np.asanyarray(color_frame.get_data())
    ts = frames.get_timestamp()
    return img, ts

def get_intrinsics(pipeline):
    profile = pipeline.get_active_profile()
    stream = profile.get_stream(rs.stream.color)
    intr = stream.as_video_stream_profile().get_intrinsics()

    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float32)

    D = np.array(intr.coeffs, dtype=np.float32)
    return K, D

# ============================================================
# ---------------- MEDIAPIPE SETUP ----------------------------
# ============================================================

mp_hands = mp.solutions.hands

def create_hand():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

hands1 = create_hand()
hands2 = create_hand()
hands3 = create_hand()

def detect(img, hands):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if not res.multi_hand_landmarks:
        return None

    h, w, _ = img.shape
    pts = []

    for lm in res.multi_hand_landmarks[0].landmark:
        pts.append([lm.x * w, lm.y * h])

    return np.array(pts, dtype=np.float32)

# ============================================================
# ---------------- GEOMETRY ----------------------------------
# ============================================================

def undistort_points(pts, K, D):
    pts = pts.reshape(-1, 1, 2).astype(np.float32)
    undist = cv2.undistortPoints(pts, K, D)
    return undist.reshape(-1, 2)

def triangulate_pair(pts1, pts2, K1, D1, K2, D2, R, T):
    pts1_u = undistort_points(pts1, K1, D1)
    pts2_u = undistort_points(pts2, K2, D2)

    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = np.hstack((R, T))

    pts4d = cv2.triangulatePoints(P1, P2, pts1_u.T, pts2_u.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    return pts3d

def triangulate_multi(pts1, pts2, pts3,
                      K1, D1, K2, D2, K3, D3,
                      R12, T12, R13, T13):

    pts_12 = triangulate_pair(pts1, pts2, K1, D1, K2, D2, R12, T12)
    pts_13 = triangulate_pair(pts1, pts3, K1, D1, K3, D3, R13, T13)

    R23 = R13 @ R12.T
    T23 = T13 - R23 @ T12

    pts_23 = triangulate_pair(pts2, pts3, K2, D2, K3, D3, R23, T23)

    return (pts_12 + pts_13 + pts_23) / 3.0

def reproject(pts3d, K, R=None, T=None):
    rvec = np.zeros(3) if R is None else cv2.Rodrigues(R)[0]
    tvec = np.zeros(3) if T is None else T.flatten()

    pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
    return pts2d.reshape(-1, 2)

# ============================================================
# ---------------- SMOOTHING ---------------------------------
# ============================================================

class Smoother:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.prev = None

    def update(self, pts):
        if self.prev is None:
            self.prev = pts
            return pts
        smoothed = self.alpha * pts + (1 - self.alpha) * self.prev
        self.prev = smoothed
        return smoothed

# ============================================================
# ---------------- 3D VISUALIZATION ---------------------------
# ============================================================

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def plot_3d(pts3d):
    ax.cla()
    ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2])
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(0, 0.5)
    plt.draw()
    plt.pause(0.001)

# ============================================================
# ---------------- MAIN --------------------------------------
# ============================================================

def main():

    pipe1 = create_pipeline(SERIAL_1)
    pipe2 = create_pipeline(SERIAL_2)
    pipe3 = create_pipeline(SERIAL_3)

    print("Warming up...")
    for _ in range(30):
        pipe1.wait_for_frames()
        pipe2.wait_for_frames()
        pipe3.wait_for_frames()

    K1, D1 = get_intrinsics(pipe1)
    K2, D2 = get_intrinsics(pipe2)
    K3, D3 = get_intrinsics(pipe3)

    print("K1:\n", K1)
    print("K2:\n", K2)
    print("K3:\n", K3)

    # TEMP extrinsics (replace with real calibration)
    R12 = np.eye(3)
    T12 = np.array([[BASELINE, 0, 0]]).T

    R13 = np.eye(3)
    T13 = np.array([[2*BASELINE, 0, 0]]).T

    smoother = Smoother(ALPHA)

    while True:

        img1, _ = get_frame(pipe1)
        img2, _ = get_frame(pipe2)
        img3, _ = get_frame(pipe3)

        if img1 is None or img2 is None or img3 is None:
            continue

        pts1 = detect(img1, hands1)
        pts2 = detect(img2, hands2)
        pts3 = detect(img3, hands3)

        vis1, vis2, vis3 = img1.copy(), img2.copy(), img3.copy()

        for pts, vis in [(pts1, vis1), (pts2, vis2), (pts3, vis3)]:
            if pts is not None:
                for p in pts.astype(int):
                    cv2.circle(vis, tuple(p), 3, (0,255,0), -1)

        if (pts1 is not None and pts2 is not None and pts3 is not None and
            pts1.shape == pts2.shape == pts3.shape):

            pts3d = triangulate_multi(
                pts1, pts2, pts3,
                K1, D1, K2, D2, K3, D3,
                R12, T12, R13, T13
            )

            if USE_SMOOTHING:
                pts3d = smoother.update(pts3d)

            reproj1 = reproject(pts3d, K1)
            reproj2 = reproject(pts3d, K2, R12, T12)
            reproj3 = reproject(pts3d, K3, R13, T13)

            for p in reproj1.astype(int):
                cv2.circle(vis1, tuple(p), 2, (0,0,255), -1)

            for p in reproj2.astype(int):
                cv2.circle(vis2, tuple(p), 2, (0,0,255), -1)

            for p in reproj3.astype(int):
                cv2.circle(vis3, tuple(p), 2, (0,0,255), -1)

            print(f"Wrist 3D: {pts3d[0]}")
            plot_3d(pts3d)

        cv2.imshow("Cam1", vis1)
        cv2.imshow("Cam2", vis2)
        cv2.imshow("Cam3", vis3)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.01)

    pipe1.stop()
    pipe2.stop()
    pipe3.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()