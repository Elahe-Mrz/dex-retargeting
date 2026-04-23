#!/usr/bin/env python3

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp

# ================= CONFIG =================

SERIAL_1 = "215322071654"   # Cam1
SERIAL_2 = "213622077408"   # Cam3

CALIB_FILE = "stereo_calib.npz"

WIDTH, HEIGHT, FPS = 640, 480, 30

BAD_FRAMES = [1, 5, 7, 15]

# ==========================================

# ---------- LOAD DATA ----------
data = np.load(CALIB_FILE, allow_pickle=True)

objpoints = list(data["objpoints"])
imgpoints1 = list(data["imgpoints1"])
imgpoints2 = list(data["imgpoints2"])

K1, D1 = data["K1"], data["D1"]
K2, D2 = data["K2"], data["D2"]

# ---------- REMOVE BAD FRAMES ----------
objpoints = [o for i,o in enumerate(objpoints) if i not in BAD_FRAMES]
imgpoints1 = [o for i,o in enumerate(imgpoints1) if i not in BAD_FRAMES]
imgpoints2 = [o for i,o in enumerate(imgpoints2) if i not in BAD_FRAMES]

print(f"Using {len(objpoints)} cleaned frames")

# ---------- FORMAT DATA FOR OPENCV ----------

objpoints_clean = []
imgpoints1_clean = []
imgpoints2_clean = []

for o, p1, p2 in zip(objpoints, imgpoints1, imgpoints2):

    # object points → (N,1,3) float32
    o = np.array(o, dtype=np.float32).reshape(-1,1,3)

    # image points → (N,1,2) float32
    p1 = np.array(p1, dtype=np.float32).reshape(-1,1,2)
    p2 = np.array(p2, dtype=np.float32).reshape(-1,1,2)

    # sanity check
    if o.shape[0] != p1.shape[0] or o.shape[0] != p2.shape[0]:
        continue

    objpoints_clean.append(o)
    imgpoints1_clean.append(p1)
    imgpoints2_clean.append(p2)

print(f"Valid frames after formatting: {len(objpoints_clean)}")

# ---------- RECALIBRATE ----------
ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
    objpoints_clean,
    imgpoints1_clean,
    imgpoints2_clean,
    K1, D1,
    K2, D2,
    (WIDTH, HEIGHT),
    flags=cv2.CALIB_FIX_INTRINSIC
)
print("T norm:", np.linalg.norm(T))
print("Recalibrated RMS:", ret)
np.savez(
    "stereo_cam12_1_recalibrated.npz",
    R=R, T=T,
    K1=K1, K2=K2,
    D1=D1, D2=D2,
    objpoints=np.array(objpoints_clean, dtype=object),
    imgpoints1=np.array(imgpoints1_clean, dtype=object),
    imgpoints2=np.array(imgpoints2_clean, dtype=object),
)
# ---------- RECTIFICATION ----------
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1,
    K2, D2,
    (WIDTH, HEIGHT),
    R, T,
    alpha=1
)

map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (WIDTH, HEIGHT), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (WIDTH, HEIGHT), cv2.CV_32FC1)

# ---------- TRIANGULATION ----------
def triangulate(pts1, pts2):
    pts1 = pts1.T
    pts2 = pts2.T
    pts4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts3D = pts4D[:3] / pts4D[3]
    return pts3D.T

# ---------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_landmarks(image):
    results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_hand_landmarks:
        return None, 0.0

    hand = results.multi_hand_landmarks[0]
    conf = results.multi_handedness[0].classification[0].score

    h, w = image.shape[:2]
    pts = []

    for lm in hand.landmark:
        pts.append([lm.x * w, lm.y * h])

    return np.array(pts), conf

# ---------- REPROJECTION ERROR ----------
def reprojection_error(pts3D, pts2D, K, D, rvec, tvec):
    proj, _ = cv2.projectPoints(pts3D, rvec, tvec, K, D)
    proj = proj.reshape(-1,2)
    return np.mean(np.linalg.norm(pts2D - proj, axis=1))

# ---------- REALSENSE ----------
def start_cam(serial):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipe.start(cfg)
    return pipe

pipe1 = start_cam(SERIAL_1)
pipe2 = start_cam(SERIAL_2)

print("Running tracking...")

# ---------- MAIN LOOP ----------
while True:

    f1 = pipe1.wait_for_frames().get_color_frame()
    f2 = pipe2.wait_for_frames().get_color_frame()

    if not f1 or not f2:
        continue

    img1 = np.asanyarray(f1.get_data())
    img2 = np.asanyarray(f2.get_data())

    # ---------- RECTIFY ----------
    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

    pts1, conf1 = get_landmarks(rect1)
    pts2, conf2 = get_landmarks(rect2)

    display1 = rect1.copy()
    display2 = rect2.copy()

    if pts1 is not None:
        for p in pts1.astype(int):
            cv2.circle(display1, tuple(p), 3, (0,255,0), -1)

    if pts2 is not None:
        for p in pts2.astype(int):
            cv2.circle(display2, tuple(p), 3, (0,255,0), -1)

    if pts1 is not None and pts2 is not None:

        pts3D = triangulate(pts1, pts2)

        # ---- geometric confidence ----
        # estimate pose for cam1
        ok, rvec, tvec = cv2.solvePnP(
            pts3D.astype(np.float32),
            pts1.astype(np.float32),
            K1, D1
        )

        if ok:
            err = reprojection_error(pts3D, pts1, K1, D1, rvec, tvec)
        else:
            err = 10

        # ---- final confidence ----
        conf_geom = np.exp(-err)
        conf_final = min(conf1, conf2) * conf_geom

        print(f"Conf1={conf1:.2f} | Conf2={conf2:.2f} | Geom={conf_geom:.2f} | Final={conf_final:.2f}")

        # show one 3D point (e.g., index finger tip)
        idx = 8
        p3d = pts3D[idx]
        cv2.putText(display1, f"Z={p3d[2]:.3f}m", (30,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Cam1 Rectified", display1)
    cv2.imshow("Cam2 Rectified", display2)

    if cv2.waitKey(1) & 0xFF == 27:
        break

pipe1.stop()
pipe2.stop()
cv2.destroyAllWindows()