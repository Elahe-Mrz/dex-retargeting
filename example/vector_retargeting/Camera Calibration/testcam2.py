import pyrealsense2 as rs
import numpy as np
import cv2

# =========================
# CONFIG (EDIT THESE)
# =========================
SERIAL = "215222078301"
# SERIAL_2 = "215222078301"

# MUST match your recording/calibration resolution
WIDTH = 1280
HEIGHT = 720
FPS = 30

# Paths to your saved calibration
K_PATH = "K_cam2.npy"
DIST_PATH = "dist_cam2.npy"

# Optional: test reprojection using RS intrinsics
VIDEO_PATH = "cam1.mp4"
RUN_REPROJECTION_TEST = True

# Chessboard config (for reprojection test)
BOARD_SIZE = (9, 7)
SQUARE_SIZE = 0.029  # meters


# =========================
# LOAD YOUR CALIBRATION
# =========================
K_calib = np.load(K_PATH)
dist_calib = np.load(DIST_PATH).ravel()

print("\n===== YOUR CALIBRATION =====")
print("K_calib:\n", K_calib)
print("dist_calib:\n", dist_calib)


# =========================
# GET REALSENSE INTRINSICS
# =========================
pipeline = rs.pipeline()
config = rs.config()

config.enable_device(SERIAL)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

profile = pipeline.start(config)

stream = profile.get_stream(rs.stream.color)
intr = stream.as_video_stream_profile().get_intrinsics()

pipeline.stop()

K_rs = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
])

dist_rs = np.array(intr.coeffs)

print("\n===== REALSENSE INTRINSICS =====")
print("Resolution:", intr.width, "x", intr.height)
print("K_rs:\n", K_rs)
print("dist_rs:\n", dist_rs)


# =========================
# CHECK RESOLUTION CONSISTENCY
# =========================
print("\n===== RESOLUTION CHECK =====")

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("[WARN] Could not read video for resolution check")
else:
    h, w = frame.shape[:2]
    print(f"Video resolution: {w} x {h}")
    if (w != WIDTH) or (h != HEIGHT):
        print("❗ MISMATCH: Video resolution != RealSense stream resolution")
    else:
        print("✔ Resolution matches")


# =========================
# COMPARE MATRICES
# =========================
print("\n===== DIFFERENCE ANALYSIS =====")

K_diff = K_calib - K_rs
dist_diff = dist_calib - dist_rs[:len(dist_calib)]

print("\nK difference:\n", K_diff)
print("\ndist difference:\n", dist_diff)

# Relative (%) error for focal lengths
fx_rel = abs(K_calib[0,0] - K_rs[0,0]) / K_rs[0,0] * 100
fy_rel = abs(K_calib[1,1] - K_rs[1,1]) / K_rs[1,1] * 100

print("\nRelative error:")
print(f"fx error: {fx_rel:.2f}%")
print(f"fy error: {fy_rel:.2f}%")

# Principal point shift
ppx_diff = K_calib[0,2] - K_rs[0,2]
ppy_diff = K_calib[1,2] - K_rs[1,2]

print(f"ppx shift: {ppx_diff:.2f} px")
print(f"ppy shift: {ppy_diff:.2f} px")


# =========================
# INTERPRETATION
# =========================
print("\n===== INTERPRETATION =====")

if fx_rel < 5 and fy_rel < 5:
    print("✔ Focal lengths are consistent")
else:
    print("❗ Focal length mismatch → wrong resolution or wrong stream")

if abs(ppx_diff) < 20 and abs(ppy_diff) < 20:
    print("✔ Principal point is consistent")
else:
    print("❗ Principal point mismatch → cropping/resizing or wrong stream")

if np.linalg.norm(dist_diff) < 0.1:
    print("✔ Distortion is roughly consistent")
else:
    print("❗ Distortion mismatch → different model or calibration inconsistency")


# =========================
# OPTIONAL: REPROJECTION TEST USING RS INTRINSICS
# =========================
if RUN_REPROJECTION_TEST:
    print("\n===== REPROJECTION TEST (USING REALSENSE INTRINSICS) =====")

    objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    cap = cv2.VideoCapture(VIDEO_PATH)
    errors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE)

        if not found:
            continue

        corners = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        success, rvec, tvec = cv2.solvePnP(objp, corners, K_rs, dist_rs[:len(dist_calib)])
        if not success:
            continue

        proj, _ = cv2.projectPoints(objp, rvec, tvec, K_rs, dist_rs[:len(dist_calib)])

        err = np.linalg.norm(corners - proj, axis=2).mean()
        errors.append(err)

    cap.release()

    if errors:
        print(f"Mean reprojection error (RS intrinsics): {np.mean(errors):.3f} px")
    else:
        print("No valid frames for reprojection test")