#!/usr/bin/env python3

import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import time


# ================= CONFIG =================

SERIAL_1 = "215322071654" # cam1
SERIAL_2 = "213622077408" # cam2

WIDTH, HEIGHT, FPS = 640, 480, 15
IMAGE_SIZE = (WIDTH, HEIGHT)

# 9x7 squares → 8x6 internal corners
CHECKERBOARD = (8, 6)
SQUARE_SIZE = 0.029  # meters

MAX_SAMPLES = 50       # increased from 25 for better accuracy
COOLDOWN = 1.5         # seconds between captures
MOVEMENT_THRESH = 20   # pixels — increased from 5 for better coverage diversity

SAVE_FILE = "stereo_calib12_1.npz"

# Max allowed timestamp gap between frames (ms).
# If cameras drift beyond this, the frame pair is rejected.
MAX_TIMESTAMP_DIFF_MS = 80

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
        [intr.fx, 0,       intr.ppx],
        [0,       intr.fy, intr.ppy],
        [0,       0,       1       ]
    ], dtype=np.float32)
    D = np.array(intr.coeffs, dtype=np.float32)
    return K, D

def canonicalize_corners(corners, pattern_size=None):
    """Enforce consistent corner ordering with a valid 180-degree flip to avoid chirality inversion."""
    # Find distance from the origin (0,0) for the first and last corner
    dist_first = corners[0, 0, 0] + corners[0, 0, 1]
    dist_last = corners[-1, 0, 0] + corners[-1, 0, 1]
    
    # If the last corner is closer to the top-left of the image, the board is held upside-down
    if dist_last < dist_first:
        # A full array reversal corresponds to a physical 180-degree rotation
        return corners[::-1]
    return corners

# ================= PARALLEL FRAME GRAB =================

class FrameGrabber:
    """
    Grabs frames from both cameras in parallel threads so capture timestamps
    are as close together as possible (typically < 5ms without hardware sync).
    """
    def __init__(self, pipe1, pipe2):
        self.pipe1 = pipe1
        self.pipe2 = pipe2
        self._frames1 = None
        self._frames2 = None
        self._ts1 = None
        self._ts2 = None

    def _grab1(self):
        self._frames1 = self.pipe1.wait_for_frames()
        self._ts1 = self._frames1.get_timestamp()

    def _grab2(self):
        self._frames2 = self.pipe2.wait_for_frames()
        self._ts2 = self._frames2.get_timestamp()

    def grab(self):
        """Returns (img1, img2, timestamp_diff_ms) or (None, None, None) if sync fails."""
        t1 = threading.Thread(target=self._grab1)
        t2 = threading.Thread(target=self._grab2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        ts_diff = abs(self._ts1 - self._ts2)

        if ts_diff > MAX_TIMESTAMP_DIFF_MS:
            return None, None, ts_diff

        img1 = np.asanyarray(self._frames1.get_color_frame().get_data())
        img2 = np.asanyarray(self._frames2.get_color_frame().get_data())
        return img1, img2, ts_diff

# ==========================================

# Prepare 3D object points for the checkerboard
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints  = []
imgpoints1 = []
imgpoints2 = []

print("Starting cameras...")
pipe1 = create_pipeline(SERIAL_1)
pipe2 = create_pipeline(SERIAL_2)

# Warmup: let auto-exposure settle before calibrating
print("Warming up cameras (30 frames)...")
for _ in range(30):
    pipe1.wait_for_frames()
    pipe2.wait_for_frames()

K1, D1 = get_intrinsics(pipe1)
K2, D2 = get_intrinsics(pipe2)
print("Intrinsics loaded from device.")
print(f"K1:\n{K1}\nK2:\n{K2}\n")

grabber = FrameGrabber(pipe1, pipe2)
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

last_capture = 0
prev_corners1 = None
sync_rejects = 0

# Track recent positions to calculate velocity (stillness)
recent_corner_history = []
STILLNESS_THRESH = 3.0  # max pixels of movement allowed per frame to be considered "still"

print(f"\nAuto capture started. Need {MAX_SAMPLES} samples.")
print("Move the checkerboard to a new position, THEN HOLD IT COMPLETELY STILL.\n")

while True:

    img1, img2, ts_diff = grabber.grab()

    # Frame pair rejected due to timestamp gap
    if img1 is None:
        print("Frame missing")
        sync_rejects += 1
        if sync_rejects % 10 == 0:
            print(f"  [sync] {sync_rejects} frame pairs rejected (ts_diff={ts_diff:.1f}ms > {MAX_TIMESTAMP_DIFF_MS}ms)")
        continue

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, cb_flags)
    ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, cb_flags)

    vis1 = img1.copy()
    vis2 = img2.copy()

    if ret1:
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(vis1, CHECKERBOARD, corners1, ret1)

    if ret2:
        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(vis2, CHECKERBOARD, corners2, ret2)

    # ---- Build combined side-by-side display ----
    n = len(objpoints)
    now_disp = time.time()
    cooldown_remaining = max(0.0, COOLDOWN - (now_disp - last_capture))
    both_found = ret1 and ret2

    # Board detection border flash
    border_color1 = (0, 220, 80) if ret1 else (60, 60, 60)
    border_color2 = (0, 220, 80) if ret2 else (60, 60, 60)
    cv2.rectangle(vis1, (2, 2), (WIDTH - 2, HEIGHT - 2), border_color1, 3)
    cv2.rectangle(vis2, (2, 2), (WIDTH - 2, HEIGHT - 2), border_color2, 3)

    # ---- Zone guide overlay (only when board not found yet) ----
    # Divides each frame into a 3x2 grid showing where to aim
    if not ret1:
        for gx in range(3):
            for gy in range(2):
                zx1 = int(gx * WIDTH  / 3) + 10
                zy1 = int(gy * HEIGHT / 2) + 10
                zx2 = int((gx + 1) * WIDTH  / 3) - 10
                zy2 = int((gy + 1) * HEIGHT / 2) - 10
                cv2.rectangle(vis1, (zx1, zy1), (zx2, zy2), (80, 80, 80), 1)

    # ---- Per-camera status banner ----
    def draw_banner(img, cam_label, found):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, 36), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        status_color = (0, 220, 80) if found else (80, 80, 80)
        status_text  = "BOARD FOUND" if found else "searching..."
        cv2.putText(img, cam_label,   (8,  22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, status_text, (WIDTH // 2 - 55, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv2.LINE_AA)

    draw_banner(vis1, "Camera 1", ret1)
    draw_banner(vis2, "Camera 2", ret2)

    # ---- Cooldown arc on cam1 (shows when next capture is ready) ----
    if both_found and cooldown_remaining > 0:
        frac    = 1.0 - (cooldown_remaining / COOLDOWN)
        angle   = int(360 * frac)
        cx_arc, cy_arc, r_arc = WIDTH - 36, HEIGHT - 36, 22
        cv2.circle(vis1, (cx_arc, cy_arc), r_arc, (50, 50, 50), 3)
        cv2.ellipse(vis1, (cx_arc, cy_arc), (r_arc, r_arc), -90, 0, angle,
                    (0, 200, 255), 3, cv2.LINE_AA)
    elif both_found and cooldown_remaining == 0:
        # Ready to capture — solid green circle
        cv2.circle(vis1, (WIDTH - 36, HEIGHT - 36), 22, (0, 220, 80), -1)
        cv2.putText(vis1, "GO", (WIDTH - 48, HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)

    # ---- Bottom info directly on cameras ----
    prog_color = (0, 200, 80) if n < MAX_SAMPLES * 0.6 else \
                 (0, 180, 220) if n < MAX_SAMPLES * 0.9 else (80, 220, 80)
    
    cv2.putText(vis1, f"Samples: {n} / {MAX_SAMPLES}",
                (14, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, prog_color, 2, cv2.LINE_AA)
    
    hint = "Move slowly!" if both_found and cooldown_remaining > 0 else \
           "CAPTURING — hold still!" if both_found and cooldown_remaining == 0 else \
           "Show to BOTH cameras"
    
    cv2.putText(vis2, hint, (14, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    sync_str = f"Sync: {ts_diff:.0f}ms"
    sync_col = (0, 220, 80) if ts_diff < 20 else (0, 180, 255) if ts_diff < 40 else (0, 80, 255)
    cv2.putText(vis2, sync_str, (WIDTH - 150, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sync_col, 2, cv2.LINE_AA)

    cv2.imshow("cam1", vis1)
    cv2.imshow("cam2", vis2)

    now = now_disp  # reuse timestamp already taken above

    # ---- Check if board is currently moving ----
    is_still = False
    if ret1:
        recent_corner_history.append(corners1[0, 0])
        if len(recent_corner_history) > 5:
            recent_corner_history.pop(0)
            
        if len(recent_corner_history) == 5:
            # Calculate distance moved in the last 5 frames
            recent_movement = np.linalg.norm(recent_corner_history[-1] - recent_corner_history[0])
            if recent_movement < STILLNESS_THRESH:
                is_still = True

    if not ret1:
        recent_corner_history = []

    if ret1 and ret2 and (now - last_capture > COOLDOWN) and is_still:

        # Ensure both cameras agree on the checkerboard orientation.
        # OpenCV might return normal corners for Cam1 but 180-degree flipped corners for Cam2 
        # depending on perspective. We check the diagonal vector, and if they point in 
        # opposite directions, we flip Cam2's corners to match Cam1 perfectly.
        vec1 = corners1[-1, 0] - corners1[0, 0]
        vec2 = corners2[-1, 0] - corners2[0, 0]
        
        if np.dot(vec1, vec2) < 0:
            corners2 = corners2[::-1]

        if corners1.shape != corners2.shape:
            print("  [skip] Corner shape mismatch — skipping pair.")
            continue

        # Reject near-duplicate frames compared to very LAST CAPTURE (too little movement)
        if prev_corners1 is not None:
            total_movement_since_last_cap = np.linalg.norm(corners1[0, 0] - prev_corners1[0, 0])
            if total_movement_since_last_cap < MOVEMENT_THRESH:
                cv2.putText(vis1, "MOVE BOARD MORE", (14, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("cam1", vis1)
                continue

        print(f"  Captured sample {n + 1:2d}/{MAX_SAMPLES}  "
              f"(ts_diff={ts_diff:.1f}ms)")

        objpoints.append(objp)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)
        prev_corners1 = corners1.copy()
        last_capture = now

    if len(objpoints) >= MAX_SAMPLES:
        print(f"\nAll {MAX_SAMPLES} samples collected.")
        break

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        print("\nCapture interrupted by user.")
        break

pipe1.stop()
pipe2.stop()
cv2.destroyAllWindows()

if len(objpoints) < 10:
    print(f"ERROR: Only {len(objpoints)} samples — need at least 10. Exiting.")
    exit(1)

# ================= STEREO CALIBRATION =================

print(f"\nRunning stereo calibration on {len(objpoints)} samples...")

# CALIB_FIX_INTRINSIC: trust RealSense factory intrinsics, only solve for R and T
calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS

ret, K1_out, D1_out, K2_out, D2_out, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints1,
    imgpoints2,
    K1, D1,
    K2, D2,
    IMAGE_SIZE,
    criteria=(cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 100, 1e-6),
    flags=calib_flags
)

print("\n" + "="*50)
print(f"RMS reprojection error: {ret:.4f} px")
if ret < 0.5:
    print("Quality: EXCELLENT")
elif ret < 1.0:
    print("Quality: ACCEPTABLE — consider recapturing with more board positions")
else:
    print("Quality: POOR — recapture recommended (vary distance/angle more)")
print("="*50)

print(f"\nRotation matrix R:\n{R}")
print(f"\nTranslation vector T (meters):\n{T}")

# Euler angles for sanity check
def rotation_matrix_to_euler(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])

euler = rotation_matrix_to_euler(R)
print(f"\nCamera 2 relative to Camera 1:")
print(f"  Translation : X={T[0,0]*100:.1f}cm  Y={T[1,0]*100:.1f}cm  Z={T[2,0]*100:.1f}cm")
print(f"  Rotation    : Roll={euler[0]:.1f}°  Pitch={euler[1]:.1f}°  Yaw={euler[2]:.1f}°")

# ================= SAVE =================

np.savez(
    SAVE_FILE,
    R=R, T=T,
    K1=K1_out, K2=K2_out,
    D1=D1_out, D2=D2_out,
    E=E, F=F,
    rms=ret,
    image_size=IMAGE_SIZE,
    objpoints=np.array(objpoints, dtype=object),
    imgpoints1=np.array(imgpoints1, dtype=object),
    imgpoints2=np.array(imgpoints2, dtype=object)
)

print(f"\nCalibration saved to: {SAVE_FILE}")
print("\nLoad in your tracking script with:")
print("  data = np.load('stereo_calib.npz')")
print("  R, T, K1, K2, D1, D2 = data['R'], data['T'], data['K1'], data['K2'], data['D1'], data['D2']")
