import numpy as np
import cv2

# ================= LOAD =================
# data = np.load("stereo_calib.npz", allow_pickle=True)
data = np.load("stereo_calib.npz", allow_pickle=True)


R = data["R"]
T = data["T"]
K1 = data["K1"]
K2 = data["K2"]
D1 = data["D1"]
D2 = data["D2"]

objpoints = data["objpoints"]
imgpoints1 = data["imgpoints1"]
imgpoints2 = data["imgpoints2"]

print("Loaded calibration file.")

# ================= ERROR COMPUTATION =================

total_error1 = 0
total_error2 = 0
n_points = 0

bad_frames = []

for i in range(len(objpoints)):

    objp = objpoints[i].astype(np.float32)
    imgp1 = imgpoints1[i].reshape(-1, 2).astype(np.float32)
    imgp2 = imgpoints2[i].reshape(-1, 2).astype(np.float32)

    # Step 1: estimate pose for cam1
    ret, rvec1, tvec1 = cv2.solvePnP(
        objp, imgp1, K1, D1,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ret:
        print(f"Frame {i}: solvePnP failed")
        bad_frames.append(i)
        continue

    # Step 2: project back to cam1
    proj1, _ = cv2.projectPoints(objp, rvec1, tvec1, K1, D1)
    proj1 = proj1.reshape(-1, 2)

    # Step 3: transform pose to cam2
    R1, _ = cv2.Rodrigues(rvec1)

    R2 = R @ R1
    tvec2 = R @ tvec1 + T

    rvec2, _ = cv2.Rodrigues(R2)

    # Step 4: project to cam2
    proj2, _ = cv2.projectPoints(objp, rvec2, tvec2, K2, D2)
    proj2 = proj2.reshape(-1, 2)

    # Compute per-point mean error
    err1 = np.mean(np.linalg.norm(imgp1 - proj1, axis=1))
    err2 = np.mean(np.linalg.norm(imgp2 - proj2, axis=1))

    print(f"Frame {i}: Cam1={err1:.3f}px | Cam2={err2:.3f}px")

    if err1 > 2 or err2 > 2:
        bad_frames.append(i)

    total_error1 += err1 * len(objp)
    total_error2 += err2 * len(objp)
    n_points += len(objp)

# ================= FINAL RESULT =================

mean_error1 = total_error1 / n_points
mean_error2 = total_error2 / n_points

print("\n===== FINAL RESULT =====")
print(f"Camera 1 mean error: {mean_error1:.3f} px")
print(f"Camera 2 mean error: {mean_error2:.3f} px")

# ================= QUALITY CHECK =================

if mean_error1 < 1 and mean_error2 < 1:
    print("Calibration quality: GOOD")
elif mean_error1 < 2 and mean_error2 < 2:
    print("Calibration quality: OK (usable)")
else:
    print("Calibration quality: BAD (redo recommended)")

# ================= BAD FRAMES =================

if len(bad_frames) > 0:
    print("\nBad frames detected:")
    print(bad_frames)
else:
    print("\nNo bad frames detected.")