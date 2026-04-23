import numpy as np
import cv2

# ================= LOAD =================
data = np.load("multi_cam_calib.npz", allow_pickle=True)

R12 = data["R12"]
T12 = data["T12"]

R13 = data["R13"]
T13 = data["T13"]

K1 = data["K1"]
K2 = data["K2"]
K3 = data["K3"]

D1 = data["D1"]
D2 = data["D2"]
D3 = data["D3"]

objpoints = data["objpoints"]
imgpoints1 = data["imgpoints1"]
imgpoints2 = data["imgpoints2"]
imgpoints3 = data["imgpoints3"]

print("Loaded calibration file.")

# ================= ERROR COMPUTATION =================

total_error1 = 0
total_error2 = 0
total_error3 = 0
n_points = 0

bad_frames = []

for i in range(len(objpoints)):

    objp = objpoints[i].astype(np.float32)
    imgp1 = imgpoints1[i].reshape(-1, 2).astype(np.float32)
    imgp2 = imgpoints2[i].reshape(-1, 2).astype(np.float32)
    imgp3 = imgpoints3[i].reshape(-1, 2).astype(np.float32)

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

    # Convert to rotation matrix
    R1, _ = cv2.Rodrigues(rvec1)

    # ================= CAM2 =================
    R2 = R12 @ R1
    tvec2 = R12 @ tvec1 + T12
    rvec2, _ = cv2.Rodrigues(R2)

    proj2, _ = cv2.projectPoints(objp, rvec2, tvec2, K2, D2)
    proj2 = proj2.reshape(-1, 2)

    # ================= CAM3 =================
    R3 = R13 @ R1
    tvec3 = R13 @ tvec1 + T13
    rvec3, _ = cv2.Rodrigues(R3)

    proj3, _ = cv2.projectPoints(objp, rvec3, tvec3, K3, D3)
    proj3 = proj3.reshape(-1, 2)

    # ================= ERRORS =================
    err1 = np.mean(np.linalg.norm(imgp1 - proj1, axis=1))
    err2 = np.mean(np.linalg.norm(imgp2 - proj2, axis=1))
    err3 = np.mean(np.linalg.norm(imgp3 - proj3, axis=1))

    print(f"Frame {i}: Cam1={err1:.3f}px | Cam2={err2:.3f}px | Cam3={err3:.3f}px")

    if err1 > 2 or err2 > 2 or err3 > 2:
        bad_frames.append(i)

    total_error1 += err1 * len(objp)
    total_error2 += err2 * len(objp)
    total_error3 += err3 * len(objp)
    n_points += len(objp)

# ================= FINAL RESULT =================

mean_error1 = total_error1 / n_points
mean_error2 = total_error2 / n_points
mean_error3 = total_error3 / n_points

print("\n===== FINAL RESULT =====")
print(f"Camera 1 mean error: {mean_error1:.3f} px")
print(f"Camera 2 mean error: {mean_error2:.3f} px")
print(f"Camera 3 mean error: {mean_error3:.3f} px")

# ================= QUALITY CHECK =================

if mean_error1 < 1 and mean_error2 < 1 and mean_error3 < 1:
    print("Calibration quality: GOOD")
elif mean_error1 < 2 and mean_error2 < 2 and mean_error3 < 2:
    print("Calibration quality: OK (usable)")
else:
    print("Calibration quality: BAD (redo recommended)")

# ================= BAD FRAMES =================

if len(bad_frames) > 0:
    print("\nBad frames detected:")
    print(bad_frames)
else:
    print("\nNo bad frames detected.")