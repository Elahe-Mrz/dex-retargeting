import numpy as np
import cv2

data = np.load('/home/ucml/dex-retargeting/example/vector_retargeting/Camera Calibration/stereo_calib12_1.npz', allow_pickle=True)
K1, K2, D1, D2 = data['K1'], data['K2'], data['D1'], data['D2']
objp = list(data['objpoints'])
imgp1 = list(data['imgpoints1'])
imgp2 = list(data['imgpoints2'])
IMAGE_SIZE = tuple(data['image_size'])

all_T_A, all_T_B = [], []

for i in range(len(objp)):
    # Reshape for exact OpenCV types
    o = np.array(objp[i], dtype=np.float32)
    p1 = np.array(imgp1[i], dtype=np.float32)
    p2 = np.array(imgp2[i], dtype=np.float32)

    _, r1, t1 = cv2.solvePnP(o, p1, K1, D1)
    R1, _ = cv2.Rodrigues(r1)

    # Option A: Original
    _, r2A, t2A = cv2.solvePnP(o, p2, K2, D2)
    R2A, _ = cv2.Rodrigues(r2A)
    T_A = t2A - R2A @ R1.T @ t1
    all_T_A.append(T_A)

    # Option B: Flipped point order (180 deg rigid rotate)
    _, r2B, t2B = cv2.solvePnP(o, p2[::-1], K2, D2)
    R2B, _ = cv2.Rodrigues(r2B)
    T_B = t2B - R2B @ R1.T @ t1
    all_T_B.append(T_B)

all_T_A = np.array(all_T_A).reshape(len(objp), 3)
all_T_B = np.array(all_T_B).reshape(len(objp), 3)

# Test Hypothesis 1: Assume Frame 0's Option A is the truth.
choices_H1 = []
T_H1 = []
ref_T = all_T_A[0]
for i in range(len(objp)):
    if np.linalg.norm(all_T_A[i] - ref_T) < np.linalg.norm(all_T_B[i] - ref_T):
        choices_H1.append(0)
        T_H1.append(all_T_A[i])
    else:
        choices_H1.append(1)
        T_H1.append(all_T_B[i])

# Test Hypothesis 2: Assume Frame 0's Option B is the truth.
choices_H2 = []
T_H2 = []
ref_T = all_T_B[0]
for i in range(len(objp)):
    if np.linalg.norm(all_T_A[i] - ref_T) < np.linalg.norm(all_T_B[i] - ref_T):
        choices_H2.append(0)
        T_H2.append(all_T_A[i])
    else:
        choices_H2.append(1)
        T_H2.append(all_T_B[i])

var_H1 = np.var(T_H1, axis=0).sum()
var_H2 = np.var(T_H2, axis=0).sum()

if var_H1 < var_H2:
    print(f"Chose Hypothesis 1 (Variance = {var_H1:.5f}m vs {var_H2:.5f}m)")
    final_choices = choices_H1
else:
    print(f"Chose Hypothesis 2 (Variance = {var_H2:.5f}m vs {var_H1:.5f}m)")
    final_choices = choices_H2

# Correct the imgp2 array based on choices
fixed_imgp2 = []
flipped_count = 0
for i, choice in enumerate(final_choices):
    if choice == 0:
        fixed_imgp2.append(np.array(imgp2[i], dtype=np.float32))
    else:
        fixed_imgp2.append(np.array(imgp2[i][::-1], dtype=np.float32))
        flipped_count += 1

print(f"Flipped points in {flipped_count} out of {len(objp)} frames to match stereoscopic consistency.")

objpoints_clean = [np.array(o, dtype=np.float32) for o in objp]
imgpoints1_clean = [np.array(p, dtype=np.float32) for p in imgp1]

flag = cv2.CALIB_FIX_INTRINSIC
ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints_clean, imgpoints1_clean, fixed_imgp2, K1, D1, K2, D2, IMAGE_SIZE,
    criteria=(cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 100, 1e-6),
    flags=flag
)

print(f"NEW RMS ERROR: {ret:.4f} pixels")
print(f"NEW D1 Optimized: {D1.ravel()}")
print(f"NEW D2 Optimized: {D2.ravel()}")
print(f"NEW T:\n{T}")

print(f"SUCCESS: Saving directly to stereo_calib.npz")
np.savez('stereo_calib12_1_recalib_fixed.npz',
         R=R, T=T, K1=K1, K2=K2, D1=D1, D2=D2, E=E, F=F, rms=ret,
         image_size=IMAGE_SIZE,
         objpoints=np.array(objp, dtype=object),
         imgpoints1=np.array(imgp1, dtype=object),
         imgpoints2=np.array(fixed_imgp2, dtype=object))
