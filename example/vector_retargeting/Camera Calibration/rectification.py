#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs

SERIAL_2 = "215322071654"
SERIAL_1 = "213622077408"
CALIB_FILE = "stereo_cam12_recalibrated.npz"

WIDTH, HEIGHT, FPS = 640, 480, 30

data = np.load(CALIB_FILE, allow_pickle=True)

K1 = np.array(data["K1"], dtype=np.float64)
K2 = np.array(data["K2"], dtype=np.float64)
D1 = np.array(data["D1"], dtype=np.float64).reshape(-1, 1)
D2 = np.array(data["D2"], dtype=np.float64).reshape(-1, 1)
R  = np.array(data["R"],  dtype=np.float64)
T  = np.array(data["T"],  dtype=np.float64).reshape(3, 1)

print("T norm:", np.linalg.norm(T))
print("K1:\n", K1)
print("K2:\n", K2)
print("D1:", D1.ravel())
print("D2:", D2.ravel())
print("R:\n", R)
print("T:\n", T)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, D1, K2, D2, (WIDTH, HEIGHT), R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=-1
)

print("roi1:", roi1)
print("roi2:", roi2)
print("P1:\n", P1)
print("P2:\n", P2)

map1x, map1y = cv2.initUndistortRectifyMap(
    K1, D1, R1, P1, (WIDTH, HEIGHT), cv2.CV_32FC1
)
map2x, map2y = cv2.initUndistortRectifyMap(
    K2, D2, R2, P2, (WIDTH, HEIGHT), cv2.CV_32FC1
)

def start_cam(serial):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipe.start(cfg)
    return pipe

pipe1 = start_cam(SERIAL_1)
pipe2 = start_cam(SERIAL_2)

try:
    while True:
        f1 = pipe1.wait_for_frames().get_color_frame()
        f2 = pipe2.wait_for_frames().get_color_frame()
        if not f1 or not f2:
            continue

        img1 = np.asanyarray(f1.get_data())
        img2 = np.asanyarray(f2.get_data())

        rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
        rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

        disp1 = rect1.copy()
        disp2 = rect2.copy()

        for y in range(0, HEIGHT, 40):
            cv2.line(disp1, (0, y), (WIDTH, y), (0, 255, 0), 1)
            cv2.line(disp2, (0, y), (WIDTH, y), (0, 255, 0), 1)

        cv2.imshow("Rectified 1", disp1)
        cv2.imshow("Rectified 2", disp2)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    pipe1.stop()
    pipe2.stop()
    cv2.destroyAllWindows()