#!/usr/bin/env python3

import cv2
import csv
import time
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp

# ================= CONFIG =================

SERIAL_1 = "215322071654"   # Cam1
SERIAL_2 = "213622077408"   # Cam3

CALIB_FILE = "stereo_fixed_intrinsics_cam12.npz"

WIDTH, HEIGHT, FPS = 640, 480, 30

MIN_HAND_CONF = 0.5
MIN_FRAME_SCORE = 0.025

SAVE_LOG = True
LOG_CSV = "multiview_hand_tracking_log.csv"


DRAW_IDS = True
PRINT_EVERY_N_FRAMES = 1

TRACKED_HAND = "Left"   # or "Left"

OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OPERATOR2MANO_LEFT  = np.array([[0, 0, -1], [ 1, 0, 0], [0,-1, 0]])

operator2mano = OPERATOR2MANO_RIGHT if TRACKED_HAND == "Right" else OPERATOR2MANO_LEFT
# ==========================================


# ================= LOAD CALIBRATION =================

data = np.load(CALIB_FILE, allow_pickle=True)

K1 = np.array(data["K1"], dtype=np.float64)
K2 = np.array(data["K2"], dtype=np.float64)
D1 = np.array(data["D1"], dtype=np.float64).reshape(-1, 1)
D2 = np.array(data["D2"], dtype=np.float64).reshape(-1, 1)
R = np.array(data["R"], dtype=np.float64)
T = np.array(data["T"], dtype=np.float64).reshape(3, 1)

print("Loaded calibration.")
print("Baseline (m):", np.linalg.norm(T))

# Projection matrices
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K2 @ np.hstack((R, T))


# ================= MEDIAPIPE =================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands1 = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands2 = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)




# ================= REALSENSE =================

def start_cam(serial: str):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipe.start(cfg)
    return pipe


# ================= HELPERS =================



def triangulate_points(pts1, pts2, P1, P2):
    """
    pts1, pts2: (N,2)
    returns pts3D: (N,3)
    """
    pts1_h = pts1.T
    pts2_h = pts2.T
    pts4D = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    pts3D = (pts4D[:3] / pts4D[3]).T
    return pts3D


def project_points(P, pts3D):
    """
    P: (3,4), pts3D: (N,3)
    returns projected 2D: (N,2)
    """
    pts3D_h = np.hstack([pts3D, np.ones((pts3D.shape[0], 1))])  # (N,4)
    proj = (P @ pts3D_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


def reprojection_error_per_point(P, pts3D, pts2D):
    proj = project_points(P, pts3D)
    err = np.linalg.norm(pts2D - proj, axis=1)
    return err, proj


def confidence_to_color(c):
    if c > 0.7:
        return (0, 255, 0)      # green
    if c > 0.4:
        return (0, 255, 255)    # yellow
    return (0, 0, 255)          # red


def draw_landmarks_with_confidence(image, pts, conf, ids=False):
    out = image.copy()
    if pts is None:
        return out

    for i, p in enumerate(pts):
        x, y = int(round(p[0])), int(round(p[1]))
        color = confidence_to_color(conf[i])
        cv2.circle(out, (x, y), 4, color, -1)
        if ids:
            cv2.putText(
                out, str(i), (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
            )
    return out

# def draw_triangulated_landmarks(image, proj_pts, conf):
#     out = image.copy()
#     for i, p in enumerate(proj_pts):
#         x, y = int(round(p[0])), int(round(p[1]))
#         color = confidence_to_color(conf[i])
#         cv2.circle(out, (x, y), 9, color, 1)  # hollow circle
#     return out
def draw_triangulated_landmarks(image, proj_pts, conf, min_conf=0.15):
    out = image.copy()
    for i, p in enumerate(proj_pts):
        if conf[i] < min_conf:
            continue
        x, y = int(round(p[0])), int(round(p[1]))
        color = confidence_to_color(conf[i])
        cv2.circle(out, (x, y), 9, color, 1)
    return out

def draw_hand_skeleton(image, results, tracked_hand):
    out = image.copy()

    if not results or not results.multi_hand_landmarks or not results.multi_handedness:
        return out

    for i, handedness in enumerate(results.multi_handedness):
        label = handedness.classification[0].label

        # IMPORTANT: match the SAME logic as tracker
        if label == tracked_hand:
            mp_drawing.draw_landmarks(
                out,
                results.multi_hand_landmarks[i],
                mp_hands.HAND_CONNECTIONS
            )
            return out

    # if wrong hand → draw nothing
    return out


def format_array_short(arr, precision=2):
    return "[" + ", ".join(f"{x:.{precision}f}" for x in arr) + "]"

# def compute_joint_pos(pts3D, operator2mano, prev_R=None, prev_joint=None):

#     pts3D = pts3D - pts3D[0:1]
#     # ---- normalize scale (CRITICAL) ----
#     scale = np.linalg.norm(pts3D[9] - pts3D[0]) + 1e-6   # wrist → middle MCP
#     pts3D = pts3D / scale

#     v1 = pts3D[5] - pts3D[0]
#     v2 = pts3D[17] - pts3D[0]

#     x = v1 / (np.linalg.norm(v1) + 1e-6)
#     z = np.cross(v1, v2)
#     z = z / (np.linalg.norm(z) + 1e-6)
#     y = np.cross(z, x)

#     R_new = np.stack([x, y, z], axis=1)

#     # ---- smooth rotation ----
#     if prev_R is None:
#         R = R_new
#     else:
#         # R = 0.9 * prev_R + 0.1 * R_new
#         R = 0.7 * prev_R + 0.3 * R_new


#     # ---- map to joint space ----
#     joint_pos = pts3D @ R @ operator2mano

#     # ---- smooth joints ----
#     if prev_joint is not None:
#         joint_pos = 0.6 * prev_joint + 0.4 * joint_pos

#     return joint_pos.astype(np.float32), R

# ================= LOGGING =================

csv_file = None
csv_writer = None

if SAVE_LOG:
    csv_file = open(LOG_CSV, "w", newline="")
    csv_writer = csv.writer(csv_file)

    header = [
        "timestamp",
        "conf_cam1",
        "conf_cam2",
        "frame_score",
        "mean_reproj_cam1",
        "mean_reproj_cam2",
        "gt_valid",
    ]

    for i in range(21):
        header += [
            f"lm{i}_x", f"lm{i}_y", f"lm{i}_z",
            f"c1_lm{i}_x", f"c1_lm{i}_y",
            f"c2_lm{i}_x", f"c2_lm{i}_y",
            f"lm{i}_conf",
            f"lm{i}_err1",
            f"lm{i}_err2",
        ]

    csv_writer.writerow(header)

def make_kf_3d(dt=1.0, q=1e-4, r=5e-3):
    """Constant-velocity Kalman filter for one 3D landmark."""
    kf = cv2.KalmanFilter(6, 3)
    kf.transitionMatrix = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ], dtype=np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * q
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * r
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    return kf

class MultiViewTracker:
    def __init__(self,hand_type):
        hands1 = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        hands2 = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hand_type = hand_type
        self.prev_R = None
        self.prev_joint = None
        selfie = False
        self.expected_labelD = hand_type
        self.kf3d = [make_kf_3d() for _ in range(21)]
        self.kf_initialized = False

        # self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )

        # ---- MediaPipe handedness mapping (CRITICAL) ----
        # For non-selfie cameras, MediaPipe flips labels
        inverse = {"Right": "Left", "Left": "Right"}
        self.expected_label = inverse[hand_type]

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """Exact copy from SingleHandDetector — MUST match for retargeting to work."""
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]  # wrist, index MCP, middle MCP

        x_vector = points[0] - points[2]

        pts = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(pts)
        normal = v[2, :]

        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame
    
    def compute_joint_pos(self, pts3D):
        pts3D = pts3D - pts3D[0:1]

        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(pts3D)

        joint_pos = pts3D @ mediapipe_wrist_rot @ self.operator2mano

        if self.prev_joint is not None:
            joint_pos = 0.6 * self.prev_joint + 0.4 * joint_pos

        self.prev_joint = joint_pos
        return joint_pos.astype(np.float32), mediapipe_wrist_rot

   
    def get_landmarks_and_visibility(self,image, hands):
        """
        Returns:
            pts: (21, 2) pixel coordinates or None
            vis: (21,) per-landmark visibility/confidence proxy
            hand_conf: global hand confidence
            results: raw MediaPipe results
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            # return None, None, 0.0, results
            return None, None, 0.0, results
    


        selected_idx = -1

        if results.multi_handedness:
            for i, handedness in enumerate(results.multi_handedness):
                label = results.multi_handedness[i].ListFields()[0][1][0].label
                # label = handedness.classification[0].label
                if label == self.expected_label:
                    selected_idx = i
                    break

        # 🚨 No correct hand found → reject frame
        if selected_idx == -1:
            return None, None, 0.0, results

        hand_landmarks = results.multi_hand_landmarks[selected_idx]
        cls = results.multi_handedness[selected_idx].classification[0]
        hand_conf = cls.score


        hand_label = cls.label
        # hand_conf = cls.score
        # 🚨 HARD FILTER: reject wrong hand
        if hand_label != self.expected_label:
            return None, None, 0.0, results

        # MediaPipe Hands does not always provide a robust per-landmark visibility
        # like Pose does, so we build a proxy confidence from:
        # 1) global handedness score
        # 2) whether landmark is inside image bounds
        # 3) normalized z stability can be added later if needed
        # hand_label = None

        h, w = image.shape[:2]
        pts = []
        vis = []

        for lm in hand_landmarks.landmark:
            x = lm.x * w
            y = lm.y * h
            pts.append([x, y])

            in_bounds = 1.0 if (0 <= x < w and 0 <= y < h) else 0.0

            margin = min(x, w - 1 - x, y, h - 1 - y)
            edge_score = np.clip(margin / 30.0, 0.0, 1.0)

            lm_conf = hand_conf * in_bounds * edge_score
            vis.append(lm_conf)

        return np.array(pts, dtype=np.float64), np.array(vis, dtype=np.float64), float(hand_conf), results
        # return np.array(pts, dtype=np.float64), np.array(vis, dtype=np.float64), float(hand_conf), hand_label, results
    def track_pts3D(self, pts3D_meas, meas_conf):
        """
        pts3D_meas: (21,3) current triangulated landmarks
        meas_conf:  (21,) confidence per landmark in [0,1]
        returns tracked_pts3D: (21,3)
        """
        tracked = np.zeros_like(pts3D_meas, dtype=np.float32)

        # first valid frame: initialize all filters from measurement
        if not self.kf_initialized:
            for i in range(21):
                x, y, z = pts3D_meas[i].astype(np.float32)
                self.kf3d[i].statePost[:3, 0] = np.array([x, y, z], dtype=np.float32)
                self.kf3d[i].statePost[3:, 0] = 0.0
                tracked[i] = np.array([x, y, z], dtype=np.float32)
            self.kf_initialized = True
            return tracked

        for i in range(21):
            kf = self.kf3d[i]
            pred = kf.predict()[:3, 0].copy()

            c = float(meas_conf[i])

            # adapt measurement noise to confidence
            # high confidence -> trust measurement more
            r_scale = 5e-3 if c > 0.5 else (2e-2 if c > 0.15 else 1e-1)
            kf.measurementNoiseCov[:] = np.eye(3, dtype=np.float32) * r_scale

            if c > 0.05:
                z = pts3D_meas[i].astype(np.float32).reshape(3, 1)
                kf.correct(z)
                tracked[i] = kf.statePost[:3, 0]
            else:
                tracked[i] = pred

        return tracked

    # def process(self, img1, img2, pts1, pts2, vis1, vis2):

    #     if pts1 is None or pts2 is None:
    #         return None

    #     pts3D = triangulate_points(pts1, pts2, P1, P2)
    #     # pts3D = np.zeros((21, 3))

    #     # err1, _ = reprojection_error_per_point(P1, triangulate_points(pts1, pts2, P1, P2), pts1)
    #     # err2, _ = reprojection_error_per_point(P2, triangulate_points(pts1, pts2, P1, P2), pts2)

    #     # geom_conf1 = np.exp(-err1 / 10.0)
    #     # geom_conf2 = np.exp(-err2 / 10.0)

    #     # final_conf = np.zeros(21)

    #     # for i in range(21):

    #     #     c1 = vis1[i] * geom_conf1[i]
    #     #     c2 = vis2[i] * geom_conf2[i]

    #     #     # ---- both good → triangulate ----
    #     #     if c1 > 0.3 and c2 > 0.3:
    #     #         pt3d = triangulate_points(
    #     #             pts1[i:i+1], pts2[i:i+1], P1, P2
    #     #         )[0]

    #     #         pts3D[i] = pt3d
    #     #         final_conf[i] = min(c1, c2)

    #     #     # ---- cam1 only ----
    #     #     elif c1 > c2:
    #     #         pts3D[i] = np.array([pts1[i, 0]/1000, pts1[i, 1]/1000, 0.5])
    #     #         final_conf[i] = c1 * 0.5  # penalize monocular

    #     #     # ---- cam2 only ----
    #     #     else:
    #     #         pts3D[i] = np.array([pts2[i, 0]/1000, pts2[i, 1]/1000, 0.5])
    #     #         final_conf[i] = c2 * 0.5


    #     # joint_pos_new, R_new = compute_joint_pos(
    #     #     pts3D,
    #     #     self.operator2mano,
    #     #     prev_R=self.prev_R,
    #     #     prev_joint=self.prev_joint
    #     # )
    #     joint_pos_new, R_new = self.compute_joint_pos(pts3D) 

    #     # ---- confidence gating (keep your logic) ----
    #     err1, _ = reprojection_error_per_point(P1, pts3D, pts1)
    #     err2, _ = reprojection_error_per_point(P2, pts3D, pts2)

    #     geom_conf1 = np.exp(-err1 / 10.0)
    #     geom_conf2 = np.exp(-err2 / 10.0)

    #     landmark_conf = np.minimum(vis1, vis2)
    #     final_conf = landmark_conf * np.minimum(geom_conf1, geom_conf2)

    #     frame_score = float(np.mean(final_conf))
    #     print ("frame score",frame_score)

    #     if frame_score > MIN_FRAME_SCORE:
    #         joint_pos = joint_pos_new
    #         self.prev_joint = joint_pos
    #         self.prev_R = R_new
    #     else:
    #         joint_pos = self.prev_joint

    #     # stricter GT flag
    #     gt_valid = (
    #         frame_score > 0.4 and
    #         np.mean(err1) < 5 and
    #         np.mean(err2) < 5
    #     )

    #     return joint_pos, pts3D, frame_score, gt_valid, final_conf, err1, err2
    
    def process(self, img1, img2, pts1, pts2, vis1, vis2):

        if pts1 is None or pts2 is None:
            return None

        # 1) full coherent stereo reconstruction
        pts3D_meas = triangulate_points(pts1, pts2, P1, P2)

        # 2) confidence from reprojection + visibility
        err1, _ = reprojection_error_per_point(P1, pts3D_meas, pts1)
        err2, _ = reprojection_error_per_point(P2, pts3D_meas, pts2)

        geom_conf1 = np.exp(-err1 / 10.0)
        geom_conf2 = np.exp(-err2 / 10.0)

        landmark_conf = np.minimum(vis1, vis2)
        final_conf = landmark_conf * np.minimum(geom_conf1, geom_conf2)

        # 3) temporal tracking on 3D landmarks
        pts3D = self.track_pts3D(pts3D_meas, final_conf)

        # 4) compute joint space from tracked 3D
        joint_pos_new, R_new = self.compute_joint_pos(pts3D)

        frame_score = float(np.mean(final_conf))
        # print("frame score", frame_score)

        if frame_score > MIN_FRAME_SCORE:
            joint_pos = joint_pos_new
            self.prev_joint = joint_pos
            self.prev_R = R_new
            self.prev_pts3D = pts3D.copy()
        else:
            # use tracked/predicted pose instead of snapping backward
            joint_pos = self.prev_joint if self.prev_joint is not None else joint_pos_new

        print(f"DEBUG: score={frame_score:.3f}, err1={np.mean(err1):.1f}, err2={np.mean(err2):.1f}")
        gt_valid = (
            frame_score > 0.5 and
            np.mean(err1) < 5 and
            np.mean(err2) < 5
        )
        



        return joint_pos, pts3D, frame_score, gt_valid, final_conf, err1, err2
    
# ================= MAIN =================
if __name__ == "__main__":

    pipe1 = start_cam(SERIAL_1)
    pipe2 = start_cam(SERIAL_2)

    frame_idx = 0

    print("Running multi-view tracking. Press ESC to quit.")
    lag_buffer1 = []
    lag_buffer2 = []
    WINDOW = 50
    sync_buffer1 = []   # (ts, pts, vis, conf, res)
    sync_buffer2 = []

    SYNC_MAX_BUF = 10
    SYNC_MAX_DT = 20.0  # ms
    prev_R = None
    prev_joint = None
    tracker = MultiViewTracker(hand_type=TRACKED_HAND)

    try:
        while True:
            frames1 = pipe1.wait_for_frames()
            frames2 = pipe2.wait_for_frames()
            ts1 = frames1.get_timestamp()
            ts2 = frames2.get_timestamp()

            dt = ts1 - ts2
            print(f"Timestamp diff: {dt:.2f} ms")
            color1 = frames1.get_color_frame()
            color2 = frames2.get_color_frame()

            if not color1 or not color2:
                continue

            img1 = np.asanyarray(color1.get_data())
            img2 = np.asanyarray(color2.get_data())
           

            pts1, vis1, hand_conf1, res1 = tracker.get_landmarks_and_visibility(img1, hands1)
            pts2, vis2, hand_conf2, res2 = tracker.get_landmarks_and_visibility(img2, hands2)



            # Detect if the WRONG hand is actively in frame (hand present but filtered out)
            def wrong_hand_in_frame(results, expected_label):
                if not results or not results.multi_hand_landmarks or not results.multi_handedness:
                    return False
                for i, handedness in enumerate(results.multi_handedness):
                    label = handedness.classification[0].label
                    if label != expected_label:  # a hand exists but it's the wrong one
                        return True
                return False

            if pts1 is not None:
                sync_buffer1.append((ts1, pts1, vis1, hand_conf1, res1))
            elif wrong_hand_in_frame(res1, tracker.expected_label):
                sync_buffer1.clear()  # wrong hand is blocking — flush stale good-hand data
            # else: no hand at all → also clear (already handled)
            else:
                sync_buffer1.clear()

            if pts2 is not None:
                sync_buffer2.append((ts2, pts2, vis2, hand_conf2, res2))
            elif wrong_hand_in_frame(res2, tracker.expected_label):
                sync_buffer2.clear()
            else:
                sync_buffer2.clear()
            # After appending, prune entries older than e.g. 200ms
            current_ts = max(ts1, ts2)
            sync_buffer1 = [(t, *rest) for t, *rest in sync_buffer1 if current_ts - t < 200.0]
            sync_buffer2 = [(t, *rest) for t, *rest in sync_buffer2 if current_ts - t < 200.0]
            # keep buffers small

            if len(sync_buffer1) > SYNC_MAX_BUF:
                sync_buffer1.pop(0)
            if len(sync_buffer2) > SYNC_MAX_BUF:
                sync_buffer2.pop(0)

            disp1 = img1.copy()
            disp2 = img2.copy()
            if pts1 is not None:
                disp1 = draw_hand_skeleton(disp1, res1, tracker.expected_label)

            if pts2 is not None:
                disp2 = draw_hand_skeleton(disp2, res2, tracker.expected_label)

            frame_score = 0.0
            mean_err1 = np.nan
            mean_err2 = np.nan
            # ---- find best timestamp match ----
            best_pair = None
            best_dt = float("inf")

            for t1, p1, v1, c1, r1 in sync_buffer1[-10:]:
                for t2, p2, v2, c2, r2 in sync_buffer2[-10:]:
                    dt_match = abs(t1 - t2)
                    # dt_match = t2 - t1
                    if 0 <= dt_match <= SYNC_MAX_DT and dt_match < best_dt:
                        best_dt = dt_match
                        # best_pair = (p1, v1, c1, r1, p2, v2, c2, r2)
                        best_pair = (t1, p1, v1, c1, r1, t2, p2, v2, c2, r2)
                        print(f"Found sync pair with dt={dt_match:.2f} ms | C1={c1:.2f} C2={c2:.2f}")

            # if pts1 is not None:
            #     sync_buffer1.append((ts1, pts1, vis1, hand_conf1, res1))
            # elif wrong_hand_in_frame(res1, tracker.expected_label):
            #     sync_buffer1.clear()
            # else:
            #     sync_buffer1.clear()

            # if pts2 is not None:
            #     sync_buffer2.append((ts2, pts2, vis2, hand_conf2, res2))
            # elif wrong_hand_in_frame(res2, tracker.expected_label):
            #     sync_buffer2.clear()
            # else:
            #     sync_buffer2.clear()

         

            if best_pair is not None:
                t1, pts1, vis1, hand_conf1, res1, \
                t2, pts2, vis2, hand_conf2, res2 = best_pair

                if hand_conf1 < MIN_HAND_CONF or hand_conf2 < MIN_HAND_CONF:
                    continue

                joint_pos, pts3D, frame_score, gt_valid, final_conf, err1, err2 = tracker.process(img1, img2, pts1, pts2, vis1, vis2)

                
                # pts3D= triangulate_points(pts1, pts2, P1, P2)

                #             # ---- 3D → kinematics ----
                # joint_pos_new, R_new = compute_joint_pos(
                #     pts3D,
                #     operator2mano,
                #     prev_R=prev_R,
                #     prev_joint=prev_joint
                # )

                # # ---- compute confidence first ----
                # err1, proj1 = reprojection_error_per_point(P1, pts3D, pts1)
                # err2, proj2 = reprojection_error_per_point(P2, pts3D, pts2)

                # geom_conf1 = np.exp(-err1 / 10.0)
                # geom_conf2 = np.exp(-err2 / 10.0)

                # landmark_conf = np.minimum(vis1, vis2)
                # final_conf = landmark_conf * np.minimum(geom_conf1, geom_conf2)

                # frame_score = float(np.mean(final_conf))

                # # ---- ONLY update if frame is good ----
                # if frame_score >= MIN_FRAME_SCORE:
                #     joint_pos = joint_pos_new
                #     R = R_new
                #     prev_R = R
                #     prev_joint = joint_pos
                # else:
                #     if prev_joint is not None:
                #         joint_pos = prev_joint
                #         R = prev_R
                #     else:
                #         joint_pos = joint_pos_new
                #         R = R_new
                #         prev_R = R
                #         prev_joint = joint_pos
                # geom_conf1 = np.exp(-err1 / 10.0)
                # geom_conf2 = np.exp(-err2 / 10.0)

                # landmark_conf = np.minimum(vis1, vis2)
                # final_conf = landmark_conf * np.minimum(geom_conf1, geom_conf2)

                # frame_score = float(np.mean(final_conf))
                # if frame_score < MIN_FRAME_SCORE:
                #     # skip bad frames → prevents jumps
                #     continue

                mean_err1 = float(np.mean(err1))
                mean_err2 = float(np.mean(err2))
                proj1 = project_points(P1, pts3D)
                proj2 = project_points(P2, pts3D)
                disp1 = draw_landmarks_with_confidence(disp1, pts1, vis1, ids=DRAW_IDS)
                disp2 = draw_landmarks_with_confidence(disp2, pts2, vis2, ids=DRAW_IDS)
                disp1 = draw_triangulated_landmarks(disp1, proj1, final_conf)
                disp2 = draw_triangulated_landmarks(disp2, proj2, final_conf)


                z_tip = pts3D[8, 2]   # index fingertip
                wrist = pts3D[0]

                text1 = f"C1={hand_conf1:.2f}  C2={hand_conf2:.2f}  Score={frame_score:.2f}"
                text2 = f"Err1={mean_err1:.2f}px  Err2={mean_err2:.2f}px  Z_tip={z_tip:.3f}"

                cv2.putText(disp1, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(disp1, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                cv2.putText(
                    disp2,
                    f"Wrist3D=({wrist[0]:.3f}, {wrist[1]:.3f}, {wrist[2]:.3f})",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                    print(
                        f"Frame {frame_idx} | "
                        f"C1={hand_conf1:.2f} C2={hand_conf2:.2f} | "
                        f"Score={frame_score:.2f} | "
                        f"Err1={mean_err1:.2f}px Err2={mean_err2:.2f}px"
                    )

                if SAVE_LOG:
                    row = [
                        time.time(),
                        hand_conf1,
                        hand_conf2,
                        frame_score,
                        mean_err1,
                        mean_err2,
                        int(gt_valid),
                    ]
                    for i in range(21):
                        row += [
                            float(pts3D[i, 0]),
                            float(pts3D[i, 1]),
                            float(pts3D[i, 2]),
                            float(pts1[i, 0]),
                            float(pts1[i, 1]),
                            float(pts2[i, 0]),
                            float(pts2[i, 1]),
                            float(final_conf[i]),
                            float(err1[i]),
                            float(err2[i]),
                        ]
                    csv_writer.writerow(row)

            else:
                cv2.putText(
                    disp1,
                    f"No reliable two-view hand | C1={hand_conf1:.2f} C2={hand_conf2:.2f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

                if frame_idx % PRINT_EVERY_N_FRAMES == 0:
                    print(
                        f"Frame {frame_idx} | "
                        f"No reliable two-view hand | C1={hand_conf1:.2f} C2={hand_conf2:.2f}"
                    )

            # Overall frame quality flag
            quality_text = "GOOD" if frame_score >= MIN_FRAME_SCORE else "LOW"
            quality_color = (0, 255, 0) if frame_score >= MIN_FRAME_SCORE else (0, 0, 255)

            cv2.putText(
                disp1,
                f"Frame quality: {quality_text}",
                (20, HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                quality_color,
                2
            )

            cv2.imshow("Cam1 Multi-view", disp1)
            cv2.imshow("Cam2 Multi-view", disp2)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            frame_idx += 1

    finally:
        pipe1.stop()
        pipe2.stop()
        hands1.close()
        hands2.close()
        cv2.destroyAllWindows()

        if csv_file is not None:
            csv_file.close()

