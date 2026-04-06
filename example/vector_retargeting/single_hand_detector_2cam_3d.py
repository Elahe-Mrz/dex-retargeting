
# ---- minimal, compatible upgrade to SingleHandDetector ----
# 1) Force CPU graph before importing mediapipe to avoid EGL/GPU crashes.
import os as _os
if _os.environ.get("MEDIAPIPE_DISABLE_GPU") is None:
    _os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark
import pyrealsense2 as rs

# --- unchanged transforms ---
OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OPERATOR2MANO_LEFT  = np.array([[0, 0, -1], [ 1, 0, 0], [0,-1, 0]])

WIDTH, HEIGHT, FPS = 640, 480, 30

# Thumb indices in MediaPipe order
_TH_CMC, _TH_MCP, _TH_IP, _TH_TIP = 1, 2, 3, 4

CALIB_FILE = "stereo_cam13_recalibrated.npz"

data = np.load(CALIB_FILE, allow_pickle=True)

K1 = np.array(data["K1"], dtype=np.float64)
K2 = np.array(data["K2"], dtype=np.float64)
D1 = np.array(data["D1"], dtype=np.float64).reshape(-1, 1)
D2 = np.array(data["D2"], dtype=np.float64).reshape(-1, 1)
R = np.array(data["R"], dtype=np.float64)
T = np.array(data["T"], dtype=np.float64).reshape(3, 1)

# Projection matrices
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K2 @ np.hstack((R, T))


def _seglen(a, b): return float(np.linalg.norm(a - b))

def _make_kf_2d(dt=1.0, q=1e-2, r=2e-1):
    """Constant-velocity 2D Kalman [x,y,vx,vy] -> [x,y]."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix   = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.processNoiseCov    = np.eye(4, dtype=np.float32) * q
    kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * r
    kf.errorCovPost       = np.eye(4, dtype=np.float32)
    return kf

def _make_kf_3d(dt=1.0, q=1e-4, r=5e-3):
    """Constant-velocity 3D Kalman [x,y,z,vx,vy,vz] -> [x,y,z]."""
    kf = cv2.KalmanFilter(6, 3)
    kf.transitionMatrix = np.array([
        [1,0,0,dt,0,0],
        [0,1,0,0,dt,0],
        [0,0,1,0,0,dt],
        [0,0,0,1,0,0 ],
        [0,0,0,0,1,0 ],
        [0,0,0,0,0,1 ],
    ], np.float32)
    kf.measurementMatrix = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
    ], np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * q
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * r
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    return kf


def _project_points(P, pts3d):
    pts3d_h = np.hstack([pts3d, np.ones((pts3d.shape[0], 1), dtype=np.float32)])
    proj = (P @ pts3d_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj.astype(np.float32)


def _reprojection_error(P, pts3d, pts2d):
    proj = _project_points(P, pts3d)
    err = np.linalg.norm(pts2d - proj, axis=1)
    return err, proj



class _ThumbTracker2D:
    """Lightweight 2D tracker for the 4 thumb joints; optional, off by default."""
    def __init__(self):
        self.kf = {_TH_CMC:_make_kf_2d(), _TH_MCP:_make_kf_2d(),
                   _TH_IP:_make_kf_2d(),  _TH_TIP:_make_kf_2d()}
        self.prev_frame_gray = None
        self.prev_px = None
        self.l1_ema = None; self.l2_ema = None
        self.alpha = 0.02  # EMA for MCP-IP and IP-TIP lengths

    def _quality_ok(self, px21):
        if px21 is None: return False
        try:
            mcp, ip, tip = px21[_TH_MCP], px21[_TH_IP], px21[_TH_TIP]
        except Exception:
            return False
        l1, l2 = _seglen(mcp, ip), _seglen(ip, tip)
        return (5 < l1 < 200) and (5 < l2 < 200)

    def update(self, rgb, kp2d_px_or_none):
        """
        kp2d_px_or_none: (21,2) np.float32 pixels or None
        Returns: corrected dict {idx -> (x,y)} for _TH_* indices (or None)
        """
        if kp2d_px_or_none is None or not self._quality_ok(kp2d_px_or_none):
            # No valid MP measurement; try optical flow if previous exists
            if self.prev_frame_gray is not None and self.prev_px is not None:
                p0 = np.stack([self.prev_px[i] for i in (_TH_CMC,_TH_MCP,_TH_IP,_TH_TIP)],
                              axis=0).astype(np.float32).reshape(-1,1,2)
                curr_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_frame_gray, curr_gray, p0, None,
                    winSize=(21,21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 20, 0.03)
                )
                # Predict + correct with tracked positions
                tracked = {}
                for i, idx in enumerate((_TH_CMC,_TH_MCP,_TH_IP,_TH_TIP)):
                    self.kf[idx].predict()
                    z = p1[i,0].astype(np.float32).reshape(2,1)
                    # less trust on flow
                    self.kf[idx].measurementNoiseCov[:] = np.eye(2, dtype=np.float32) * 5e-1
                    self.kf[idx].correct(z)
                    tracked[idx] = self.kf[idx].statePost[:2,0].copy()

                # Light length projection toward EMAs if available
                if self.l1_ema and self.l2_ema:
                    MCP, IP, TIP = tracked[_TH_MCP], tracked[_TH_IP], tracked[_TH_TIP]
                    v1, v2 = IP - MCP, TIP - IP
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-6: IP  = MCP + v1 * (self.l1_ema / n1)
                    if n2 > 1e-6: TIP = IP  + v2 * (self.l2_ema / n2)
                    tracked[_TH_IP], tracked[_TH_TIP] = IP, TIP

                self.prev_frame_gray = curr_gray
                self.prev_px = tracked
                return tracked
            else:
                # First frame(s): just predict forward
                tracked = {}
                for idx in (_TH_CMC,_TH_MCP,_TH_IP,_TH_TIP):
                    self.kf[idx].predict()
                    tracked[idx] = self.kf[idx].statePost[:2,0].copy()
                self.prev_px = tracked
                self.prev_frame_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                return tracked

        # Good MP measurement path
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cmc, mcp, ip, tip = (kp2d_px_or_none[_TH_CMC], kp2d_px_or_none[_TH_MCP],
                             kp2d_px_or_none[_TH_IP],  kp2d_px_or_none[_TH_TIP])
        # Update EMAs of segment lengths
        l1, l2 = _seglen(mcp, ip), _seglen(ip, tip)
        self.l1_ema = l1 if self.l1_ema is None else (1-self.alpha)*self.l1_ema + self.alpha*l1
        self.l2_ema = l2 if self.l2_ema is None else (1-self.alpha)*self.l2_ema + self.alpha*l2

        # KF predict + correct with MP measurement (more trust)
        out = {}
        for idx, zxy in zip((_TH_CMC,_TH_MCP,_TH_IP,_TH_TIP), (cmc, mcp, ip, tip)):
            self.kf[idx].predict()
            self.kf[idx].measurementNoiseCov[:] = np.eye(2, dtype=np.float32) * 2e-1
            self.kf[idx].correct(zxy.astype(np.float32).reshape(2,1))
            out[idx] = self.kf[idx].statePost[:2,0].copy()

        self.prev_px = out
        self.prev_frame_gray = rgb_gray
        return out

class SingleHandDetector:
    def __init__(
        self,
        hand_type="Right",
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        selfie=False,
        # --- new optional knobs (defaults keep old behavior) ---
        model_complexity=0,   # None => keep Mediapipe default (1)
        track_thumb=True,       # False => no change vs original
    ):
        # Keep original defaults unless user passes model_complexity
        kwargs = dict(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        if model_complexity is not None:
            kwargs["model_complexity"] = int(model_complexity)  # e.g., 1 or 2
        self.hand_detector = mp.solutions.hands.Hands(**kwargs)

        self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

        # New: optional thumb tracker
        self._track_thumb = bool(track_thumb)
        self._thumb_tracker = _ThumbTracker2D() if self._track_thumb else None
        # ---- new multi-view tracking state ----
        self._kf3d = {i: _make_kf_3d() for i in range(21)}
        self._prev_joint_pos = None
        self._triangulation_alpha = 0.35   # EMA smoothing weight for new measurement
        self._max_reproj_error_px = 12.0   # reject / soften bad triangulation
        self.active_cam = 1

    def project_points(self, P, pts3D):
        pts3D_h = np.hstack([pts3D, np.ones((pts3D.shape[0], 1))])
        proj = (P @ pts3D_h.T).T
        return proj[:, :2] / proj[:, 2:3]
    
    # @staticmethod
    def draw_skeleton_on_image(self, image, keypoints, proj_pts=None, color=(0,255,0)):

        out = image.copy()

        # =========================================================
        # 1. Draw MediaPipe skeleton (if available)
        # =========================================================
        if keypoints is not None and hasattr(keypoints, "landmark"):
            mp.solutions.drawing_utils.draw_landmarks(
                out,
                keypoints,
                mp.solutions.hands.HAND_CONNECTIONS
            )

        # =========================================================
        # 2. Draw triangulated reprojection (if available)
        # =========================================================
        if proj_pts is not None:
            for p in proj_pts.astype(int):
                cv2.circle(out, tuple(p), 5, (255, 0, 0), 1)  # blue hollow

        return out
 
    def triangulate_points(self, pts1, pts2, P1, P2):
        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        return (pts4D[:3] / pts4D[3]).T

    def project_points(self, P, pts3D):
        pts3D_h = np.hstack([pts3D, np.ones((pts3D.shape[0], 1))])
        proj = (P @ pts3D_h.T).T
        return proj[:, :2] / proj[:, 2:3]

    def reprojection_error(self, P, pts3D, pts2D):
        proj = self.project_points(P, pts3D)
        err = np.linalg.norm(pts2D - proj, axis=1)
        return err
    def _smooth_joint_pos(self, joint_pos):
        if not hasattr(self, "prev_joint"):
            self.prev_joint = joint_pos
            return joint_pos

        alpha = 0.7
        joint_pos = alpha * self.prev_joint + (1 - alpha) * joint_pos
        self.prev_joint = joint_pos
        return joint_pos

    def _detect_single(self, rgb):
        """
        Returns (unchanged):
          num_box (int), joint_pos (21x3), keypoint_2d (NormalizedLandmarkList), mediapipe_wrist_rot (3x3)
        """
        H, W = rgb.shape[:2]
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return 0, None, None, None

        # pick desired hand
        desired = -1
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label == self.detected_hand_type:
                desired = i; break
        if desired < 0:
            return 0, None, None, None

        keypoint_3d = results.multi_hand_world_landmarks[desired]
        keypoint_2d = results.multi_hand_landmarks[desired]
        num_box = len(results.multi_hand_landmarks)

        # --- parse 3D world landmarks (unchanged math) ---
        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano

        # --- optional: thumb tracking on 2D, written back into keypoint_2d (normalized) ---
        if self._track_thumb:
            # Convert current normalized 2D to pixels
            kp2d_px = self.parse_keypoint_2d(keypoint_2d, (H, W))  # (21,2) float32
            corrected = self._thumb_tracker.update(rgb, kp2d_px)   # dict or None
            if corrected:
                # Build a new NormalizedLandmarkList with corrected thumb pixels
                new_list = landmark_pb2.NormalizedLandmarkList()
                new_list.landmark.extend(keypoint_2d.landmark)  # copy all landmarks
                for idx in (_TH_CMC,_TH_MCP,_TH_IP,_TH_TIP):
                    u, v = corrected[idx]
                    new_list.landmark[idx].x = float(np.clip(u / W, 0.0, 1.0))
                    new_list.landmark[idx].y = float(np.clip(v / H, 0.0, 1.0))
                    # z stays as produced by MP; visibility not exposed by Hands
                keypoint_2d = new_list
                # (We keep joint_pos as-is; world landmarks were already smoothed by MP.)

        return num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot
    
    # def detect(self, rgb1, rgb2, P1, P2):

    #     # ---- run single-view detection ----
    #     num1, joint1, kp2d_1, rot1 = self._detect_single(rgb1)
    #     num2, joint2, kp2d_2, rot2 = self._detect_single(rgb2)

    #     # ---- build pixel points ----
    #     pts1, pts2 = None, None

    #     if kp2d_1 is not None:
    #         H1, W1 = rgb1.shape[:2]
    #         pts1 = np.array([[lm.x * W1, lm.y * H1] for lm in kp2d_1.landmark], dtype=np.float32)

    #     if kp2d_2 is not None:
    #         H2, W2 = rgb2.shape[:2]
    #         pts2 = np.array([[lm.x * W2, lm.y * H2] for lm in kp2d_2.landmark], dtype=np.float32)

    #     has1 = joint1 is not None
    #     has2 = joint2 is not None
    #     has_pts1 = pts1 is not None
    #     has_pts2 = pts2 is not None

    #     # =========================================================
    #     # TRY TRIANGULATION (only if both available)
    #     # =========================================================
    #     if has_pts1 and has_pts2:
    #         try:
    #             pts3D = self.triangulate_points(pts1, pts2, P1, P2)
    #             proj1 = self.project_points(P1, pts3D)
    #             proj2 = self.project_points(P2, pts3D)

    #             # ---- reprojection validation ----
    #             err1 = self.reprojection_error(P1, pts3D, pts1)
    #             err2 = self.reprojection_error(P2, pts3D, pts2)

    #             mean_err1 = np.mean(err1)
    #             mean_err2 = np.mean(err2)

    #             # ---- gating (CRITICAL) ----
    #             if mean_err1 < 5 and mean_err2 < 5:

    #                 # ---- normalize ----
    #                 pts3D = pts3D - pts3D[0:1]

    #                 # ---- wrist frame ----
    #                 R = self.estimate_frame_from_hand_points(pts3D)

    #                 # ---- map to MANO ----
    #                 joint_pos = pts3D @ R @ self.operator2mano
    #                 joint_pos = joint_pos.astype(np.float32)

    #                 # ---- smoothing ----
    #                 joint_pos = self._smooth_joint_pos(joint_pos)

    #                 self.active_cam = 0  # 0 = 3D

    #                 # return 1, joint_pos, kp2d_1, R
    #                 return 1, joint_pos, (kp2d_1, proj1, proj2), R

    #             else:
    #                 print(f"[WARN] Bad triangulation: err1={mean_err1:.2f}, err2={mean_err2:.2f}")

    #         except Exception as e:
    #             print("[ERROR] Triangulation failed:", e)

    #     # =========================================================
    #     # FALLBACK (single view)
    #     # =========================================================

    #     if has1 and has2:
    #         # keep previous camera if possible
    #         prev = getattr(self, "active_cam", 1)
    #         if prev == 1:
    #             self.active_cam = 1
    #             return num1, joint1, kp2d_1, rot1
    #         else:
    #             self.active_cam = 2
    #             return num2, joint2, kp2d_2, rot2

    #     elif has1:
    #         self.active_cam = 1
    #         return num1, joint1, kp2d_1, rot1

    #     elif has2:
    #         self.active_cam = 2
    #         return num2, joint2, kp2d_2, rot2

    #     # =========================================================
    #     # NO DETECTION
    #     # =========================================================
    #     return None, None, None, None


    @staticmethod
    def parse_keypoint_3d(keypoint_3d: framework.formats.landmark_pb2.LandmarkList) -> np.ndarray:
        keypoint = np.empty([21, 3], dtype=np.float32)
        for i in range(21):
            keypoint[i, 0] = keypoint_3d.landmark[i].x
            keypoint[i, 1] = keypoint_3d.landmark[i].y
            keypoint[i, 2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size) -> np.ndarray:
        H, W = img_size
        keypoint = np.empty([21, 2], dtype=np.float32)
        for i in range(21):
            keypoint[i, 0] = keypoint_2d.landmark[i].x * W
            keypoint[i, 1] = keypoint_2d.landmark[i].y * H
        return keypoint
    
    @staticmethod
    def triangulate_points(pts1, pts2, P1, P2):
        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3D = (pts4D[:3] / pts4D[3]).T
        return pts3D.astype(np.float32)

    def _smooth_joint_pos(self, joint_pos):
        """EMA smoothing in 3D."""
        if self._prev_joint_pos is None:
            self._prev_joint_pos = joint_pos.copy()
            return joint_pos

        smoothed = (
            (1.0 - self._triangulation_alpha) * self._prev_joint_pos
            + self._triangulation_alpha * joint_pos
        )
        self._prev_joint_pos = smoothed.copy()
        return smoothed

    def _kalman_filter_joint_pos(self, joint_pos, mean_err):
        """Per-joint 3D Kalman filtering with adaptive measurement noise."""
        filtered = np.empty_like(joint_pos, dtype=np.float32)

        # trust less when reprojection error is high
        r = 5e-3 if mean_err < 3.0 else 2e-2 if mean_err < 8.0 else 8e-2

        for i in range(21):
            self._kf3d[i].predict()
            self._kf3d[i].measurementNoiseCov[:] = np.eye(3, dtype=np.float32) * r
            z = joint_pos[i].astype(np.float32).reshape(3, 1)
            self._kf3d[i].correct(z)
            filtered[i] = self._kf3d[i].statePost[:3, 0]

        return filtered

    def _reject_bad_joints(self, joint_pos, err1, err2):
        """
        If a joint has very high reprojection error, keep previous filtered value if available.
        """
        if self._prev_joint_pos is None:
            return joint_pos

        out = joint_pos.copy()
        bad = (err1 > self._max_reproj_error_px) | (err2 > self._max_reproj_error_px)
        out[bad] = self._prev_joint_pos[bad]
        return out
    def triangulate_points(self, pts1, pts2, P1, P2):
        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3D = (pts4D[:3] / pts4D[3]).T
        return pts3D.astype(np.float32)

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]
        x_vector = points[0] - points[2]
        pts = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(pts)
        normal = v[2, :]
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1; z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

    def detect_fused(self, rgb1, rgb2):
        """
        Run detection on both cameras and fuse per-landmark,
        picking the best estimate for each joint individually.
        Returns same signature as original detect(): (num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot)
        """
        # Call _detect_single on each camera independently
        num_box1, joint_pos1, keypoint_2d1, wrist_rot1 = self._detect_single(rgb1)
        num_box2, joint_pos2, keypoint_2d2, wrist_rot2 = self._detect_single(rgb2)

        # Neither camera sees the hand
        if joint_pos1 is None and joint_pos2 is None:
            return 0, None, None, None

        # Only one camera sees the hand — use it directly
        if joint_pos1 is None:
            return num_box2, joint_pos2, keypoint_2d2, wrist_rot2
        if joint_pos2 is None:
            return num_box1, joint_pos1, keypoint_2d1, wrist_rot1

        # Both cameras see the hand — fuse per-landmark
        H1, W1 = rgb1.shape[:2]
        H2, W2 = rgb2.shape[:2]

        vis1 = self._landmark_visibility(keypoint_2d1, W1, H1)
        vis2 = self._landmark_visibility(keypoint_2d2, W2, H2)

        # Per-landmark pick the better camera's world landmark estimate
        joint_pos_fused = np.where(
            vis1[:, np.newaxis] >= vis2[:, np.newaxis],
            joint_pos1,
            joint_pos2
        )

        # For drawing, use whichever camera has higher overall confidence
        if np.mean(vis1) >= np.mean(vis2):
            keypoint_2d = keypoint_2d1
            wrist_rot = wrist_rot1
        else:
            keypoint_2d = keypoint_2d2
            wrist_rot = wrist_rot2

        return max(num_box1, num_box2), joint_pos_fused, keypoint_2d, wrist_rot

    def _landmark_visibility(self, keypoint_2d, W, H):
        """
        Compute per-landmark visibility score (21,) from normalized landmarks.
        Based on: in-bounds check + edge margin, same as multiview tracker.
        """
        vis = np.zeros(21, dtype=np.float32)
        for i, lm in enumerate(keypoint_2d.landmark):
            x = lm.x * W
            y = lm.y * H
            in_bounds = 1.0 if (0 <= x < W and 0 <= y < H) else 0.0
            margin = min(x, W - 1 - x, y, H - 1 - y)
            edge_score = np.clip(margin / 30.0, 0.0, 1.0)
            # Also use mediapipe's own visibility if available
            mp_vis = lm.visibility if hasattr(lm, 'visibility') else 1.0
            vis[i] = float(mp_vis * in_bounds * edge_score)
        return vis
    
def start_cam(serial: str):
    print (f"Starting camera with serial: {serial}")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipe.start(cfg)
    return pipe

if __name__ == "__main__":

    SERIAL_1 = "215322071654"   # Cam1
    SERIAL_2 = "213622077408"   # Cam3
    detector = SingleHandDetector(hand_type="Right", track_thumb=True)
    pipe1 = start_cam(SERIAL_1)
    pipe2 = start_cam(SERIAL_2)

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

            frame1 = np.asanyarray(color1.get_data())
            frame2 = np.asanyarray(color2.get_data())
       
            num_box, joint_pos, keypoint_2d, wrist_rot = detector.detect_fused(frame1, frame2)
            print(f"Detected {num_box} hand(s), joint_pos shape: {joint_pos.shape if joint_pos is not None else None}")

            # For visualization, draw on the first camera's frame
            cv2.imshow("Hand Detection", frame1)
            cv2.imshow("cam2", frame2)
            if keypoint_2d is not None:
                vis_frame = detector.draw_skeleton_on_image(frame1, keypoint_2d)
                vis_frame_2 = detector.draw_skeleton_on_image(frame2, keypoint_2d)
                cv2.imshow("Hand Detection", vis_frame)
                cv2.imshow("cam2", vis_frame_2)
                

            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

       

    finally:
        pipe1.stop()
        pipe2.stop()
        cv2.destroyAllWindows()
