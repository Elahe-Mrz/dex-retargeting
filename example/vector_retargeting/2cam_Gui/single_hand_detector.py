# # import mediapipe as mp
# # import mediapipe.framework as framework
# # import numpy as np
# # from mediapipe.framework.formats import landmark_pb2
# # from mediapipe.python.solutions import hands_connections
# # from mediapipe.python.solutions.drawing_utils import DrawingSpec
# # from mediapipe.python.solutions.hands import HandLandmark

# # OPERATOR2MANO_RIGHT = np.array(
# #     [
# #         [0, 0, -1],
# #         [-1, 0, 0],
# #         [0, 1, 0],
# #     ]
# # )

# # OPERATOR2MANO_LEFT = np.array(
# #     [
# #         [0, 0, -1],
# #         [1, 0, 0],
# #         [0, -1, 0],
# #     ]
# # )


# # class SingleHandDetector:
# #     def __init__(
# #         self,
# #         hand_type="Right",
# #         min_detection_confidence=0.8,
# #         min_tracking_confidence=0.8,
# #         selfie=False,
# #     ):
# #         self.hand_detector = mp.solutions.hands.Hands(
# #             static_image_mode=False,
# #             max_num_hands=1,
# #             min_detection_confidence=min_detection_confidence,
# #             min_tracking_confidence=min_tracking_confidence,
# #         )
# #         self.selfie = selfie
# #         self.operator2mano = (
# #             OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
# #         )
# #         inverse_hand_dict = {"Right": "Left", "Left": "Right"}
# #         self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

# #     @staticmethod
# #     def draw_skeleton_on_image(
# #         image, keypoint_2d: landmark_pb2.NormalizedLandmarkList, style="white"
# #     ):
# #         if style == "default":
# #             mp.solutions.drawing_utils.draw_landmarks(
# #                 image,
# #                 keypoint_2d,
# #                 mp.solutions.hands.HAND_CONNECTIONS,
# #                 mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
# #                 mp.solutions.drawing_styles.get_default_hand_connections_style(),
# #             )
# #         elif style == "white":
# #             landmark_style = {}
# #             for landmark in HandLandmark:
# #                 landmark_style[landmark] = DrawingSpec(
# #                     color=(255, 48, 48), circle_radius=4, thickness=-1
# #                 )

# #             connections = hands_connections.HAND_CONNECTIONS
# #             connection_style = {}
# #             for pair in connections:
# #                 connection_style[pair] = DrawingSpec(thickness=2)

# #             mp.solutions.drawing_utils.draw_landmarks(
# #                 image,
# #                 keypoint_2d,
# #                 mp.solutions.hands.HAND_CONNECTIONS,
# #                 landmark_style,
# #                 connection_style,
# #             )

# #         return image

# #     def detect(self, rgb):
# #         results = self.hand_detector.process(rgb)
# #         if not results.multi_hand_landmarks:
# #             return 0, None, None, None

# #         desired_hand_num = -1
# #         for i in range(len(results.multi_hand_landmarks)):
# #             label = results.multi_handedness[i].ListFields()[0][1][0].label
# #             if label == self.detected_hand_type:
# #                 desired_hand_num = i
# #                 break
# #         if desired_hand_num < 0:
# #             return 0, None, None, None

# #         keypoint_3d = results.multi_hand_world_landmarks[desired_hand_num]
# #         keypoint_2d = results.multi_hand_landmarks[desired_hand_num]
# #         num_box = len(results.multi_hand_landmarks)

# #         # Parse 3d keypoint from MediaPipe hand detector
# #         keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
# #         keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
# #         mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
# #         joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano

# #         return num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot

# #     @staticmethod
# #     def parse_keypoint_3d(
# #         keypoint_3d: framework.formats.landmark_pb2.LandmarkList,
# #     ) -> np.ndarray:
# #         keypoint = np.empty([21, 3])
# #         for i in range(21):
# #             keypoint[i][0] = keypoint_3d.landmark[i].x
# #             keypoint[i][1] = keypoint_3d.landmark[i].y
# #             keypoint[i][2] = keypoint_3d.landmark[i].z
# #         return keypoint

# #     @staticmethod
# #     def parse_keypoint_2d(
# #         keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size
# #     ) -> np.ndarray:
# #         keypoint = np.empty([21, 2])
# #         for i in range(21):
# #             keypoint[i][0] = keypoint_2d.landmark[i].x
# #             keypoint[i][1] = keypoint_2d.landmark[i].y
# #         keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
# #         return keypoint

# #     @staticmethod
# #     def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
# #         """
# #         Compute the 3D coordinate frame (orientation only) from detected 3d key points
# #         :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
# #         :return: the coordinate frame of wrist in MANO convention
# #         """
# #         assert keypoint_3d_array.shape == (21, 3)
# #         points = keypoint_3d_array[[0, 5, 9], :]

# #         # Compute vector from palm to the first joint of middle finger
# #         x_vector = points[0] - points[2]

# #         # Normal fitting with SVD
# #         points = points - np.mean(points, axis=0, keepdims=True)
# #         u, s, v = np.linalg.svd(points)

# #         normal = v[2, :]

# #         # Gram–Schmidt Orthonormalize
# #         x = x_vector - np.sum(x_vector * normal) * normal
# #         x = x / np.linalg.norm(x)
# #         z = np.cross(x, normal)

# #         # We assume that the vector from pinky to index is similar the z axis in MANO convention
# #         if np.sum(z * (points[1] - points[2])) < 0:
# #             normal *= -1
# #             z *= -1
# #         frame = np.stack([x, normal, z], axis=1)
# #         return frame






# import os
# os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"   # respected by many official graphs
# import numpy as np
# import cv2
# import mediapipe as mp
# import mediapipe.framework as framework
# from mediapipe.framework.formats import landmark_pb2
# from mediapipe.python.solutions import hands_connections
# from mediapipe.python.solutions.drawing_utils import DrawingSpec
# from mediapipe.python.solutions.hands import HandLandmark

# # ======= Coordinate transforms (unchanged idea) =======

# OPERATOR2MANO_RIGHT = np.array(
#     [[0, 0, -1],
#      [-1, 0,  0],
#      [0, 1,  0]]
# )

# OPERATOR2MANO_LEFT = np.array(
#     [[0, 0, -1],
#      [1, 0,  0],
#      [0, -1, 0]]
# )

# # Thumb indices in MediaPipe
# TH_CMC, TH_MCP, TH_IP, TH_TIP = 1, 2, 3, 4

# # ======= Small Kalman helpers =======

# def make_kf(dim_state, dim_meas, q=1e-3, r=1e-2, dt=1.0):
#     """Generic constant-velocity Kalman filter builder for 2D or 3D."""
#     if dim_meas == 2:  # [x,y,vx,vy]
#         kf = cv2.KalmanFilter(4, 2)
#         kf.transitionMatrix = np.array([[1,0,dt,0],
#                                         [0,1,0,dt],
#                                         [0,0,1, 0],
#                                         [0,0,0, 1]], np.float32)
#         kf.measurementMatrix = np.array([[1,0,0,0],
#                                          [0,1,0,0]], np.float32)
#         kf.processNoiseCov = np.eye(4, dtype=np.float32) * q
#         kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
#         kf.errorCovPost = np.eye(4, dtype=np.float32)
#     elif dim_meas == 3:  # [x,y,z,vx,vy,vz]
#         kf = cv2.KalmanFilter(6, 3)
#         kf.transitionMatrix = np.array([[1,0,0,dt,0,0],
#                                         [0,1,0,0,dt,0],
#                                         [0,0,1,0,0,dt],
#                                         [0,0,0,1,0,0 ],
#                                         [0,0,0,0,1,0 ],
#                                         [0,0,0,0,0,1 ]], np.float32)
#         kf.measurementMatrix = np.array([[1,0,0,0,0,0],
#                                          [0,1,0,0,0,0],
#                                          [0,0,1,0,0,0]], np.float32)
#         kf.processNoiseCov = np.eye(6, dtype=np.float32) * q
#         kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * r
#         kf.errorCovPost = np.eye(6, dtype=np.float32)
#     else:
#         raise ValueError("Unsupported measurement dimension")
#     return kf

# def seglen(a, b): return float(np.linalg.norm(a - b))

# # ======= Detector with built-in thumb tracking =======

# class SingleHandDetector:
#     def __init__(
#         self,
#         hand_type="Right",
#         min_detection_confidence=0.8,
#         min_tracking_confidence=0.8,
#         model_complexity=1,       # NEW: expose MP stability knob
#         track_thumb=True,         # NEW: enable/disable thumb tracking
#         selfie=False,
#     ):
#         self.hand_detector = mp.solutions.hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=min_detection_confidence,
#             min_tracking_confidence=min_tracking_confidence,
#             model_complexity=model_complexity,
#         )
#         self.selfie = selfie
#         self.operator2mano = (
#             OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
#         )
#         inverse_hand_dict = {"Right": "Left", "Left": "Right"}
#         self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

#         # Tracking state
#         self.track_thumb = track_thumb
#         self.prev_frame_gray = None
#         # Per-joint Kalman filters for thumb in 2D (pixels) and 3D (world)
#         self.kf2d = {i: make_kf(4, 2, q=1e-2, r=2e-1, dt=1.0) for i in (TH_CMC, TH_MCP, TH_IP, TH_TIP)}
#         self.kf3d = {i: make_kf(6, 3, q=1e-4, r=5e-3, dt=1.0) for i in (TH_CMC, TH_MCP, TH_IP, TH_TIP)}
#         self.prev_thumb_px = None
#         self.l1_ema = None
#         self.l2_ema = None
#         self.alpha_len = 0.02  # EMA for thumb segment lengths

#     # ---------- Drawing API (unchanged) ----------
#     @staticmethod
#     def draw_skeleton_on_image(image, keypoint_2d: landmark_pb2.NormalizedLandmarkList, style="white"):
#         if style == "default":
#             mp.solutions.drawing_utils.draw_landmarks(
#                 image,
#                 keypoint_2d,
#                 mp.solutions.hands.HAND_CONNECTIONS,
#                 mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
#                 mp.solutions.drawing_styles.get_default_hand_connections_style(),
#             )
#         elif style == "white":
#             landmark_style = {}
#             for landmark in HandLandmark:
#                 landmark_style[landmark] = DrawingSpec(color=(255, 48, 48), circle_radius=4, thickness=-1)
#             connections = hands_connections.HAND_CONNECTIONS
#             connection_style = {pair: DrawingSpec(thickness=2) for pair in connections}
#             mp.solutions.drawing_utils.draw_landmarks(
#                 image, keypoint_2d, mp.solutions.hands.HAND_CONNECTIONS, landmark_style, connection_style
#             )
#         return image

#     # ---------- Core detect (same signature) ----------
#     def detect(self, rgb):
#         """
#         Returns:
#           num_box (int),
#           joint_pos (np.ndarray 21x3),
#           keypoint_2d (NormalizedLandmarkList),
#           mediapipe_wrist_rot (3x3)
#         """
#         H, W = rgb.shape[:2]
#         results = self.hand_detector.process(rgb)
#         if not results.multi_hand_landmarks:
#             self.prev_frame_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
#             self.prev_thumb_px = None
#             return 0, None, None, None

#         # Pick the desired hand by handedness
#         desired_idx = -1
#         for i in range(len(results.multi_hand_landmarks)):
#             label = results.multi_handedness[i].ListFields()[0][1][0].label
#             if label == self.detected_hand_type:
#                 desired_idx = i
#                 break
#         if desired_idx < 0:
#             self.prev_frame_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
#             self.prev_thumb_px = None
#             return 0, None, None, None

#         keypoint_3d = results.multi_hand_world_landmarks[desired_idx]
#         keypoint_2d = results.multi_hand_landmarks[desired_idx]
#         num_box = len(results.multi_hand_landmarks)

#         # Parse 3D world landmarks (canonical units)
#         keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
#         keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
#         mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
#         joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano  # (21,3)

#         # Optionally run thumb tracking/smoothing
#         if self.track_thumb:
#             keypoint_2d_px = self.parse_keypoint_2d(keypoint_2d, (H, W))  # (21,2) pixels
#             thumb_px, mode = self._update_thumb_tracking(rgb, keypoint_2d_px)

#             if thumb_px is not None:
#                 # Update 2D landmarks (normalized) with corrected pixels
#                 keypoint_2d = self._replace_thumb_in_landmarklist(keypoint_2d, thumb_px, W, H)

#                 # Also smooth/patch 3D thumb joints using KF on world landmarks
#                 # Measurement: current MP world points; if low quality, rely more on prediction
#                 for j in (TH_CMC, TH_MCP, TH_IP, TH_TIP):
#                     self.kf3d[j].predict()
#                     meas3d = joint_pos[j].astype(np.float32).reshape(3, 1)
#                     # Increase measurement noise if mode != MP
#                     R = (5e-3 if mode == "MP" else 5e-2) * np.eye(3, dtype=np.float32)
#                     self.kf3d[j].measurementNoiseCov[:] = R
#                     self.kf3d[j].correct(meas3d)
#                     # Write back filtered 3D
#                     joint_pos[j] = self.kf3d[j].statePost[:3, 0]

#         # Cache frame gray for next optical flow
#         self.prev_frame_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
#         return num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot

#     # ---------- Parsing helpers (unchanged) ----------
#     @staticmethod
#     def parse_keypoint_3d(keypoint_3d: framework.formats.landmark_pb2.LandmarkList) -> np.ndarray:
#         keypoint = np.empty([21, 3], dtype=np.float32)
#         for i in range(21):
#             keypoint[i, 0] = keypoint_3d.landmark[i].x
#             keypoint[i, 1] = keypoint_3d.landmark[i].y
#             keypoint[i, 2] = keypoint_3d.landmark[i].z
#         return keypoint

#     @staticmethod
#     def parse_keypoint_2d(keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size) -> np.ndarray:
#         H, W = img_size
#         pts = np.empty([21, 2], dtype=np.float32)
#         for i in range(21):
#             pts[i, 0] = keypoint_2d.landmark[i].x * W
#             pts[i, 1] = keypoint_2d.landmark[i].y * H
#         return pts

#     @staticmethod
#     def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
#         assert keypoint_3d_array.shape == (21, 3)
#         points = keypoint_3d_array[[0, 5, 9], :]  # wrist, index MCP, middle MCP

#         # vector from palm to middle MCP
#         x_vector = points[0] - points[2]

#         # normal via SVD
#         pts = points - np.mean(points, axis=0, keepdims=True)
#         _, _, v = np.linalg.svd(pts)
#         normal = v[2, :]

#         # Gram–Schmidt
#         x = x_vector - np.sum(x_vector * normal) * normal
#         x = x / np.linalg.norm(x)
#         z = np.cross(x, normal)

#         # Adjust sign so pinky->index roughly matches z
#         if np.sum(z * (points[1] - points[2])) < 0:
#             normal *= -1
#             z *= -1
#         frame = np.stack([x, normal, z], axis=1)
#         return frame

#     # ---------- Thumb tracking internals ----------
#     def _replace_thumb_in_landmarklist(self, lm_norm: landmark_pb2.NormalizedLandmarkList,
#                                        thumb_px: dict, W: int, H: int):
#         """Create a new NormalizedLandmarkList with corrected thumb 2D (in normalized coords)."""
#         new_list = landmark_pb2.NormalizedLandmarkList()
#         new_list.landmark.extend(lm_norm.landmark)  # copy all
#         mapping = {TH_CMC: "CMC", TH_MCP: "MCP", TH_IP: "IP", TH_TIP: "TIP"}
#         for idx, name in mapping.items():
#             px, py = thumb_px[name]
#             nx = float(np.clip(px / W, 0.0, 1.0))
#             ny = float(np.clip(py / H, 0.0, 1.0))
#             new_list.landmark[idx].x = nx
#             new_list.landmark[idx].y = ny
#             # keep z & visibility as-is (MP doesn't provide per-pt conf here)
#         return new_list

#     def _thumb_quality_ok(self, px21, prev_thumb_px):
#         """Basic sanity on thumb geometry in pixels."""
#         if px21 is None: return False
#         try:
#             mcp = px21[TH_MCP]; ip = px21[TH_IP]; tip = px21[TH_TIP]
#         except Exception:
#             return False

#         l1 = seglen(mcp, ip); l2 = seglen(ip, tip)
#         if not (5 < l1 < 200 and 5 < l2 < 200): return False

#         if prev_thumb_px is not None:
#             v = seglen(tip, prev_thumb_px["TIP"])
#             if v > 30: return False
#         return True

#     def _update_thumb_tracking(self, rgb, keypoint_2d_px):
#         """Returns (thumb_px_dict, mode), where mode in {'MP','FUSE','PRED'}."""
#         H, W = rgb.shape[:2]
#         # Build dict from current MP output
#         thumb_px = {
#             "CMC": keypoint_2d_px[TH_CMC].astype(np.float32) if keypoint_2d_px is not None else None,
#             "MCP": keypoint_2d_px[TH_MCP].astype(np.float32) if keypoint_2d_px is not None else None,
#             "IP" : keypoint_2d_px[TH_IP ].astype(np.float32) if keypoint_2d_px is not None else None,
#             "TIP": keypoint_2d_px[TH_TIP].astype(np.float32) if keypoint_2d_px is not None else None,
#         } if keypoint_2d_px is not None else None

#         use_mp = self._thumb_quality_ok(keypoint_2d_px, self.prev_thumb_px)

#         # Predict step for all thumb joints (2D)
#         for j in (TH_CMC, TH_MCP, TH_IP, TH_TIP):
#             self.kf2d[j].predict()

#         if use_mp and thumb_px is not None:
#             # Update EMA of segment lengths
#             l1 = seglen(thumb_px["MCP"], thumb_px["IP"])
#             l2 = seglen(thumb_px["IP"],  thumb_px["TIP"])
#             self.l1_ema = l1 if self.l1_ema is None else (1 - self.alpha_len) * self.l1_ema + self.alpha_len * l1
#             self.l2_ema = l2 if self.l2_ema is None else (1 - self.alpha_len) * self.l2_ema + self.alpha_len * l2

#             # Correct Kalman with MP measurements
#             for idx, name in zip((TH_CMC, TH_MCP, TH_IP, TH_TIP), ("CMC","MCP","IP","TIP")):
#                 z = np.asarray(thumb_px[name], np.float32).reshape(2, 1)
#                 self.kf2d[idx].measurementNoiseCov[:] = np.eye(2, dtype=np.float32) * 2e-1
#                 self.kf2d[idx].correct(z)
#             mode = "MP"
#         else:
#             # Fallback: optical flow from previous frame if possible
#             if self.prev_frame_gray is not None and self.prev_thumb_px is not None:
#                 p0 = np.stack([self.prev_thumb_px[k] for k in ("CMC","MCP","IP","TIP")], axis=0).astype(np.float32)
#                 p0 = p0.reshape(-1, 1, 2)
#                 curr_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
#                 p1, st, err = cv2.calcOpticalFlowPyrLK(
#                     self.prev_frame_gray, curr_gray, p0, None,
#                     winSize=(21,21), maxLevel=3,
#                     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 20, 0.03)
#                 )
#                 tracked = {name: p1[i,0] for i,name in enumerate(("CMC","MCP","IP","TIP"))}

#                 # Light projection toward average segment lengths (if known)
#                 if self.l1_ema is not None and self.l2_ema is not None:
#                     MCP, IP, TIP = tracked["MCP"], tracked["IP"], tracked["TIP"]
#                     v1, v2 = IP - MCP, TIP - IP
#                     n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
#                     if n1 > 1e-6: IP  = MCP + v1 * (self.l1_ema / n1)
#                     if n2 > 1e-6: TIP = IP  + v2 * (self.l2_ema / n2)
#                     tracked["IP"], tracked["TIP"] = IP, TIP

#                 # Correct Kalman with tracked measurements (higher R → softer trust)
#                 for idx, name in zip((TH_CMC, TH_MCP, TH_IP, TH_TIP), ("CMC","MCP","IP","TIP")):
#                     z = np.asarray(tracked[name], np.float32).reshape(2, 1)
#                     self.kf2d[idx].measurementNoiseCov[:] = np.eye(2, dtype=np.float32) * 5e-1
#                     self.kf2d[idx].correct(z)
#                 thumb_px = tracked
#                 mode = "FUSE"
#             else:
#                 # Pure prediction
#                 thumb_px = {name: self.kf2d[idx].statePost[:2, 0] for idx, name in zip(
#                     (TH_CMC, TH_MCP, TH_IP, TH_TIP), ("CMC","MCP","IP","TIP")
#                 )}
#                 mode = "PRED"

#         # Final 2D output from filters
#         out = {name: self.kf2d[idx].statePost[:2, 0] for idx, name in zip(
#             (TH_CMC, TH_MCP, TH_IP, TH_TIP), ("CMC","MCP","IP","TIP")
#         )}
#         self.prev_thumb_px = out
#         return out, mode







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

# --- unchanged transforms ---
OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OPERATOR2MANO_LEFT  = np.array([[0, 0, -1], [ 1, 0, 0], [0,-1, 0]])

# Thumb indices in MediaPipe order
_TH_CMC, _TH_MCP, _TH_IP, _TH_TIP = 1, 2, 3, 4

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
        model_complexity=1,   # None => keep Mediapipe default (1)
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

    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d: landmark_pb2.NormalizedLandmarkList, style="white"):
        if style == "default":
            mp.solutions.drawing_utils.draw_landmarks(
                image, keypoint_2d, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
        elif style == "white":
            landmark_style = {lm: DrawingSpec(color=(255, 48, 48), circle_radius=4, thickness=-1)
                              for lm in HandLandmark}
            connection_style = {pair: DrawingSpec(thickness=2) for pair in hands_connections.HAND_CONNECTIONS}
            mp.solutions.drawing_utils.draw_landmarks(
                image, keypoint_2d, mp.solutions.hands.HAND_CONNECTIONS, landmark_style, connection_style
            )
        return image

    def detect(self, rgb):
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
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]
        x_vector = points[0] - points[2]
        pts = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(pts)
        normal = v[2, :]
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / (np.linalg.norm(x) + 1e-6)
        z = np.cross(x, normal)
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1; z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame
