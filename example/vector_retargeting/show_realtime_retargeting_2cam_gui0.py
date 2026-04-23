#!/usr/bin/env python3

import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import multiprocessing
import ctypes
import time
from pathlib import Path
from queue import Empty
from typing import Optional
import os
import csv
import queue as pyqueue
import threading
from datetime import datetime
from numpy.polynomial import Polynomial

import cv2
import numpy as np
import pyrealsense2 as rs
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from ah_wrapper import AHSerialClient
from hand_data_recorder import HandDataRecorder

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
import mediapipe as mp

# Keep the fusion logic exactly from landmarkfusion.py
from landmarkfusion import *
from single_hand_detector import SingleHandDetector

SERIAL_1 = "215322071654"   # Cam1
SERIAL_2 = "213622077408"   # Cam2
WIDTH, HEIGHT, FPS = 640, 480, 30
SYNC_MAX_BUF = 10
SYNC_MAX_DT = 20.0  # ms


class FrameWriter:
    def __init__(self, path, fps=30, frame_size=(640, 480), fourcc_list=("mp4v", "XVID")):
        self.path = path
        self.fps = fps
        self.frame_size = frame_size
        self.out = None
        self._q = pyqueue.Queue(maxsize=120)
        self._t = None
        self._stop = threading.Event()
        self._fourcc_list = fourcc_list

    def _open_writer(self):
        for cc in self._fourcc_list:
            fourcc = cv2.VideoWriter_fourcc(*cc)
            out = cv2.VideoWriter(self.path, fourcc, self.fps, self.frame_size)
            if out.isOpened():
                self.out = out
                return True
        return False

    def start(self):
        if self.out is None:
            if not self._open_writer():
                raise RuntimeError(f"Could not open VideoWriter for {self.path} with any FOURCC {self._fourcc_list}")
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.2)
            except pyqueue.Empty:
                continue
            try:
                self.out.write(frame)
            except Exception:
                pass

    def write(self, frame_bgr):
        try:
            self._q.put_nowait(frame_bgr)
        except pyqueue.Full:
            pass

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2.0)
        if self.out:
            try:
                self.out.release()
            except Exception:
                pass


class AngleCSVLogger(object):
    def __init__(self, save_dir="./data_logs", base_name="angles", finger_names=None):
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base = os.path.join(save_dir, f"{base_name}_{ts}")
        self.finger_names = list(finger_names) if finger_names else None
        self._raw_fp = None
        self._cal_fp = None
        self._read_fp = None
        self._raw_writer = None
        self._cal_writer = None
        self._read_writer = None
        self._opened = False

    def _ensure_open(self, n_cols):
        if self._opened:
            return
        if not self.finger_names:
            self.finger_names = [f"F{i+1}" for i in range(n_cols)]
        header = ["frame_capture_t", "command_sent_t"] + self.finger_names

        self._raw_fp = open(self.base + "_raw.csv", "w", newline="")
        self._cal_fp = open(self.base + "_calibrated.csv", "w", newline="")
        self._read_fp = open(self.base + "_read.csv", "w", newline="")
        self._raw_writer = csv.writer(self._raw_fp)
        self._cal_writer = csv.writer(self._cal_fp)
        self._read_writer = csv.writer(self._read_fp)
        self._raw_writer.writerow(header)
        self._cal_writer.writerow(header)
        self._read_writer.writerow(header)
        self._opened = True

    def write(self, frame_capture_t, command_sent_t, raw=None, calibrated=None, readback=None):
        ref = raw if raw is not None else (calibrated if calibrated is not None else readback)
        if ref is None:
            return
        ref = list(ref)
        self._ensure_open(len(ref))
        
        prefix = [f"{frame_capture_t:.6f}", f"{command_sent_t:.6f}"]

        if raw is not None:
            self._raw_writer.writerow(prefix + list(raw))
        if calibrated is not None:
            self._cal_writer.writerow(prefix + list(calibrated))
        if readback is not None:
            self._read_writer.writerow(prefix + list(readback))

        if int(frame_capture_t) % 2 == 0:
            self._raw_fp.flush()
            self._cal_fp.flush()
            self._read_fp.flush()

    def close(self):
        for fp in (self._raw_fp, self._cal_fp, self._read_fp):
            try:
                if fp:
                    fp.close()
            except Exception:
                pass
        self._opened = False

    @property
    def filepaths(self):
        return {
            "raw": self.base + "_raw.csv",
            "calibrated": self.base + "_calibrated.csv",
            "read": self.base + "_read.csv",
        }


def calibrate2(commanded_angles):
    if len(commanded_angles) != 6:
        raise ValueError("Input must be a list or array of 6 commanded angles (one per finger).")

    calibration_params = {
        0: [0.0000447829, -0.0055038815, 1.2407873038, -2.3608804008],
        1: [0.0000361534, -0.0041874131, 1.1805583523, -1.8249082122],
        2: [0.0000461200, -0.0057085443, 1.2455288954, -2.3906153698],
        3: [0.0000408076, -0.0055052341, 1.2510940056, -2.5055910350],
        4: [0.0000086317, -0.0011940936, 1.0658503059, -0.3982449601],
        5: [0.0000328414,  0.0053339517, 1.2574827929,  0.2875786203]
    }

    calibrated = []
    for finger_index, x in enumerate(commanded_angles):
        if x > 40:
            x = x + 7
        elif x > 10:
            x = x - 10
        a, b, c, d = calibration_params[finger_index]
        y = a * x**3 + b * x**2 + c * x + d
        if finger_index == 5:
            y = max(y, -120)
        else:
            y = min(max(y, 0), 120)
        calibrated.append(y)

    return np.array(calibrated)


def put_latest(q: multiprocessing.Queue, item):
    while True:
        try:
            q.put_nowait(item)
            return
        except pyqueue.Full:
            try:
                q.get_nowait()
            except Exception:
                return


def camera_worker(serial: str, out_queue: multiprocessing.Queue, stop_event: multiprocessing.Event):
    pipeline = None
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        pipeline.start(config)

        while stop_event is None or not stop_event.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=100)
            except RuntimeError:
                continue

            color = frames.get_color_frame()
            if not color:
                continue

            ts_ms = frames.get_timestamp()
            img = np.asanyarray(color.get_data())
            put_latest(out_queue, (ts_ms, img))
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass


def drain_latest(q: multiprocessing.Queue):
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except Exception:
            break
    return latest


def wrong_hand_in_frame(results, expected_label):
    if not results or not results.multi_hand_landmarks or not results.multi_handedness:
        return False
    for i, handedness in enumerate(results.multi_handedness):
        label = handedness.classification[0].label
        if label != expected_label:
            return True
    return False

# ═════════════════════════════════════════════════════════════════════════════
#  Shared-memory frame writer (for GUI)
# ═════════════════════════════════════════════════════════════════════════════
def _write_frame_to_shm(shm, frame_rgb: np.ndarray):
    """
    Write an HWC uint8 RGB frame into a multiprocessing.Array.
    Layout: [flag(1), wlo, whi, hlo, hhi, c, ...pixels]
    """
    if shm is None:
        return
    try:
        h, w, c = frame_rgb.shape
        frame_rgb = np.ascontiguousarray(frame_rgb)
        n = w * h * c
        buf = shm.get_obj()
        if 6 + n > len(buf):
            return
        buf[1] = w & 0xFF;  buf[2] = (w >> 8) & 0xFF
        buf[3] = h & 0xFF;  buf[4] = (h >> 8) & 0xFF
        buf[5] = c
        ctypes.memmove(
            ctypes.addressof(buf) + 6,
            frame_rgb.ctypes.data,
            n
        )
        buf[0] = 1
    except Exception:
        pass


def _sapien_cam_to_rgb(cam) -> np.ndarray | None:
    try:
        rgba = cam.get_color_rgba()
        if rgba is not None:
            return (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
    except Exception: pass
    try:
        rgba = cam.get_picture_uint8("Color")
        if rgba is not None:
            return rgba[:, :, :3]
    except Exception: pass
    try:
        rgba = cam.get_color_rgba()
        arr  = np.array(rgba)
        if arr.dtype == np.uint8:
            return arr[:, :, :3]
        return (np.clip(arr[:, :, :3], 0, 1) * 255).astype(np.uint8)
    except Exception: pass
    return None

def start_retargeting(
    queue1: multiprocessing.Queue,
    queue2: multiprocessing.Queue,
    stop_event,
    robot_dir: str,
    config_path: str,
    hand_type: HandType,
    save_event=None,
    cam_shm=None,
    robot_shm=None,
    hw_retargeting_enabled=False
):
    always_save = (save_event is None)
    base_dir = os.environ.get("NEUROCAPTURE_BASE_DIR", "data_logs")

    hand_client = AHSerialClient(write_thread=False) if hw_retargeting_enabled else None
    recorder = None
    
    output_folder = None
    record_video = True
    video_writer1 = None
    video_writer2 = None
    angle_logger = None
    log_file = None
    tracking_csv_file = None
    tracking_csv_writer = None
    timestamp = None
    video_path1 = None
    video_path2 = None
    _was_saving = False
    _frame_counter = 0
    _RENDER_EVERY = 3

    finger_names = ['index_q1', 'middle_q1', 'ring_q1', 'pinky_q1', 'thumb_q2', 'thumb_q1']

    def _open_save_files():
        nonlocal output_folder, video_writer1, video_writer2, angle_logger, log_file, timestamp, video_path1, video_path2, recorder, tracking_csv_file, tracking_csv_writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(base_dir, f"retargeting_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)
        
        if hand_client:
            output_csv = os.path.join(output_folder, f"teleop_data_{timestamp}.csv")
            recorder = HandDataRecorder(reply_mode=0, output_csv=output_csv, client=hand_client)
            recorder.start()
            logger.info(f"HandDataRecorder → {output_csv}")

        angle_logger = AngleCSVLogger(save_dir=output_folder, base_name="retargeting_angles", finger_names=finger_names)
        
        log_file_path = f"{output_folder}/retargeting_log_{timestamp}.txt"
        log_file = open(log_file_path, "w")
        log_file.write(f"{'frame_t':<18} {'cmd_t':<18} {'Original Angles':<40} | {'Calibrated Angles':<40} | {'Measured Angles'}\n")
        log_file.write("-" * 140 + "\n")

        tracking_csv_path = os.path.join(output_folder, f"tracking_log_{timestamp}.csv")
        tracking_csv_file = open(tracking_csv_path, "w", newline="")
        tracking_csv_writer = csv.writer(tracking_csv_file)
        tracking_header = [
            "timestamp_cam1_ms", "timestamp_cam2_ms", "timestamp_fused_ms",
            "conf_cam1", "conf_cam2", "frame_score", "mean_reproj_cam1",
            "mean_reproj_cam2", "gt_valid",
        ]
        for i in range(21):
            tracking_header += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z", f"c1_lm{i}_x", f"c1_lm{i}_y", f"c2_lm{i}_x", f"c2_lm{i}_y", f"lm{i}_conf", f"lm{i}_err1", f"lm{i}_err2"]
        tracking_csv_writer.writerow(tracking_header)

        video_path1 = os.path.join(output_folder, f"raw_cam1_color_{timestamp}.mp4")
        video_path2 = os.path.join(output_folder, f"raw_cam2_color_{timestamp}.mp4")
        logger.info(f"Retargeting: saving opened → {output_folder}")

    def _close_save_files():
        nonlocal video_writer1, video_writer2, angle_logger, log_file, recorder, tracking_csv_file, tracking_csv_writer
        if recorder is not None:
            try: recorder.stop()
            except Exception: pass
            recorder = None
        if video_writer1:
            video_writer1.stop()
            video_writer1 = None
        if video_writer2:
            video_writer2.stop()
            video_writer2 = None
        if log_file is not None:
            try: log_file.close()
            except Exception: pass
            log_file = None
        if angle_logger is not None:
            angle_logger.close()
            angle_logger = None
        if tracking_csv_file is not None:
            try: tracking_csv_file.close()
            except Exception: pass
            tracking_csv_file = None

    mp_hands = mp.solutions.hands
    hands1 = mp_hands.Hands(max_num_hands=1)
    hands2 = mp_hands.Hands(max_num_hands=1)
    if isinstance(hand_type, str):
        ht_str = "Right" if "right" in hand_type.lower() else "Left"
    else:
        ht_str = "Right" if "right" in config_path.lower() else "Left"
    tracker = MultiViewTracker(ht_str)
    single_detector_cam1 = SingleHandDetector(hand_type=ht_str)
    single_detector_cam2 = SingleHandDetector(hand_type=ht_str)

    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    _use_viewer = (cam_shm is None and robot_shm is None)
    if _use_viewer:
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())
    else:
        viewer = None

    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5

    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    if "ability" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))

    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    sync_buffer1 = []
    sync_buffer2 = []
    last_cam1 = None
    last_cam2 = None

    try:
        while stop_event is None or not stop_event.is_set():
            currently_saving = always_save or (save_event is not None and save_event.is_set())

            if currently_saving and not _was_saving:
                _open_save_files()
                _was_saving = True
            elif not currently_saving and _was_saving:
                _close_save_files()
                _was_saving = False

            _frame_counter += 1

            item1 = drain_latest(queue1)
            item2 = drain_latest(queue2)

            if item1 is not None:
                last_cam1 = item1
            if item2 is not None:
                last_cam2 = item2

            if last_cam1 is None or last_cam2 is None:
                if viewer:
                    viewer.render()
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        if stop_event: stop_event.set()
                        break
                continue
            
            frame_capture_t = time.time()
            ts1, img1 = last_cam1
            ts2, img2 = last_cam2

            if currently_saving and record_video and video_writer1 is None and video_path1:
                h, w = img1.shape[:2]
                video_writer1 = FrameWriter(path=video_path1, fps=30, frame_size=(w,h), fourcc_list=("avc1", "mp4v", "XVID"))
                video_writer2 = FrameWriter(path=video_path2, fps=30, frame_size=(w,h), fourcc_list=("avc1", "mp4v", "XVID"))
                video_writer1.start()
                video_writer2.start()

            pts1, vis1, hand_conf1, res1 = tracker.get_landmarks_and_visibility(img1, hands1)
            pts2, vis2, hand_conf2, res2 = tracker.get_landmarks_and_visibility(img2, hands2)

            if pts1 is not None:
                sync_buffer1.append((ts1, pts1, vis1, hand_conf1, res1, img1))
            elif wrong_hand_in_frame(res1, tracker.expected_label):
                sync_buffer1.clear()
            else:
                sync_buffer1.clear()

            if pts2 is not None:
                sync_buffer2.append((ts2, pts2, vis2, hand_conf2, res2, img2))
            elif wrong_hand_in_frame(res2, tracker.expected_label):
                sync_buffer2.clear()
            else:
                sync_buffer2.clear()

            current_ts = max(ts1, ts2)
            sync_buffer1 = [x for x in sync_buffer1 if current_ts - x[0] < 200.0]
            sync_buffer2 = [x for x in sync_buffer2 if current_ts - x[0] < 200.0]

            if len(sync_buffer1) > SYNC_MAX_BUF:
                sync_buffer1 = sync_buffer1[-SYNC_MAX_BUF:]
            if len(sync_buffer2) > SYNC_MAX_BUF:
                sync_buffer2 = sync_buffer2[-SYNC_MAX_BUF:]

            _do_display = (_frame_counter % _RENDER_EVERY == 0)
            if _do_display or (currently_saving and video_writer1 is not None):
                disp1 = img1.copy()
                disp2 = img2.copy()
                if pts1 is not None:
                    disp1 = draw_hand_skeleton(disp1, res1, tracker.expected_label)
                if pts2 is not None:
                    disp2 = draw_hand_skeleton(disp2, res2, tracker.expected_label)
            else:
                disp1 = img1
                disp2 = img2

            frame_score = 0.0
            best_pair = None
            best_dt = float("inf")

            for a in sync_buffer1[-SYNC_MAX_BUF:]:
                for b in sync_buffer2[-SYNC_MAX_BUF:]:
                    dt_match = abs(a[0] - b[0])
                    if 0 <= dt_match <= SYNC_MAX_DT and dt_match < best_dt:
                        best_dt = dt_match
                        best_pair = (a, b)

            joint_pos = None

            if best_pair is not None:
                (t1, pts1, vis1, hand_conf1, res1, img1_used), (t2, pts2, vis2, hand_conf2, res2, img2_used) = best_pair

                if hand_conf1 >= MIN_HAND_CONF and hand_conf2 >= MIN_HAND_CONF:
                    result = tracker.process(img1_used, img2_used, pts1, pts2, vis1, vis2)

                    if result is not None:
                        joint_pos, pts3D, frame_score, gt_valid, final_conf, err1, err2 = result
                        if joint_pos is not None and currently_saving and tracking_csv_writer is not None:
                            mean_err1 = float(np.mean(err1))
                            mean_err2 = float(np.mean(err2))
                            tracking_row = [
                            float(t1), float(t2), float((t1+t2)/2.0),
                            float(hand_conf1), float(hand_conf2), float(frame_score),
                            mean_err1, mean_err2, int(gt_valid),
                        ]
                            for i in range(21):
                                tracking_row += [
                                    float(pts3D[i, 0]), float(pts3D[i, 1]), float(pts3D[i, 2]),
                                    float(pts1[i, 0]), float(pts1[i, 1]), float(pts2[i, 0]), float(pts2[i, 1]),
                                    float(final_conf[i]), float(err1[i]), float(err2[i]),
                                ]
                            tracking_csv_writer.writerow(tracking_row)

                        if joint_pos is not None and (_do_display or (currently_saving and video_writer1 is not None)):
                            proj1 = project_points(P1, pts3D)
                            proj2 = project_points(P2, pts3D)
                            disp1 = draw_landmarks_with_confidence(disp1, pts1, vis1, ids=DRAW_IDS)
                            disp2 = draw_landmarks_with_confidence(disp2, pts2, vis2, ids=DRAW_IDS)
                            disp1 = draw_triangulated_landmarks(disp1, proj1, final_conf)
                            disp2 = draw_triangulated_landmarks(disp2, proj2, final_conf)

                            z_tip = pts3D[8, 2]
                            wrist = pts3D[0]

                            text1 = f"C1={hand_conf1:.2f}  C2={hand_conf2:.2f}  Score={frame_score:.2f}"
                            text2 = f"Err1={np.mean(err1):.2f}px  Err2={np.mean(err2):.2f}px  Z_tip={z_tip:.3f}"
                            cv2.putText(disp1, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                            cv2.putText(disp1, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                            cv2.putText(disp2, f"Wrist3D=({wrist[0]:.3f}, {wrist[1]:.3f}, {wrist[2]:.3f})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            if joint_pos is None:
                if pts1 is not None:
                    rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                    num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = single_detector_cam1.detect(rgb1)
                elif pts2 is not None:
                    rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = single_detector_cam2.detect(rgb2)

            if joint_pos is None:
                joint_pos = tracker.prev_joint
            else:
                tracker.prev_joint = joint_pos
            

            if currently_saving and video_writer1 is not None and video_writer2 is not None:
                video_writer1.write(disp1)
                video_writer2.write(disp2)
                
            if _use_viewer:
                cv2.imshow("cam1", disp1)
                cv2.imshow("cam2", disp2)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    if stop_event: stop_event.set()
                    break

            if cam_shm is not None and _do_display:
                disp_combined = cv2.hconcat([disp1, disp2])
                small = cv2.resize(disp_combined, (320, 240), interpolation=cv2.INTER_LINEAR)
                _write_frame_to_shm(cam_shm, cv2.cvtColor(small, cv2.COLOR_BGR2RGB))


            if joint_pos is not None:
                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                if retargeting_type == "POSITION":
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

                qpos = retargeting.retarget(ref_value)
                robot.set_qpos(qpos[retargeting_to_sapien])

                idxs = [retargeting_joint_names.index(name) for name in finger_names]
                hand_angles = qpos[idxs] * 180 / np.pi
                commanded_angles = calibrate2(hand_angles).tolist()

                if hand_client:
                    hand_client.set_position(commanded_angles, reply_mode=0)
                    hand_client.send_command()
                    command_sent_t = time.time()
                    measured = hand_client.hand.get_position()
                else:
                    command_sent_t = time.time()
                    measured = [0] * 6

                if currently_saving and log_file is not None:
                    log_line = f"{frame_capture_t:<18.6f} {command_sent_t:<18.6f} {str(hand_angles.tolist()):<40} | {str(commanded_angles):<40} | {str(measured)}\n"
                    log_file.write(log_line)
                
                if currently_saving and angle_logger is not None:
                    angle_logger.write(frame_capture_t, command_sent_t, raw=hand_angles, calibrated=commanded_angles, readback=measured)

            if _use_viewer and viewer is not None:
                viewer.render()
            elif robot_shm is not None and _do_display:
                try:
                    scene.update_render()
                    cam.take_picture()
                    rgb = _sapien_cam_to_rgb(cam)
                    if rgb is not None:
                        small = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_LINEAR)
                        _write_frame_to_shm(robot_shm, small)
                except Exception:
                    pass

    finally:
        _close_save_files()
        hands1.close()
        hands2.close()
        cv2.destroyAllWindows()
        print("Done")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    tyro.cli(start_retargeting)
