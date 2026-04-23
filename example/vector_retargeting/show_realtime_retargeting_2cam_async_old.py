import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional
import pyrealsense2 as rs
import os 
import mediapipe as mp

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from ah_wrapper import AHSerialClient
from hand_data_recorder import HandDataRecorder
from datetime import datetime  
from numpy.polynomial import Polynomial
import threading
from dataclasses import dataclass
import threading

stop_capture_event = multiprocessing.Event()

@dataclass
class LatestFrame:
    t_capture: float = 0.0
    bgr1: Optional[np.ndarray] = None
    bgr2: Optional[np.ndarray] = None
    seq: int = -1

@dataclass
class LatestPose:
    t_capture: float = 0.0
    joint_pos: Optional[np.ndarray] = None
    disp1: Optional[np.ndarray] = None
    disp2: Optional[np.ndarray] = None
    seq: int = -1

frame_lock = threading.Lock()
pose_lock = threading.Lock()
latest_frame = LatestFrame()
latest_pose = LatestPose()
stop_event = threading.Event()


from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
# from single_hand_detector import SingleHandDetector
from landmarkfusion import *
import os
import csv
from datetime import datetime
import queue as pyqueue




class FrameWriter:
    def __init__(self, path, fps=30, frame_size=(640, 480), fourcc_list=("mp4v", "XVID")):
        """
        fourcc_list is a fallback list; we'll try them in order.
        """
        self.path = path
        self.fps = fps
        self.frame_size = frame_size
        self.out = None
        self._q = pyqueue.Queue(maxsize=120)  # ~ 4s at 30 FPS
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
                # ignore write errors to avoid blocking the main loop
                pass

    def write(self, frame_bgr):
        """Non-blocking best-effort: drop if queue is full."""
        try:
            self._q.put_nowait(frame_bgr)
        except pyqueue.Full:
            pass  # drop frame to avoid stalling real-time loop

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
    """
    Writes 3 separate CSVs:
      - <base>_raw.csv
      - <base>_calibrated.csv
      - <base>_read.csv

    Each CSV: header = ["timestamp", <finger_1>, <finger_2>, ...]
    Call .write(raw=<list/np1d>, calibrated=<list/np1d>, readback=<list/np1d>)
    """
    def __init__(self, save_dir="./data_logs", base_name="angles", finger_names=None):
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base = os.path.join(save_dir, f"{base_name}_{ts}")

        # If not provided, fall back to generic F1..Fn at runtime when first write happens
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
        # If no names provided, make F1..Fn
        if not self.finger_names:
            self.finger_names = [f"F{i+1}" for i in range(n_cols)]
        header = ["timestamp"] + self.finger_names

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

    def write(self, timestamp, raw=None, calibrated=None, readback=None):
        """
        Provide arrays/lists (same length) for whichever are available this tick.
        We lazily open files on the first write and derive column count from the first non-None vector.
        """
        # Pick any available vector to infer column count on first call
        ref = raw if raw is not None else (calibrated if calibrated is not None else readback)
        if ref is None:
            return
        ref = list(ref)
        self._ensure_open(len(ref))

        if raw is not None:
            self._raw_writer.writerow([timestamp] + list(raw))
        if calibrated is not None:
            self._cal_writer.writerow([timestamp] + list(calibrated))
        if readback is not None:
            self._read_writer.writerow([timestamp] + list(readback))

        # optional: flush periodically to disk
        if int(timestamp) % 2 == 0:
            self._raw_fp.flush()
            self._cal_fp.flush()
            self._read_fp.flush()

    def close(self):
        for fp in (self._raw_fp, self._cal_fp, self._read_fp):
            try:
                if fp: fp.close()
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


coefficients = {
    'Index': [0.789947, 0.974522, -0.000485],
    'Middle': [0.548836, 0.993359, -0.000636],
    'Ring': [0.679707, 0.981869, -0.000526],
    'Pinky': [0.935938, 0.957580, -0.000146],
    'Thumb Flexor': [0.060796, 0.980640, -0.000038],
    'Thumb Rotator': [0.807911, 0.914552, -0.000650]
}

def calibrate(measured_vals):
    command_vals = []
    calibrated = [0.0] * 6
    for i, (finger, coeffs) in enumerate(coefficients.items()):
        a, b, c = coeffs
        poly_eq = Polynomial([a - measured_vals[i], b, c])
        roots = poly_eq.roots()

        if finger == "Thumb Rotator":
            valid_range = (-100, 0)
        else:
            valid_range = (0, 100)

        real_roots = [r.real for r in roots if np.isreal(r) and valid_range[0] <= r.real <= valid_range[1]]

        if real_roots:
            command_val = round(real_roots[0], 2)
        else:
            # Clip manually based on the measured value
            if measured_vals[i] < a + b * valid_range[0] + c * valid_range[0] ** 2:
                command_val = valid_range[0]
            else:
                command_val = valid_range[1]

        command_vals.append(command_val)
        if command_val>30 and command_val<90:
            calibrated[i]=command_val+10
        elif command_val>0:
            calibrated[i]=command_val-5
        else:
            calibrated[i]=command_val-5

            
        
    command_vals=np.array(command_vals)
    # print (f"calibration output,{calibrated}")
    return calibrated



def calibrate2(commanded_angles):
    """
    Takes an array of commanded angles [x0, x1, x2, x3, x4, x5] for 6 fingers,
    and returns the corresponding calibrated (measured) angles using the
    fitted cubic parameters.
    """
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
        if x>40:
            x=x+7
        elif x>10:
            x=x-10
        a, b, c, d = calibration_params[finger_index]
        y = a * x**3 + b * x**2 + c * x + d
        if finger_index == 5:
            y = max(y, -120)
        else:
            y = min(max(y, 0), 120)
            # if y>30:
            #     y=y+(((1/7)*y)-(4.2))
        calibrated.append(y)

    return np.array(calibrated)

def frame_ingest_loop(queue):
    global latest_frame

    while not stop_event.is_set():
        seq, t_capture, bgr1, bgr2 = queue.get()

        with frame_lock:
            latest_frame.seq = seq
            latest_frame.t_capture = t_capture
            latest_frame.bgr1 = bgr1
            latest_frame.bgr2 = bgr2

def tracking_loop(tracker, hands1, hands2):
    global latest_frame, latest_pose
    last_seq = -1

    while not stop_event.is_set():
        with frame_lock:
            if latest_frame.seq == last_seq or latest_frame.bgr1 is None:
                pass
            else:
                seq = latest_frame.seq
                bgr1 = latest_frame.bgr1.copy()
                bgr2 = latest_frame.bgr2.copy()
                t_capture = latest_frame.t_capture
                last_seq = seq
                break_flag = False
                # just to avoid extra nesting
                current = (seq, bgr1, bgr2, t_capture)
                break_flag = True

        if not ('break_flag' in locals() and break_flag):
            time.sleep(0.001)
            continue

        seq, bgr1, bgr2, t_capture = current
        del break_flag
        del current

        pts1, vis1, conf1, res1 = tracker.get_landmarks_and_visibility(bgr1, hands1)
        pts2, vis2, conf2, res2 = tracker.get_landmarks_and_visibility(bgr2, hands2)

        disp1 = bgr1.copy()
        disp2 = bgr2.copy()

        if (
            pts1 is not None and pts2 is not None and
            conf1 >= MIN_HAND_CONF and conf2 >= MIN_HAND_CONF
        ):
            joint_pos, pts3D, frame_score, final_conf, err1, err2 = tracker.process(
                bgr1, bgr2, pts1, pts2, vis1, vis2
            )

            mean_err1 = float(np.mean(err1))
            mean_err2 = float(np.mean(err2))
            proj1 = project_points(P1, pts3D)
            proj2 = project_points(P2, pts3D)

            disp1 = draw_landmarks_with_confidence(disp1, pts1, vis1, ids=DRAW_IDS)
            disp2 = draw_landmarks_with_confidence(disp2, pts2, vis2, ids=DRAW_IDS)

            disp1 = draw_triangulated_landmarks(disp1, proj1, final_conf)
            disp2 = draw_triangulated_landmarks(disp2, proj2, final_conf)

            z_tip = pts3D[8, 2]
            wrist = pts3D[0]

            text1 = f"C1={conf1:.2f}  C2={conf2:.2f}  Score={frame_score:.2f}"
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
        else:
            joint_pos = tracker.prev_joint

        if pts1 is not None:
            disp1 = draw_hand_skeleton(disp1, res1, tracker.expected_label)

        if pts2 is not None:
            disp2 = draw_hand_skeleton(disp2, res2, tracker.expected_label)

        with pose_lock:
            latest_pose.seq = seq
            latest_pose.t_capture = t_capture
            latest_pose.joint_pos = joint_pos
            latest_pose.disp1 = disp1
            latest_pose.disp2 = disp2

def retarget_loop(
    retargeting,
    robot,
    retargeting_to_sapien,
    retargeting_joint_names,
    viewer,
    angle_logger,
    log_file,
    video_writer=None
):
    global latest_pose
    last_seq = -1

    target_joint_names = ['index_q1', 'middle_q1', 'ring_q1', 'pinky_q1', 'thumb_q2', 'thumb_q1']

    while not stop_event.is_set():
        with pose_lock:
            if latest_pose.seq == last_seq:
                viewer.render()
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_event.set()
                continue

            seq = latest_pose.seq
            joint_pos = latest_pose.joint_pos
            disp1 = latest_pose.disp1.copy() if latest_pose.disp1 is not None else None
            disp2 = latest_pose.disp2.copy() if latest_pose.disp2 is not None else None
            t_capture = latest_pose.t_capture

        last_seq = seq

        if disp1 is not None:
            cv2.imshow("cam1", disp1)
            if video_writer is not None:
                video_writer.write(disp1)

        if disp2 is not None:
            cv2.imshow("cam2", disp2)

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

            idxs = [retargeting_joint_names.index(name) for name in target_joint_names]
            hand_angles = qpos[idxs] * 180 / np.pi
            commanded_angles = calibrate2(hand_angles).tolist()

            log_line = f"{str(hand_angles.tolist()):<40} | {str(commanded_angles):<40}\n"
            log_file.write(log_line)

            angle_logger.write(
                timestamp=t_capture,
                raw=hand_angles,
                calibrated=commanded_angles
            )

        viewer.render()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_event.set()
            stop_capture_event.set()

def start_pipeline(
    queue,
    retargeting,
    robot,
    retargeting_to_sapien,
    retargeting_joint_names,
    tracker,
    hands1,
    hands2,
    viewer,
    angle_logger,
    log_file,
    video_writer=None
):
    t1 = threading.Thread(target=frame_ingest_loop, args=(queue,), daemon=True)
    t2 = threading.Thread(target=tracking_loop, args=(tracker, hands1, hands2), daemon=True)
    t3 = threading.Thread(
        target=retarget_loop,
        args=(
            retargeting,
            robot,
            retargeting_to_sapien,
            retargeting_joint_names,
            viewer,
            angle_logger,
            log_file,
            video_writer
        ),
        daemon=True
    )

    t1.start()
    t2.start()
    t3.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.05)
    finally:
        stop_event.set()
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)
        t3.join(timeout=1.0)

def produce_frame(queue: multiprocessing.Queue, stop_event,camera_path: Optional[str] = "realsense"):
    if camera_path == "realsense":
        SERIAL_2 = "213622077408"
        SERIAL_1 = "215322071654"

        pipeline1 = rs.pipeline()
        pipeline2 = rs.pipeline()

        config1 = rs.config()
        config2 = rs.config()

        config1.enable_device(SERIAL_1)
        config2.enable_device(SERIAL_2)

        config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        pipeline1.start(config1)
        print(f"Started RealSense pipeline for camera 1 (serial {SERIAL_1})")
        pipeline2.start(config2)
        print(f"Started RealSense pipeline for camera 2 (serial {SERIAL_2})")

        seq = 0

        while not stop_event.is_set():
            print("Waiting for frames...")
          
            try:
            
                frames1 = pipeline1.wait_for_frames()
                frames2 = pipeline2.wait_for_frames()

                f1 = frames1.get_color_frame()
                f2 = frames2.get_color_frame()

                if not f1 or not f2:
                    continue

                img1 = np.asanyarray(f1.get_data())
                img2 = np.asanyarray(f2.get_data())
                t_capture = time.time()

                item = (seq, t_capture, img1, img2)

                try:
                    queue.put_nowait(item)
                except:
                    try:
                        queue.get_nowait()
                    except:
                        pass
                    try:
                        queue.put_nowait(item)
                    except:
                        pass

                seq += 1

            finally:
                print("Releasing cameras...")

                if pipeline1 is not None:
                    pipeline1.stop()

                if pipeline2 is not None:
                    pipeline2.stop()

def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = "realsense",
):
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    queue = multiprocessing.Queue(maxsize=30)

    producer_process = multiprocessing.Process(
        target=produce_frame,
        args=(queue, stop_capture_event, camera_path)
    )
    producer_process.start()

    
    mp_hands = mp.solutions.hands
    hands1 = mp_hands.Hands(max_num_hands=1)
    hands2 = mp_hands.Hands(max_num_hands=1)

    hand_type_str = "Right" if "right" in str(config_path).lower() else "Left"
    tracker = MultiViewTracker(hand_type_str)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = "data_logs"
    os.makedirs(output_folder, exist_ok=True)
    output_csv = f"{output_folder}/teleop_data_{timestamp}.csv"

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

    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name_str = filepath.stem
    loader.load_multiple_collisions_from_file = True

    if "ability" in robot_name_str:
        loader.scale = 1.5
    elif "dclaw" in robot_name_str:
        loader.scale = 1.25
    elif "allegro" in robot_name_str:
        loader.scale = 1.4
    elif "shadow" in robot_name_str:
        loader.scale = 0.9
    elif "bhand" in robot_name_str:
        loader.scale = 1.5
    elif "leap" in robot_name_str:
        loader.scale = 1.4
    elif "svh" in robot_name_str:
        loader.scale = 1.5

    if "glb" not in robot_name_str:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    if "ability" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))

    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    log_file_path = f"{output_folder}/retargeting_log_{timestamp}.txt"
    log_file = open(log_file_path, "w")
    log_file.write(f"{'Original Angles':<40} | {'Calibrated Angles':<40} | {'Measured Angles'}\n")
    log_file.write("-" * 120 + "\n")

    finger_names = ['index_q1', 'middle_q1', 'ring_q1', 'pinky_q1', 'thumb_q2', 'thumb_q1']
    angle_logger = AngleCSVLogger(
        save_dir="./data_logs",
        base_name="retargeting_angles",
        finger_names=finger_names
    )

    record_video = True
    video_writer = None
    video_path = os.path.join(output_folder, f"retargeting_color_{timestamp}.mp4")

    if record_video:
        video_writer = FrameWriter(
            path=video_path,
            fps=30,
            frame_size=(640, 480),
            fourcc_list=("avc1", "mp4v", "XVID")
        )
        video_writer.start()
        logger.info(f"Video recording → {video_path} (640x480@30)")

    try:
        start_pipeline(
            queue,
            retargeting,
            robot,
            retargeting_to_sapien,
            retargeting_joint_names,
            tracker,
            hands1,
            hands2,
            viewer,
            angle_logger,
            log_file,
            video_writer
        )
    
    except KeyboardInterrupt:
        print("Ctrl+C detected")

    finally:

        print("Shutting down cleanly...")

        # stop threads
        stop_event.set()

        # stop capture process
        stop_capture_event.set()

        # wait for process to exit cleanly
        if producer_process.is_alive():
            producer_process.join(timeout=3.0)

        # fallback (rare)
        if producer_process.is_alive():
            print("Force killing producer")
            producer_process.terminate()

        # close everything else
        if video_writer is not None:
            video_writer.stop()

        log_file.close()
        angle_logger.close()

        cv2.destroyAllWindows()
        

        if video_writer is not None:
            video_writer.stop()

        log_file.close()
        angle_logger.close()

        cv2.destroyAllWindows()

        print("Saved CSVs:", angle_logger.filepaths)
        print(f"📁 Data saved to {output_csv}")
        if record_video:
            print(f"🎥 Video saved to {video_path}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    tyro.cli(main)
