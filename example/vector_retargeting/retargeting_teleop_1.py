import multiprocessing
import ctypes
import time
from pathlib import Path
from queue import Empty
from typing import Optional
import os

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
import csv
import queue as pyqueue

from dex_retargeting.constants import (
    RobotName, RetargetingType, HandType, get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector


# ═════════════════════════════════════════════════════════════════════════════
#  Video writer (threaded, non-blocking)
# ═════════════════════════════════════════════════════════════════════════════
class FrameWriter:
    def __init__(self, path, fps=30, frame_size=(640, 480),
                 fourcc_list=("avc1", "mp4v", "XVID")):
        self.path        = path
        self.fps         = fps
        self.frame_size  = frame_size
        self.out         = None
        self._q          = pyqueue.Queue(maxsize=120)
        self._t          = None
        self._stop       = threading.Event()
        self._fourcc_list = fourcc_list

    def _open_writer(self):
        for cc in self._fourcc_list:
            fourcc = cv2.VideoWriter_fourcc(*cc)
            out    = cv2.VideoWriter(self.path, fourcc, self.fps, self.frame_size)
            if out.isOpened():
                self.out = out
                return True
        return False

    def start(self):
        if self.out is None:
            if not self._open_writer():
                raise RuntimeError(
                    f"Could not open VideoWriter for {self.path} "
                    f"with any FOURCC {self._fourcc_list}")
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
            pass  # drop frame rather than stall

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2.0)
        if self.out:
            try: self.out.release()
            except Exception: pass


# ═════════════════════════════════════════════════════════════════════════════
#  Angle CSV logger
#  Columns: frame_capture_t, command_sent_t, <fingers...>
#  frame_capture_t = time.time() stamped immediately after queue.get()
#  command_sent_t  = time.time() stamped after hand_client.send_command()
# ═════════════════════════════════════════════════════════════════════════════
class AngleCSVLogger:
    """
    Writes 3 separate CSVs:
      <base>_raw.csv        – retargeting output angles (deg)
      <base>_calibrated.csv – commanded angles after calibration
      <base>_read.csv       – readback from the hand

    Every row has two timestamps:
      frame_capture_t  – host time when the camera frame was dequeued
      command_sent_t   – host time after send_command() returned
    This lets you quantify the retargeting latency precisely.
    """
    def __init__(self, save_dir="./data_logs", base_name="angles",
                 finger_names=None):
        os.makedirs(save_dir, exist_ok=True)
        ts        = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base = os.path.join(save_dir, f"{base_name}_{ts}")

        self.finger_names = list(finger_names) if finger_names else None
        self._fps   = {k: None for k in ("raw", "cal", "read")}
        self._ws    = {k: None for k in ("raw", "cal", "read")}
        self._opened = False
        self._row_count = 0

    def _ensure_open(self, n_cols):
        if self._opened:
            return
        if not self.finger_names:
            self.finger_names = [f"F{i+1}" for i in range(n_cols)]
        header = ["frame_capture_t", "command_sent_t"] + self.finger_names

        for key, suffix in [("raw","_raw.csv"),
                             ("cal","_calibrated.csv"),
                             ("read","_read.csv")]:
            fp = open(self.base + suffix, "w", newline="")
            w  = csv.writer(fp)
            w.writerow(header)
            self._fps[key] = fp
            self._ws[key]  = w
        self._opened = True

    def write(self, frame_capture_t, command_sent_t,
              raw=None, calibrated=None, readback=None):
        ref = raw if raw is not None else (
              calibrated if calibrated is not None else readback)
        if ref is None:
            return
        self._ensure_open(len(list(ref)))
        prefix = [f"{frame_capture_t:.6f}", f"{command_sent_t:.6f}"]

        if raw        is not None:
            self._ws["raw"].writerow(prefix + list(raw))
        if calibrated is not None:
            self._ws["cal"].writerow(prefix + list(calibrated))
        if readback   is not None:
            self._ws["read"].writerow(prefix + list(readback))

        self._row_count += 1
        if self._row_count % 60 == 0:   # flush ~every 2 s at 30 fps
            for fp in self._fps.values():
                if fp: fp.flush()

    def close(self):
        for fp in self._fps.values():
            try:
                if fp: fp.close()
            except Exception:
                pass
        self._opened = False

    @property
    def filepaths(self):
        return {k: self.base + s for k, s in [
            ("raw","_raw.csv"),
            ("calibrated","_calibrated.csv"),
            ("read","_read.csv")]}


# ═════════════════════════════════════════════════════════════════════════════
#  Shared-memory frame writer
# ═════════════════════════════════════════════════════════════════════════════
def _write_frame_to_shm(shm, frame_rgb: np.ndarray):
    """
    Write an HWC uint8 RGB frame into a multiprocessing.Array.
    Layout: [flag(1), wlo, whi, hlo, hhi, c, ...pixels]
    Uses ctypes.memmove for zero-copy speed (~0.3ms for 640x480).
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
        # Write header (don't set flag yet)
        buf[1] = w & 0xFF;  buf[2] = (w >> 8) & 0xFF
        buf[3] = h & 0xFF;  buf[4] = (h >> 8) & 0xFF
        buf[5] = c
        # Fast pixel copy via memmove
        ctypes.memmove(
            ctypes.addressof(buf) + 6,
            frame_rgb.ctypes.data,
            n
        )
        buf[0] = 1   # flag last — reader sees complete frame
    except Exception:
        pass


def _sapien_cam_to_rgb(cam) -> np.ndarray | None:
    """
    Grab an RGB frame from a SAPIEN camera after take_picture().
    Tries every known API variant across SAPIEN 2.x and 3.x.
    Returns uint8 (H, W, 3) or None on failure.
    """
    # SAPIEN 3.x ─────────────────────────────────────────────────────────────
    # Method A: get_color_rgba()
    try:
        rgba = cam.get_color_rgba()
        if rgba is not None:
            return (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
    except Exception:
        pass

    # Method B: get_picture("Color") — float RGBA
    try:
        rgba = cam.get_picture("Color")
        if rgba is not None:
            return (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
    except Exception:
        pass

    # Method C: get_picture_uint8("Color") — direct uint8
    try:
        rgba = cam.get_picture_uint8("Color")
        if rgba is not None:
            return rgba[:, :, :3]
    except Exception:
        pass

    # SAPIEN 2.x ─────────────────────────────────────────────────────────────
    # Method D: get_color_rgba() returning uint8
    try:
        rgba = cam.get_color_rgba()
        arr  = np.array(rgba)
        if arr.dtype == np.uint8:
            return arr[:, :, :3]
        return (np.clip(arr[:, :, :3], 0, 1) * 255).astype(np.uint8)
    except Exception:
        pass

    return None


# ═════════════════════════════════════════════════════════════════════════════
#  Calibration
# ═════════════════════════════════════════════════════════════════════════════
coefficients = {
    'Index':          [0.789947,  0.974522, -0.000485],
    'Middle':         [0.548836,  0.993359, -0.000636],
    'Ring':           [0.679707,  0.981869, -0.000526],
    'Pinky':          [0.935938,  0.957580, -0.000146],
    'Thumb Flexor':   [0.060796,  0.980640, -0.000038],
    'Thumb Rotator':  [0.807911,  0.914552, -0.000650],
}

calibration_params = {
    0: [0.0000447829, -0.0055038815, 1.2407873038, -2.3608804008],
    1: [0.0000361534, -0.0041874131, 1.1805583523, -1.8249082122],
    2: [0.0000461200, -0.0057085443, 1.2455288954, -2.3906153698],
    3: [0.0000408076, -0.0055052341, 1.2510940056, -2.5055910350],
    4: [0.0000086317, -0.0011940936, 1.0658503059, -0.3982449601],
    5: [0.0000328414,  0.0053339517, 1.2574827929,  0.2875786203],
}

def calibrate2(commanded_angles):
    calibrated = []
    for i, x in enumerate(commanded_angles):
        if x > 40:   x = x + 7
        elif x > 10: x = x - 10
        a, b, c, d = calibration_params[i]
        y = a*x**3 + b*x**2 + c*x + d
        if i == 5:
            y = max(y, -120)
        else:
            y = min(max(y, 0), 120)
        calibrated.append(y)
    return np.array(calibrated)


# ═════════════════════════════════════════════════════════════════════════════
#  Retargeting consumer process
# ═════════════════════════════════════════════════════════════════════════════
def start_retargeting(queue: multiprocessing.Queue,
                      robot_dir: str,
                      config_path: str,
                      save_event=None,
                      cam_shm=None,
                      robot_shm=None):
    """
    Main retargeting loop.

    save_event : multiprocessing.Event (optional)
        While clear  → stream and retarget, do NOT write any files.
        When set     → open output folder (derived from NEUROCAPTURE_BASE_DIR
                       env var) and start saving angles + video.
        Cleared again → flush & close files, keep streaming.

    If save_event is None the old behaviour is preserved (always save).
    """
    always_save  = (save_event is None)
    base_dir     = os.environ.get("NEUROCAPTURE_BASE_DIR", "data_logs")

    hand_client  = AHSerialClient(write_thread=False)
    # HandDataRecorder started unconditionally (manages its own CSV gating)
    tmp_csv      = os.path.join(base_dir, "_teleop_tmp.csv")
    os.makedirs(base_dir, exist_ok=True)
    recorder     = HandDataRecorder(reply_mode=0, output_csv=tmp_csv,
                                    client=hand_client)
    recorder.start()

    # These are opened only when saving starts
    output_folder = None
    record_video  = True
    video_writer  = None
    angle_logger  = None
    log_file      = None
    timestamp     = None
    _was_saving   = False   # track transitions

    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector  = SingleHandDetector(hand_type=hand_type, selfie=False)

    # ── Sapien scene setup ──────────────────────────────────────────────────
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)
    scene  = sapien.Scene()

    render_mat            = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic   = 0.0
    render_mat.roughness  = 0.9
    render_mat.specular   = 0.8
    scene.add_ground(-0.2, render_material=render_mat,
                     render_half_size=[1000, 1000])
    scene.add_directional_light(np.array([1,1,-1]),  np.array([3,3,3]))
    scene.add_point_light(np.array([2, 2, 2]),  np.array([2,2,2]), shadow=False)
    scene.add_point_light(np.array([2,-2, 2]),  np.array([2,2,2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2,0.2,0.2],
                           ground_color=[0.2,0.2,0.2]))
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2,1,2],[0.707,0,0.707,0]), np.array([1,1,1]), 5, 5)

    cam = scene.add_camera(name="Cheese!", width=600, height=600,
                           fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.50,0,0.0],[0,0,0,-1]))

    # When embedded in the GUI we use headless rendering (no Viewer window).
    # We still need to call scene.update_render() + cam.take_picture() each frame.
    _use_viewer = (cam_shm is None and robot_shm is None)
    if _use_viewer:
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame    = False
        viewer.control_window.move_speed           = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())
    else:
        viewer = None

    loader = scene.create_urdf_loader()
    filepath   = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True

    scale_map = {"ability":1.5,"dclaw":1.25,"allegro":1.4,"shadow":0.9,
                 "bhand":1.5,"leap":1.4,"svh":1.5}
    for k,v in scale_map.items():
        if k in robot_name:
            loader.scale = v; break

    filepath = (str(filepath).replace(".urdf","_glb.urdf")
                if "glb" not in robot_name else str(filepath))
    robot    = loader.load(filepath)

    pose_map = {"ability":sapien.Pose([0,0,-0.15]),"shadow":sapien.Pose([0,0,-0.2]),
                "dclaw":sapien.Pose([0,0,-0.15]),"allegro":sapien.Pose([0,0,-0.05]),
                "bhand":sapien.Pose([0,0,-0.2]),"leap":sapien.Pose([0,0,-0.15]),
                "svh":sapien.Pose([0,0,-0.13])}
    for k,v in pose_map.items():
        if k in robot_name:
            robot.set_pose(v); break

    sapien_joint_names    = [j.get_name() for j in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(n) for n in sapien_joint_names]).astype(int)

    finger_names = ['index_q1','middle_q1','ring_q1',
                    'pinky_q1','thumb_q2','thumb_q1']
    video_path   = None   # set when saving starts

    def _open_save_files():
        """Called once when save_event transitions clear→set."""
        nonlocal output_folder, video_writer, angle_logger, log_file
        nonlocal timestamp, video_path
        timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(base_dir,
                                     f"retargeting_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)

        angle_logger = AngleCSVLogger(save_dir=output_folder,
                                      base_name="retargeting_angles",
                                      finger_names=finger_names)
        log_path = os.path.join(output_folder,
                                f"retargeting_log_{timestamp}.txt")
        log_file = open(log_path, "w")
        log_file.write(f"{'frame_t':<18} {'cmd_t':<18} "
                       f"{'Raw':<40} {'Calibrated':<40} Measured\n")
        log_file.write("-" * 140 + "\n")
        video_path = os.path.join(output_folder,
                                  f"retargeting_color_{timestamp}.mp4")
        logger.info(f"Retargeting: saving opened → {output_folder}")

    def _close_save_files():
        """Called once when save_event transitions set→clear (or on exit)."""
        nonlocal video_writer, angle_logger, log_file
        if video_writer is not None:
            video_writer.stop()
            video_writer = None
        if log_file is not None:
            try: log_file.close()
            except Exception: pass
            log_file = None
        if angle_logger is not None:
            angle_logger.close()
            logger.info(f"Retargeting CSVs: {angle_logger.filepaths}")
            angle_logger = None

    try:
        while True:
            # ── Check save_event transition ─────────────────────────────
            currently_saving = always_save or (
                save_event is not None and save_event.is_set())

            if currently_saving and not _was_saving:
                _open_save_files()
                _was_saving = True
            elif not currently_saving and _was_saving:
                _close_save_files()
                _was_saving = False

            # ── Grab frame ─────────────────────────────────────────────────
            try:
                bgr = queue.get(timeout=5)
            except Empty:
                logger.error("No frame for 5 s — exiting retargeting loop.")
                return

            # ★ Timestamp immediately after dequeue
            frame_capture_t = time.time()

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Init video writer on first frame of a save session
            if currently_saving and record_video and video_writer is None and video_path:
                h, w = bgr.shape[:2]
                video_writer = FrameWriter(path=video_path, fps=30,
                                           frame_size=(w,h),
                                           fourcc_list=("avc1","mp4v","XVID"))
                video_writer.start()
                logger.info(f"Video → {video_path} ({w}×{h}@30)")

            _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
            bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d,
                                                   style="default")
            if video_writer is not None and currently_saving:
                video_writer.write(bgr)

            if _use_viewer:
                cv2.imshow("retargeting_live", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # Write camera frame (BGR→RGB, downscaled to 320×240) to shm
            if cam_shm is not None:
                small = cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_LINEAR)
                _write_frame_to_shm(cam_shm,
                                    cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

            if joint_pos is not None:
                retargeting_type = retargeting.optimizer.retargeting_type
                indices          = retargeting.optimizer.target_link_human_indices
                if retargeting_type == "POSITION":
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices   = indices[1, :]
                    ref_value = (joint_pos[task_indices, :]
                                 - joint_pos[origin_indices, :])

                qpos        = retargeting.retarget(ref_value)
                robot.set_qpos(qpos[retargeting_to_sapien])

                target_joint_names = finger_names
                idx_list   = [retargeting_joint_names.index(n)
                               for n in target_joint_names]
                hand_angles       = qpos[idx_list] * 180 / np.pi
                commanded_angles  = calibrate2(hand_angles).tolist()

                hand_client.set_position(commanded_angles, reply_mode=0)
                hand_client.send_command()
                # ★ Second timestamp — after command was physically sent
                command_sent_t = time.time()

                measured = hand_client.hand.get_position()

                # Log latency to text file
                lag_ms = (command_sent_t - frame_capture_t) * 1000
                if currently_saving and log_file is not None:
                    log_file.write(
                        f"{frame_capture_t:<18.6f} {command_sent_t:<18.6f} "
                        f"{str(hand_angles.tolist()):<40} "
                        f"{str(commanded_angles):<40} "
                        f"{str(measured)}  [{lag_ms:.1f}ms]\n")

                if currently_saving and angle_logger is not None:
                    angle_logger.write(
                        frame_capture_t=frame_capture_t,
                        command_sent_t=command_sent_t,
                        raw=hand_angles,
                        calibrated=commanded_angles,
                        readback=measured)

                logger.info(f"TX {commanded_angles}  lag={lag_ms:.0f}ms"
                            f"  {'[SAVING]' if currently_saving else '[streaming]'}")

            if _use_viewer and viewer is not None:
                for _ in range(2):
                    viewer.render()
            elif robot_shm is not None:
                # Headless robot render — tries SAPIEN 3.x and 2.x APIs
                try:
                    scene.update_render()
                    cam.take_picture()
                    rgb = _sapien_cam_to_rgb(cam)
                    if rgb is not None:
                        # Downscale to 320×240 to keep shm writes fast
                        small = cv2.resize(rgb, (320, 240),
                                           interpolation=cv2.INTER_LINEAR)
                        _write_frame_to_shm(robot_shm, small)
                except Exception:
                    pass

    finally:
        _close_save_files()   # no-op if already closed
        try: recorder.stop()
        except Exception: pass
        print("Retargeting process exited.")


# ═════════════════════════════════════════════════════════════════════════════
#  Camera producer process
# ═════════════════════════════════════════════════════════════════════════════
def produce_frame(queue: multiprocessing.Queue,
                  camera_path: Optional[str] = "realsense"):
    if camera_path == "realsense":
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise RuntimeError("pyrealsense2 not installed.")
        pipeline = rs.pipeline()
        cfg      = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(cfg)
        try:
            while True:
                frames       = pipeline.wait_for_frames()
                color_frame  = frames.get_color_frame()
                if not color_frame:
                    continue
                image = np.asanyarray(color_frame.get_data())
                # Use put_nowait with a cap so producer never stalls consumer
                try:
                    queue.put_nowait(image)
                except Exception:
                    pass  # drop frame if queue is full
        finally:
            pipeline.stop()
    else:
        cam_idx = int(camera_path) if str(camera_path).isdigit() else camera_path
        cap     = cv2.VideoCapture(cam_idx)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                time.sleep(0.01)
                continue
            try:
                queue.put_nowait(image)
            except Exception:
                pass
            time.sleep(1 / 30.0)


# ═════════════════════════════════════════════════════════════════════════════
#  Standalone CLI entry point
# ═════════════════════════════════════════════════════════════════════════════
def main(
    robot_name:       RobotName,
    retargeting_type: RetargetingType,
    hand_type:        HandType,
    camera_path:      Optional[str] = "realsense",
):
    """Detect human hand pose from video and retarget to robot."""
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir   = (Path(__file__).absolute().parent.parent.parent
                   / "assets" / "robots" / "hands")

    q        = multiprocessing.Queue(maxsize=1000)
    producer = multiprocessing.Process(target=produce_frame,
                                       args=(q, camera_path))
    consumer = multiprocessing.Process(target=start_retargeting,
                                       args=(q, str(robot_dir),
                                             str(config_path)))
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    tyro.cli(main)
