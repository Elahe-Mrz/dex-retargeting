import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional
import pyrealsense2 as rs

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector


def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):

    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")

    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"

    # --------- TWO DETECTORS ----------
    detector1 = SingleHandDetector(hand_type=hand_type, selfie=False)
    detector2 = SingleHandDetector(hand_type=hand_type, selfie=False)

    # --------- LOAD CALIB ----------
    calib = np.load("stereo_cam13_recalibrated.npz", allow_pickle=True)

    K1, D1 = calib["K1"], calib["D1"]
    K2, D2 = calib["K2"], calib["D2"]
    R, T = calib["R"], calib["T"]

    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K2 @ np.hstack([R, T.reshape(3,1)])

    # --------- SAPIEN SETUP (UNCHANGED) ----------
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
    scene.set_environment_map(create_dome_envmap())

    cam = scene.add_camera("cam", 600, 600, 1, 0.1, 10)
    cam.set_local_pose(sapien.Pose([0.5, 0, 0], [0,0,0,-1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_pose(cam.get_local_pose())

    loader = scene.create_urdf_loader()
    loader.load_multiple_collisions_from_file = True

    robot = loader.load(str(config.urdf_path))
    robot.set_pose(sapien.Pose([0,0,-0.15]))

    sapien_joint_names = [j.get_name() for j in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    # --------- TRIANGULATION UTILS ----------
    def triangulate(pts1, pts2):
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        return (pts4d[:3] / pts4d[3]).T

    def project(P, pts):
        pts_h = np.hstack([pts, np.ones((len(pts),1))])
        proj = (P @ pts_h.T).T
        return proj[:,:2] / proj[:,2:3]

    def to_joint(points3d):
        if points3d is None or np.isnan(points3d).any():
            return None
        pts = points3d - points3d[0:1]
        return pts.astype(np.float32)

    prev_joint = None
    alpha = 0.25

    while True:

        try:
            bgr1, bgr2 = queue.get()
        except Empty:
            logger.error("Camera queue timeout")
            return

        rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

        # -------- DETECTION --------
        _, jp1, kp2d_1, _ = detector1.detect(rgb1)
        _, jp2, kp2d_2, _ = detector2.detect(rgb2)

        pts1 = detector1.parse_keypoint_2d(kp2d_1, (480,640)) if kp2d_1 is not None else None
        pts2 = detector2.parse_keypoint_2d(kp2d_2, (480,640)) if kp2d_2 is not None else None
        # -------- VISUALIZATION --------

        vis1 = bgr1.copy()
        vis2 = bgr2.copy()

        if kp2d_1 is not None:
            vis1 = detector1.draw_skeleton_on_image(vis1, kp2d_1, style="default")

        if kp2d_2 is not None:
            vis2 = detector2.draw_skeleton_on_image(vis2, kp2d_2, style="default")

        # Add debug text
        cv2.putText(vis1, f"cam1", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(vis2, f"cam2", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Show both
        cv2.imshow("Camera 1", vis1)
        cv2.imshow("Camera 2", vis2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # -------- TRIANGULATION --------
        tri_pts = None
        valid_count = 0

        if pts1 is not None and pts2 is not None:
            raw3d = triangulate(pts1.astype(np.float32), pts2.astype(np.float32))

            proj1 = project(P1, raw3d)
            proj2 = project(P2, raw3d)

            err1 = np.linalg.norm(proj1 - pts1, axis=1)
            err2 = np.linalg.norm(proj2 - pts2, axis=1)

            valid = (err1 < 8) & (err2 < 8)

            tri_pts = np.full((21,3), np.nan)
            tri_pts[valid] = raw3d[valid]
            valid_count = np.sum(valid)

        # -------- HYBRID --------
        joint_pos = None

        if tri_pts is not None and valid_count > 12:
            joint_pos = to_joint(tri_pts)
            mode = "3D"
        elif jp1 is not None:
            joint_pos = jp1
            mode = "cam1"
        elif jp2 is not None:
            joint_pos = jp2
            mode = "cam2"
        else:
            mode = "none"

        # -------- SMOOTH --------
        if joint_pos is not None:
            if prev_joint is None:
                prev_joint = joint_pos.copy()
            else:
                joint_pos = (1-alpha)*prev_joint + alpha*joint_pos
                prev_joint = joint_pos.copy()

        # -------- RETARGET --------
        if joint_pos is not None:
            try:
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
            except Exception as e:
                logger.warning(f"Retarget failed: {e}")

        # -------- DEBUG --------
        logger.info(f"mode={mode} valid3D={valid_count}")

        viewer.render()
# def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
#     if camera_path is None:
#         cap = cv2.VideoCapture(0)
#     else:
#         cap = cv2.VideoCapture(camera_path)

#     while cap.isOpened():
#         success, image = cap.read()
#         time.sleep(1 / 30.0)
#         if not success:
#             continue
#         queue.put(image)

def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = "realsense"):

    if camera_path == "realsense":

        SERIAL_1 = "215322071654"
        SERIAL_2 = "215222078301"

        WIDTH, HEIGHT, FPS = 640, 480, 30

        # -------- Pipeline 1 --------
        pipe1 = rs.pipeline()
        cfg1 = rs.config()
        cfg1.enable_device(SERIAL_1)
        cfg1.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        pipe1.start(cfg1)

        # -------- Pipeline 2 --------
        pipe2 = rs.pipeline()
        cfg2 = rs.config()
        cfg2.enable_device(SERIAL_2)
        cfg2.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        pipe2.start(cfg2)

        try:
            while True:

                frames1 = pipe1.wait_for_frames()
                frames2 = pipe2.wait_for_frames()

                f1 = frames1.get_color_frame()
                f2 = frames2.get_color_frame()

                if not f1 or not f2:
                    continue

                img1 = np.asanyarray(f1.get_data())
                img2 = np.asanyarray(f2.get_data())

                # push BOTH frames together
                queue.put((img1, img2))

                time.sleep(1 / FPS)

        finally:
            pipe1.stop()
            pipe2.stop()

    else:
        # fallback to single webcam (unchanged)
        if camera_path is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(camera_path)

        while cap.isOpened():
            success, image = cap.read()
            time.sleep(1 / 30.0)
            if not success:
                continue

            # duplicate frame to match expected format
            queue.put((image, image))

def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = "realsense",
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    queue = multiprocessing.Queue(maxsize=1000)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(queue, str(robot_dir), str(config_path))
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
