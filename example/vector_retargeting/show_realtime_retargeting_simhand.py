import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional
import pyrealsense2 as rs
import os 

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from ah_wrapper import AHSerialClient
from datetime import datetime  

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector

import numpy as np

import math


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

def calibrate_right_hand(desired_angles):
    """
    Self-contained right-hand calibration function with hardcoded coefficients.
    Copy this function into any script to convert desired angles into
    compensated command angles without loading any file.
    """
    if len(desired_angles) != 6:
        raise ValueError("Expected 6 desired angles.")

    finger_names = ["index_q1", "middle_q1", "ring_q1", "pinky_q1", "thumb_q2", "thumb_q1"]
    limits = {
        "index_q1": (0.0, 110.0),
        "middle_q1": (0.0, 110.0),
        "ring_q1": (0.0, 110.0),
        "pinky_q1": (0.0, 110.0),
        "thumb_q2": (0.0, 110.0),
        "thumb_q1": (-110.0, 0.0),
    }
    inverse_coeffs = {
        "index_q1": [1.450168872322505e-05, -0.0020396791082368296, 1.0885138718422227, -0.957344262586079],
        "middle_q1": [2.6234610423498844e-05, -0.003317173488678682, 1.1237917793545344, -1.050577136141581],
        "ring_q1": [1.7450438384571825e-05, -0.002365520378227283, 1.1036365050983854, -1.1205961795813622],
        "pinky_q1": [9.94809381026937e-06, -0.0015651032826367775, 1.0822708367328515, -1.1457815398046043],
        "thumb_q2": [1.284184387618003e-05, -0.00193938962149148, 1.0906360231296044, -0.960763176585547],
        "thumb_q1": [5.025818357530463e-05, 0.014119317005845426, 2.1929171784308124, 30.055337207013633],
    }

    commanded = []
    for finger_name, desired in zip(finger_names, desired_angles):
        value = float(np.polyval(np.asarray(inverse_coeffs[finger_name], dtype=float), float(desired)))
        lower, upper = limits[finger_name]
        commanded.append(float(np.clip(value, lower, upper)))

    return np.array(commanded, dtype=float)

def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):

    hand_client = AHSerialClient(write_thread=False)  


    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

    # Setup
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # Camera
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

    # Load robot and set it to a good pose to take picture
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

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error(
                "Fail to fetch image from camera in 5 secs. Please check your web camera device."
            )
            return

        _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        cv2.imshow("realtime_retargeting_demo", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if joint_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
        else:
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)
            # qpos[0] = qpos[0]*(1+math.tanh(3*(qpos[0]-0.8))) 
            # qpos[1] = qpos[0]*1.05851325+0.72349796 
            # qpos[2] = qpos[2]*(1+math.tanh(3*(qpos[2]-0.85))) 
            # qpos[3] = qpos[2]*1.05851325+0.72349796 
            # qpos[4] = qpos[4]*(1+math.tanh(3*(qpos[4]-0.9)))
            # qpos[5] = qpos[4]*1.05851325+0.72349796 
            # qpos[6] = qpos[6]*(1+math.tanh(3*(qpos[6]-0.8))) 
            # qpos[7] = qpos[6]*1.05851325+0.72349796
            # for joint in robot.get_active_joints():
            #     print(joint.get_name())
            # print("Joint names in URDF:", sapien_joint_names)

            robot.set_qpos(qpos[retargeting_to_sapien])
            target_joint_names = ['index_q1', 'middle_q1', 'ring_q1', 'pinky_q1', 'thumb_q2','thumb_q1' ]
            # [index, middle, ring, pinky, thumb flexor,  thumb rotator]
            indices = [retargeting_joint_names.index(name) for name in target_joint_names]
            hand_angles = qpos[indices] * 180 / np.pi
            print (f"hand angles", hand_angles)

            # Convert radians to degrees 
            # hand_angles = qpos[retargeting_to_sapien[0:6]] * 180 / np.pi
            # calibrated = calibrate_right_hand(hand_angles)
            calibrated= calibrate2(hand_angles)
            hand_client.set_position(calibrated.tolist(), reply_mode=0)
            hand_client.send_command()
            print (f"calibrated", calibrated)

        for _ in range(2):
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
        # Initialize RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                image = np.asanyarray(color_frame.get_data())
                queue.put(image)
                time.sleep(1 / 30.0)
        finally:
            pipeline.stop()

    else:
        # Original webcam code
        if camera_path is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(camera_path)

        while cap.isOpened():
            success, image = cap.read()
            time.sleep(1 / 30.0)
            if not success:
                continue
            queue.put(image)

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
