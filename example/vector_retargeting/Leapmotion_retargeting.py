from __future__ import print_function

import numpy as np
import time
import os
from pathlib2 import Path
import sapien
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer
from loguru import logger
import Leap
from dex_retargeting.retargeting_config import RetargetingConfig


class LeapMotionListener(Leap.Listener):
    def __init__(self):
        super(LeapMotionListener, self).__init__()
        self.hand_data = None

    def on_frame(self, controller):
        frame = controller.frame()
        if not frame.hands.is_empty:
            hand = frame.hands[0]
            joints = []
            for finger in hand.fingers:
                for b in range(4):
                    bone = finger.bone(b)
                    pos = bone.next_joint
                    joints.append([pos.x, pos.y, pos.z])
            self.hand_data = np.array(joints)  # shape (20, 3)


def start_retargeting(robot_dir, config_path):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info("Start retargeting with config {}".format(config_path))
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    if "right" in config_path.lower():
        hand_type = "Right"
    else:
        hand_type = "Left"

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

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

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

    robot = loader.load(str(filepath))
    robot.set_pose(sapien.Pose([0, 0, -0.15]))

    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    listener = LeapMotionListener()
    controller = Leap.Controller()
    controller.add_listener(listener)

    logger.info("Started Leap Motion retargeting loop")

    try:
        while True:
            if listener.hand_data is None:
                logger.warning("{} hand is not detected.".format(hand_type))
                time.sleep(0.01)
                continue

            joint_pos = listener.hand_data
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

            for _ in range(2):
                viewer.render()
    finally:
        controller.remove_listener(listener)


def main():
    # Manually specify the robot name, retargeting type, and hand type
    from dex_retargeting.constants import (
        RobotName,
        RetargetingType,
        HandType,
        get_default_config_path,
    )

    robot_name = RobotName.AbilityHand
    retargeting_type = RetargetingType.Vector
    hand_type = HandType.Right

    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"

    start_retargeting(str(robot_dir), str(config_path))


if __name__ == "__main__":
    main()
