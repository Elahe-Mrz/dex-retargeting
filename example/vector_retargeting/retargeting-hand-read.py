import time
import threading
import multiprocessing
import numpy as np
import cv2
from math import pi
from queue import Empty
from pathlib import Path
from typing import Optional

from ah_wrapper import AHSerialClient
from ah_plotting.plots import CombinedRealTimePlot

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector

RUNNING = True

def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = "realsense"):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while RUNNING:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            queue.put(image)
            time.sleep(1 / 30.0)
    finally:
        pipeline.stop()

def control_hand_thread(hand_client, queue, config_path):
    RetargetingConfig.set_default_urdf_dir(str(Path(config_path).parent))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)
    retargeting_joint_names = retargeting.joint_names

    target_joint_names = ['index_q1', 'middle_q1', 'ring_q1', 'pinky_q1', 'thumb_q2', 'thumb_q1']
    indices = [retargeting_joint_names.index(name) for name in target_joint_names]

    alternate = True

    while RUNNING:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            print("Camera timeout")
            continue

        _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        cv2.imshow("Retargeting", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if joint_pos is not None:
            retargeting_type = retargeting.optimizer.retargeting_type
            indices_mapping = retargeting.optimizer.target_link_human_indices

            if retargeting_type == "position":
                ref_value = joint_pos[indices_mapping, :]
            else:
                ref_value = joint_pos[indices_mapping[1], :] - joint_pos[indices_mapping[0], :]

            qpos = retargeting.retarget(ref_value)
            hand_angles = qpos[indices] * 180 / np.pi

            # Clamp ranges
            hand_angles = np.clip(hand_angles, -100, 100)
            hand_angles[5] = np.clip(hand_angles[5], -100, 0)  # thumb rotator

            print("Sending angles to hand:", hand_angles.tolist())
            hand_client.set_position(hand_angles.tolist(), reply_mode=0 if alternate else 1)
            hand_client.send_command()
            alternate = not alternate

        time.sleep(1 / hand_client.rate_hz)

def main():
    global RUNNING
    robot_name = RobotName.ability
    retargeting_type = RetargetingType.position
    hand_type = HandType.right
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)

    queue = multiprocessing.Queue(maxsize=100)
    camera_process = multiprocessing.Process(target=produce_frame, args=(queue,))
    camera_process.start()

    client = AHSerialClient(write_thread=False)

    control_thread = threading.Thread(
        target=control_hand_thread,
        args=(client, queue, str(config_path)),
    )
    control_thread.start()

    # Start real-time plot
    plotter = CombinedRealTimePlot(client.hand)
    try:
        plotter.start()
    except KeyboardInterrupt:
        pass
    finally:
        RUNNING = False
        camera_process.terminate()
        control_thread.join()
        client.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
