def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    import os
    import numpy as np
    import cv2
    from datetime import datetime

    from ah_wrapper import AHSerialClient
    from single_hand_detector import SingleHandDetector
    from dex_retargeting.retargeting_config import RetargetingConfig
    from sapien.utils import Viewer
    import sapien

    # Initialize Ability Hand
    hand_client = AHSerialClient(write_thread=False)
    hand_client.set_position([0, 0, 0, 0, 0, 0], reply_mode=0)
    hand_client.send_command()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data_logs", exist_ok=True)
    log_path = f"data_logs/retargeting_angles_{timestamp}.txt"
    log_file = open(log_path, "w")

    # Setup retargeting
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    config = RetargetingConfig.load_from_file(config_path)
    retargeting = config.build()

    detector = SingleHandDetector(hand_type="Right" if "right" in config_path.lower() else "Left", selfie=False)

    # Set up virtual robot rendering
    scene = sapien.Scene()
    robot = scene.create_urdf_loader().load(config.urdf_path)
    robot.set_qpos(np.zeros(len(robot.get_qpos())))

    viewer = Viewer()
    viewer.set_scene(scene)

    target_joint_names = ['index_q1', 'middle_q1', 'ring_q1', 'pinky_q1', 'thumb_q2', 'thumb_q1']
    retargeting_joint_names = retargeting.joint_names
    joint_indices = [retargeting_joint_names.index(name) for name in target_joint_names]

    try:
        while True:
            try:
                bgr = queue.get(timeout=5)
            except Empty:
                print("❌ Camera queue timeout")
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _, joint_pos, keypoint_2d, _ = detector.detect(rgb)

            if joint_pos is None:
                print("⚠️ Hand not detected.")
                continue

            indices = retargeting.optimizer.target_link_human_indices
            ref_value = joint_pos[indices, :] if retargeting.optimizer.retargeting_type == "POSITION" \
                else joint_pos[indices[1]] - joint_pos[indices[0]]

            qpos = retargeting.retarget(ref_value)
            robot.set_qpos(qpos)

            # Send position to real hand
            angles_deg = (qpos[joint_indices] * 180 / np.pi).tolist()
            hand_client.set_position(angles_deg, reply_mode=0)
            hand_client.send_command()
            time.sleep(0.03)

            # Get measured position
            measured = hand_client.hand.get_position()
            measured = [round(x, 2) for x in measured] if measured else ["None"] * 6

            # Print + Log
            log_line = f"Commanded: {np.round(angles_deg, 2).tolist()} | Measured: {measured}\n"
            print(log_line.strip())
            log_file.write(log_line)
            log_file.flush()

            viewer.render()

    except KeyboardInterrupt:
        print("⛔ Interrupted by user.")
    finally:
        log_file.close()
        hand_client.close()
        print(f"📁 Data saved to {log_path}")
