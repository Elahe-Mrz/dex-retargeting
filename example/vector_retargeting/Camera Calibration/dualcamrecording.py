import pyrealsense2 as rs
import numpy as np
import cv2
import time
from pathlib import Path


def start_pipeline(serial, width=1280, height=720, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipeline.start(config)
    return pipeline


def record_two_realsense(
    serials,
    out_paths=("cam0.mp4", "cam1.mp4"),
    width=1280,
    height=720,
    fps=30
):
    assert len(serials) == 2, "Provide exactly 2 serial numbers"

    # Start pipelines
    pipelines = [start_pipeline(s, width, height, fps) for s in serials]

    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = [
        cv2.VideoWriter(p, fourcc, fps, (width, height))
        for p in out_paths
    ]

    print("[INFO] Recording started")
    print("[INFO] Press 'q' to stop")

    try:
        while True:
            frames = []

            # Grab frames from both cameras
            for pipe in pipelines:
                frameset = pipe.wait_for_frames()
                color_frame = frameset.get_color_frame()

                if not color_frame:
                    frames.append(None)
                    continue

                frame = np.asanyarray(color_frame.get_data())
                frames.append(frame)

            if any(f is None for f in frames):
                print("[WARN] Missing frame")
                continue

            # Write frames
            for frame, writer in zip(frames, writers):
                writer.write(frame)

            # Visualization
            combined = cv2.hconcat(frames)
            cv2.imshow("RealSense Cameras", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for p in pipelines:
            p.stop()
        for w in writers:
            w.release()
        cv2.destroyAllWindows()

        print("[INFO] Recording stopped")


if __name__ == "__main__":
    # 🔴 PUT YOUR SERIAL NUMBERS HERE
    SERIAL_1 = "215322071654"   # Cam1
    SERIAL_2 = "213622077408"   # Cam3

    record_two_realsense(
        serials=(SERIAL_1, SERIAL_2),
        out_paths=("dualcam0.mp4", "dualcam1.mp4"),
        width=1280,
        height=720,
        fps=30
    )