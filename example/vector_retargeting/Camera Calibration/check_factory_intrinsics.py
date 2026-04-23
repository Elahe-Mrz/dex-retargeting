#!/usr/bin/env python3
"""
check_factory_intrinsics.py
----------------------------
Reads and validates the factory-calibrated intrinsics stored in
RealSense device firmware. No checkerboard, no saved files needed.

Checks every stream the device publishes:
  - Color (RGB)
  - Depth
  - Infrared 1 & 2

Also prints extrinsics between streams (e.g. depth-to-color offset).
"""

import pyrealsense2 as rs
import numpy as np

# ================= CONFIG =================

SERIAL_1 = "215322071654"   # Cam1
SERIAL_2 = "213622077408"   # Cam2

# Resolution you will actually USE in your tracking pipeline
# Intrinsics are resolution-dependent — always check at your working resolution
WIDTH, HEIGHT, FPS = 640, 480, 15

# ==========================================

DISTORTION_MODEL_NAMES = {
    rs.distortion.none                  : "None",
    rs.distortion.modified_brown_conrady: "Modified Brown-Conrady (RealSense standard)",
    rs.distortion.inverse_brown_conrady : "Inverse Brown-Conrady",
    rs.distortion.ftheta                : "F-Theta (fisheye)",
    rs.distortion.brown_conrady         : "Brown-Conrady",
    rs.distortion.kannala_brandt4       : "Kannala-Brandt (fisheye)",
}

def start_pipeline(serial):
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color,       WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth,       WIDTH, HEIGHT, rs.format.z16,  FPS)
    # stream_index (1 or 2) must be the second argument, before width/height
    config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8,   FPS)
    config.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8,   FPS)
    pipeline.start(config)
    return pipeline

def get_all_stream_intrinsics(pipeline):
    """Returns a dict of stream_name -> (intrinsics_object, K, D)"""
    profile = pipeline.get_active_profile()
    streams = profile.get_streams()
    results = {}

    for s in streams:
        try:
            vs   = s.as_video_stream_profile()
            intr = vs.get_intrinsics()
            name = f"{s.stream_name()} (index {s.stream_index()})" \
                   if s.stream_type() == rs.stream.infrared else s.stream_name()

            K = np.array([
                [intr.fx, 0,       intr.ppx],
                [0,       intr.fy, intr.ppy],
                [0,       0,       1       ]
            ], dtype=np.float64)

            D = np.array(intr.coeffs, dtype=np.float64)
            results[name] = (intr, K, D)
        except Exception:
            pass  # non-video streams (motion etc.) — skip

    return results

def get_extrinsics(pipeline, from_stream, from_index, to_stream, to_index):
    """Returns the extrinsics (R 3x3, T 3x1) between two streams."""
    profile = pipeline.get_active_profile()
    try:
        src = profile.get_stream(from_stream, from_index)
        dst = profile.get_stream(to_stream,   to_index)
        ext = src.get_extrinsics_to(dst)
        R   = np.array(ext.rotation,    dtype=np.float64).reshape(3, 3)
        T   = np.array(ext.translation, dtype=np.float64).reshape(3, 1)
        return R, T
    except Exception:
        return None, None

def sanity_check(name, intr, K, D):
    """Validates intrinsics and prints a detailed report."""

    fx, fy  = K[0, 0], K[1, 1]
    cx, cy  = K[0, 2], K[1, 2]

    ratio   = fx / fy
    cx_norm = cx / intr.width
    cy_norm = cy / intr.height

    model_str = DISTORTION_MODEL_NAMES.get(intr.model, str(intr.model))

    print(f"\n  {'─'*50}")
    print(f"  Stream : {name}")
    print(f"  {'─'*50}")
    print(f"  Resolution      : {intr.width} x {intr.height}")
    print(f"  fx              : {fx:.4f} px")
    print(f"  fy              : {fy:.4f} px")
    print(f"  cx (ppx)        : {cx:.4f} px  ({cx_norm:.4f} of width)")
    print(f"  cy (ppy)        : {cy:.4f} px  ({cy_norm:.4f} of height)")
    print(f"  fx / fy ratio   : {ratio:.6f}")
    print(f"  Distortion model: {model_str}")
    print(f"  Coeffs [k1,k2,p1,p2,k3]: {np.round(D, 6).tolist()}")

    issues  = []
    notices = []

    # fx/fy ratio — should be very close to 1.0
    if not (0.995 <= ratio <= 1.005):
        issues.append(f"fx/fy ratio {ratio:.5f} deviates > 0.5% — pixels are not square")
    else:
        notices.append(f"fx/fy ratio {ratio:.5f}  ✓")

    # Principal point — should be near image center
    if not (0.43 <= cx_norm <= 0.57):
        issues.append(f"cx={cx:.1f} ({cx_norm:.3f} of width) — far from center, expected 0.43–0.57")
    else:
        notices.append(f"cx position {cx_norm:.3f} of width  ✓")

    if not (0.43 <= cy_norm <= 0.57):
        issues.append(f"cy={cy:.1f} ({cy_norm:.3f} of height) — far from center, expected 0.43–0.57")
    else:
        notices.append(f"cy position {cy_norm:.3f} of height  ✓")

    # Focal length plausibility for ~90° FOV camera at 640px
    # Typical D435 color: fx ~ 600–650 at 640x480
    if fx < 300 or fx > 1000:
        issues.append(f"fx={fx:.1f} is outside typical range 300–1000 for this resolution")
    else:
        notices.append(f"fx={fx:.1f} within plausible range  ✓")

    # Distortion coefficients — should be small
    if abs(D[0]) > 0.5:
        issues.append(f"k1={D[0]:.4f} is large (|k1| > 0.5) — check firmware")
    if abs(D[1]) > 0.5:
        issues.append(f"k2={D[1]:.4f} is large (|k2| > 0.5)")
    if any(abs(c) > 1.0 for c in D):
        issues.append(f"A distortion coeff exceeds 1.0 — likely corrupted: {D.tolist()}")

    # All-zero distortion is suspicious for color stream
    if "Color" in name and np.all(D == 0):
        issues.append("All distortion coeffs are 0.0 — device may not have published correct values yet")

    print()
    for n in notices:
        print(f"    ✓ {n}")
    if issues:
        print()
        for w in issues:
            print(f"    ⚠  {w}")
        print(f"\n  Result: WARNINGS FOUND — review above")
    else:
        print(f"\n  Result: ALL CHECKS PASSED")

    return issues

def check_device(serial, label):
    print(f"\n{'═'*54}")
    print(f"  {label}   Serial: {serial}")
    print(f"{'═'*54}")

    pipeline = start_pipeline(serial)

    print("  Warming up (30 frames)...")
    for _ in range(30):
        pipeline.wait_for_frames()

    streams    = get_all_stream_intrinsics(pipeline)
    all_issues = {}

    for name, (intr, K, D) in streams.items():
        issues = sanity_check(name, intr, K, D)
        all_issues[name] = issues

    # ---- Depth-to-Color extrinsics (built-in hardware offset) ----
    print(f"\n  {'─'*50}")
    print(f"  Depth → Color extrinsics (hardware offset)")
    print(f"  {'─'*50}")
    R, T = get_extrinsics(pipeline, rs.stream.depth, 0, rs.stream.color, 0)
    if R is not None:
        print(f"  Translation (mm): X={T[0,0]*1000:.2f}  Y={T[1,0]*1000:.2f}  Z={T[2,0]*1000:.2f}")
        eye_diff = np.linalg.norm(R - np.eye(3))
        print(f"  Rotation deviation from identity: {eye_diff:.6f}  (< 0.01 expected)")
        if eye_diff > 0.05:
            print("  ⚠  Rotation between depth and color is large — may affect alignment")
        else:
            print("  ✓  Rotation is near-identity as expected")
    else:
        print("  Could not retrieve depth-to-color extrinsics")

    # ---- IR1 → IR2 baseline (stereo pair) ----
    print(f"\n  {'─'*50}")
    print(f"  IR1 → IR2 baseline (stereo depth baseline)")
    print(f"  {'─'*50}")
    R_ir, T_ir = get_extrinsics(pipeline, rs.stream.infrared, 1, rs.stream.infrared, 2)
    if T_ir is not None:
        baseline_mm = abs(T_ir[0, 0]) * 1000
        print(f"  Baseline: {baseline_mm:.2f} mm  (D435 expected ~50mm, D455 expected ~95mm)")
        if baseline_mm < 20 or baseline_mm > 150:
            print("  ⚠  Baseline is outside expected range — check device model")
        else:
            print("  ✓  Baseline looks correct for this device")
    else:
        print("  Could not retrieve IR1 → IR2 extrinsics")

    pipeline.stop()

    # ---- Summary ----
    print(f"\n  {'─'*50}")
    print(f"  Summary for {label}")
    print(f"  {'─'*50}")
    total_issues = sum(len(v) for v in all_issues.values())
    if total_issues == 0:
        print("  All streams: factory intrinsics look healthy ✓")
    else:
        print(f"  {total_issues} issue(s) found across streams:")
        for stream, issues in all_issues.items():
            if issues:
                for i in issues:
                    print(f"    [{stream}] {i}")

    return total_issues

# ==========================================

if __name__ == "__main__":

    issues1 = check_device(SERIAL_1, "Camera 1")
    issues2 = check_device(SERIAL_2, "Camera 2")

    print(f"\n{'═'*54}")
    print(f"  FINAL RESULT")
    print(f"{'═'*54}")
    if issues1 == 0 and issues2 == 0:
        print("  Both cameras: factory intrinsics are healthy.")
        print("  Safe to use CALIB_FIX_INTRINSIC in stereo calibration.")
    else:
        print("  One or more issues found.")
        print("  Consider running Intel RealSense Dynamic Calibration Tool")
        print("  (open RealSense Viewer → calibration) before proceeding.")
    print()
