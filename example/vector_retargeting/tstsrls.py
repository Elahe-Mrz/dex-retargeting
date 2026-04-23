import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("❌ No RealSense devices detected")
else:
    print(f"✅ Found {len(devices)} device(s):")
    for d in devices:
        print(" -", d.get_info(rs.camera_info.name),
              "| Serial:", d.get_info(rs.camera_info.serial_number))