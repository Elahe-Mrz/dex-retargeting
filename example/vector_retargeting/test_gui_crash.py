import sys
import os
import io
import multiprocessing
import traceback

def test_1_cam():
    try:
        from data_collector_gui_2 import DataCollectorGUI
        import tkinter as tk
        root = tk.Tk()
        gui = DataCollectorGUI(root)
        
        gui.retarg_enabled_var.set(True)
        gui.retarg_setup_var.set("1 Camera (teleop_2)")
        gui.retarg_hw_var.set(False)

        print("[TEST] Calling _start_retargeting for 1 Cam")
        gui._start_retargeting()
        
        proc = gui.retarg_consumer
        if proc:
            proc.join(2.0)
            print(f"[TEST] 1-Cam Consumer exit code: {proc.exitcode}")
        else:
            print("[TEST] Consumer process not created")
            
        gui._stop_retargeting()
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()

def test_2_cam():
    try:
        from data_collector_gui_2 import DataCollectorGUI
        import tkinter as tk
        root = tk.Tk()
        gui = DataCollectorGUI(root)
        
        gui.retarg_enabled_var.set(True)
        gui.retarg_setup_var.set("2 Cameras (Async)")
        gui.retarg_hw_var.set(False)

        print("[TEST] Calling _start_retargeting for 2 Cam")
        gui._start_retargeting()
        
        proc = gui.retarg_consumer
        if proc:
            proc.join(2.0)
            print(f"[TEST] 2-Cam Consumer exit code: {proc.exitcode}")
        else:
            print("[TEST] Consumer process not created")
            
        gui._stop_retargeting()
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    test_1_cam()
    test_2_cam()
