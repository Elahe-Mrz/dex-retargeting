import re

target = '/home/ucml/dex-retargeting/example/vector_retargeting/data_collector_gui_2.py'

with open(target, 'r') as f:
    code = f.read()

p1 = r'''            def _consumer_entry_2cam\(q1, q2, rd, cp, base_dir, save_event, c_shm, r_shm, hwe, ht\):
                import os
                os\.environ\["NEUROCAPTURE_BASE_DIR"\] = base_dir
                start_retargeting\(q1, q2, None, rd, cp, ht, 
                                  save_event=save_event, cam_shm=c_shm, robot_shm=r_shm, hw_retargeting_enabled=hwe\)'''

r1 = '''            def _consumer_entry_2cam(q1, q2, rd, cp, base_dir, save_event, c_shm, r_shm, hwe, ht):
                try:
                    import os
                    os.environ["NEUROCAPTURE_BASE_DIR"] = base_dir
                    start_retargeting(q1, q2, None, rd, cp, ht, 
                                      save_event=save_event, cam_shm=c_shm, robot_shm=r_shm, hw_retargeting_enabled=hwe)
                except Exception as e:
                    import traceback
                    with open("/tmp/consumer_error.txt", "a") as f:
                        f.write(f"2-cam error:\\n{traceback.format_exc()}\\n")
                    raise e'''

p2 = r'''            def _consumer_entry\(q, rd, cp, base_dir, save_event,
                                cam_shm, robot_shm\):
                import os
                os\.environ\["NEUROCAPTURE_BASE_DIR"\] = base_dir
                start_retargeting\(q, rd, cp, save_event=save_event,
                                  cam_shm=cam_shm, robot_shm=robot_shm\)'''

r2 = '''            def _consumer_entry(q, rd, cp, base_dir, save_event,
                                cam_shm, robot_shm):
                try:
                    import os
                    os.environ["NEUROCAPTURE_BASE_DIR"] = base_dir
                    start_retargeting(q, rd, cp, save_event=save_event,
                                      cam_shm=cam_shm, robot_shm=robot_shm)
                except Exception as e:
                    import traceback
                    with open("/tmp/consumer_error.txt", "a") as f:
                        f.write(f"1-cam error:\\n{traceback.format_exc()}\\n")
                    raise e'''

code = code.replace(p1, r1)
code = code.replace(p2, r2)

with open(target, 'w') as f:
    f.write(code)
print("Patch applied.")
