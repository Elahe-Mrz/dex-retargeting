import sys
import time
import csv
import threading
from datetime import datetime

from ah_wrapper import AHSerialClient


class HandDataRecorder:
    def __init__(self, reply_mode=0, output_csv="hand_data.csv", client=None):
        self.client_provided = client is not None
        try:
            self.client = client or AHSerialClient(write_thread=False)
        except SystemExit:
            print("❌ Failed to connect to the Ability Hand.")
            sys.exit(1)

        self.hand = self.client.hand
        self.reply_mode = reply_mode
        self.output_csv = output_csv
        self.running = True
        self.thread = None

        # Setup CSV header
        self.header = ["timestamp"]
        self.header += [f'pos_{i}' for i in range(6)]
        self.header += [f'cur_{i}' for i in range(6)]
        self.header += [f'vel_{i}' for i in range(6)]
        self.header += [f'touch_{i}' for i in range(30)]

        self.file = open(self.output_csv, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.header)

    def record_loop(self):
        while self.running:
            try:
                time.sleep(1 / self.client.rate_hz)

                row = [time.time()]

                pos = self.hand.get_position()
                row += pos if pos is not None else [None] * 6

                cur = self.hand.get_current()
                row += cur if cur is not None else [None] * 6

                vel = self.hand.get_velocity()
                row += vel if vel is not None else [None] * 6

                fsr = self.hand.get_fsr()
                row += fsr if fsr is not None else [None] * 30

                self.writer.writerow(row)

            except Exception as e:
                print(f"⚠️ Error during recording: {e}")
                time.sleep(0.1)

    def wait_for_data(self, timeout=5):
        print("⏳ Waiting for hand data...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.hand.get_position() is not None:
                print("✅ Hand data received.")
                return True
            time.sleep(0.1)
        print("❌ No hand data received within timeout.")
        return False

    def start(self):
        self.thread = threading.Thread(target=self.record_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.file.close()
        if not self.client_provided:
            self.client.close()
