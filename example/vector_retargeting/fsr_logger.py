import serial
import time
import threading
from datetime import datetime
from collections import deque
import os
import argparse


class FSRLogger:
    def __init__(self,
                 port="/dev/ttyUSB0",
                 baudrate=115200,
                 save_dir="./",
                 base_name="fsr",
                 debug=False):
        self.port      = port
        self.baudrate  = baudrate
        self.save_dir  = (save_dir or "").rstrip("/")
        self.base_name = base_name
        self.debug     = debug

        self.ser      = None
        self.file     = None
        self.running  = False
        self.thread   = None
        self.filename = None

        # latest scalar values (for legacy compat)
        self.latest_values = [0.0, 0.0, 0.0]   # [middle_N, index_N, thumb_N]

        # ── GUI ring buffer ──────────────────────────────────────────────
        # Each entry: (host_t_s, middle_N, index_N, thumb_N)
        # The GUI's _pump_fsr reads from here, thread-safe via a lock.
        self._gui_buf      = deque(maxlen=2000)
        self._gui_buf_lock = threading.Lock()

        # plotting
        self.plotting      = False
        self._plot_thread  = None

        # format detection
        self._format      = None
        self._seen_header = False

    # ── calibration ──────────────────────────────────────────────────────────
    def calibrated_force(self, adc_val):
        return max(0.0, 0.00203 * float(adc_val) - 1.16)

    # ── line parsing ─────────────────────────────────────────────────────────
    def _parse_line(self, line):
        """Return (device_t_s_or_None, middle_adc, index_adc, thumb_adc)."""
        if self._format in (None, "csv"):
            parts = line.split(',')
            if len(parts) == 4:
                try:
                    t_ms = float(parts[0].strip())
                    m = int(float(parts[1]))
                    i = int(float(parts[2]))
                    t = int(float(parts[3]))
                    self._format = "csv"
                    return (t_ms / 1000.0, m, i, t)
                except Exception:
                    pass

        if self._format in (None, "labels"):
            if ("Middle:" in line) and ("Index:" in line) and ("Thumb:" in line):
                vals = {}
                for part in line.split('\t'):
                    if ":" in part:
                        k, v = part.split(":")
                        vals[k.strip().lower()] = int(float(v.strip()))
                if all(k in vals for k in ("middle", "index", "thumb")):
                    self._format = "labels"
                    return (None, vals["middle"], vals["index"], vals["thumb"])

        return (None, None, None, None)

    # ── logging thread ────────────────────────────────────────────────────────
    def _log_loop(self):
        while self.running:
            try:
                raw      = self.ser.readline()
                host_ts  = time.time()
                if not raw:
                    continue
                line = raw.decode(errors='ignore').strip()
                if not line:
                    continue
                if line.lower().startswith("t_ms"):
                    self._seen_header = True
                    continue

                device_t_s, m_adc, i_adc, t_adc = self._parse_line(line)
                if m_adc is None:
                    if self.debug:
                        print(f"[debug] unparsable: {line!r}")
                    continue

                m_n = self.calibrated_force(m_adc)
                i_n = self.calibrated_force(i_adc)
                t_n = self.calibrated_force(t_adc)

                device_ts_out = device_t_s if device_t_s is not None else host_ts

                # ── push to GUI ring buffer ──────────────────────────────
                with self._gui_buf_lock:
                    self._gui_buf.append((host_ts, m_n, i_n, t_n))

                # ── update latest_values (legacy) ────────────────────────
                self.latest_values = [m_n, i_n, t_n]

                # ── write to CSV (only when file is open) ────────────────
                if self.file:
                    self.file.write(
                        f"{device_ts_out:.6f},{host_ts:.6f},"
                        f"{m_adc},{m_n:.4f},{i_adc},{i_n:.4f},{t_adc},{t_n:.4f}\n"
                    )

                if self.debug:
                    print(f"[ok] host_t={host_ts:.3f} "
                          f"ADC=({m_adc},{i_adc},{t_adc}) "
                          f"N=({m_n:.2f},{i_n:.2f},{t_n:.2f})")

            except Exception as e:
                print(f"⚠️ Logging error: {e}")
                continue

    # ── public start/stop ─────────────────────────────────────────────────────
    def start_logging(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
        except serial.SerialException as e:
            raise RuntimeError(f"❌ Could not open port {self.port}: {e}")

        out_dir = self.save_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f"{out_dir}/{self.base_name}_{timestamp_str}.csv"
        self.file = open(self.filename, "w")
        self.file.write(
            "device_t_s,host_t_s,middle_adc,middle_N,"
            "index_adc,index_N,thumb_adc,thumb_N\n"
        )

        # clear GUI buffer at start
        with self._gui_buf_lock:
            self._gui_buf.clear()

        self.running = True
        self.thread  = threading.Thread(target=self._log_loop, daemon=True)
        self.thread.start()
        print(f"✅ Logging started. Saving to {self.filename}")

    def start_streaming(self):
        """Preview mode: open serial but do NOT write to a file."""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
        except serial.SerialException as e:
            raise RuntimeError(f"❌ Could not open port {self.port}: {e}")

        self.filename = None
        self.file     = None

        with self._gui_buf_lock:
            self._gui_buf.clear()

        self.running = True
        self.thread  = threading.Thread(target=self._log_loop, daemon=True)
        self.thread.start()
        print(f"✅ FSR streaming (no save) on {self.port}")

    def stop_logging(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        if self.file:
            self.file.close()
            self.file = None
        if self.ser:
            self.ser.close()
            self.ser = None
        print(f"🛑 Logging stopped. Data saved to {self.filename}")
        return self.filename

    def stop_streaming(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        if self.ser:
            self.ser.close()
            self.ser = None
        print("🛑 FSR streaming stopped.")

    # ── GUI buffer read ───────────────────────────────────────────────────────
    def get_buffer(self):
        """
        Called by the GUI pump every 50 ms.
        Returns (t_arr, x_arr) where:
          t_arr : shape (N,)       – host timestamps (float64)
          x_arr : shape (N, 3)    – [middle_N, index_N, thumb_N]
        Returns (None, None) if no new data.
        """
        with self._gui_buf_lock:
            if not self._gui_buf:
                return None, None
            rows = list(self._gui_buf)
            self._gui_buf.clear()

        import numpy as np
        arr   = np.array(rows, dtype=np.float64)   # (N, 4)
        t_arr = arr[:, 0]
        x_arr = arr[:, 1:].astype(np.float32)       # (N, 3)
        return t_arr, x_arr

    # ── optional standalone plotting ──────────────────────────────────────────
    def start_plotting(self, ax=None, y_max=10.0):
        self._plot_ax   = ax
        self._plot_ymax = y_max
        self.plotting   = True
        self._plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
        self._plot_thread.start()

    def stop_plotting(self):
        self.plotting  = False
        self._plot_ax  = None

    def _plot_loop(self):
        import matplotlib.pyplot as plt
        max_len = 200
        labels  = ["Middle", "Index", "Thumb"]
        data    = [deque([0] * max_len, maxlen=max_len) for _ in range(3)]

        if self._plot_ax is None:
            plt.ion()
            fig, ax = plt.subplots()
        else:
            ax  = self._plot_ax
            fig = ax.figure

        lines = [ax.plot([], [])[0] for _ in range(3)]
        ax.set_xlim(0, max_len)
        ax.set_ylim(0, getattr(self, "_plot_ymax", 10.0))
        ax.set_title("Real-Time FSR Force (N)")
        ax.legend(labels)

        while self.plotting and self.running:
            for i in range(3):
                data[i].append(self.latest_values[i])
                lines[i].set_xdata(range(len(data[i])))
                lines[i].set_ydata(data[i])
            ax.relim()
            ax.autoscale_view(scaley=True)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.05)


# ── standalone CLI ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Standalone FSR logger (auto-detects Arduino formats).")
    parser.add_argument("--port",      default="/dev/ttyUSB0")
    parser.add_argument("--baud",      type=int, default=115200)
    parser.add_argument("--save_dir",  default="./Data")
    parser.add_argument("--base_name", default="fsr")
    parser.add_argument("--plot",      action="store_true")
    parser.add_argument("--ymax",      type=float, default=10.0)
    parser.add_argument("--debug",     action="store_true")
    args = parser.parse_args()

    logger = FSRLogger(port=args.port, baudrate=args.baud,
                       save_dir=args.save_dir, base_name=args.base_name,
                       debug=args.debug)
    try:
        logger.start_logging()
        if args.plot:
            logger.start_plotting(y_max=args.ymax)
        print("⏺ Press Ctrl+C to stop.")
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("❌ Error:", e)
    finally:
        logger.stop_logging()


if __name__ == "__main__":
    main()
