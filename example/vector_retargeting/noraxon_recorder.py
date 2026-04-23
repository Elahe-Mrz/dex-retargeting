# noraxon_recorder.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, threading, re, csv
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import requests
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError

SESSION = requests.Session()

# ── naming / parsing helpers ─────────────────────────────────────────────────

def _normalize_type(t: str) -> str:
    return (t.replace('real.', '')
             .replace('vector3.accel', 'acceleration')
             .replace('vector3.rot', 'quaternion')
             .replace('vector3.pos', 'position')
             .replace('switch', 'on/off'))

def _coerce_epoch(ts_raw):
    if ts_raw is None:
        return time.time()
    try:
        ts = float(ts_raw)
    except Exception:
        return time.time()
    if ts > 1e12:
        ts /= 1e9
    elif ts > 1e10:
        ts /= 1e3
    return ts

_emg_in_type  = re.compile(r'\bemg\b', re.I)
_imu_in_type  = re.compile(r'(accel|gyro|mag|quaternion|vector3|pos|orientation|angular)', re.I)
_emg_in_name  = re.compile(r'\bemg\b', re.I)
_imu_in_name  = re.compile(r'\bimu\b', re.I)
_digits_any   = re.compile(r'(\d+)')
_axis_from_id = re.compile(r'\.(a[xyz]|g[xyz]|m[xyz])$', re.I)
_axis_from_name = re.compile(r'\b(Ax|Ay|Az|Gx|Gy|Gz|Mx|My|Mz)\b')

def _axis_suffix(h):
    hid  = h.get('id') or ''
    m    = _axis_from_id.search(hid)
    if m: return m.group(1).lower()
    name = h.get('name') or ''
    m    = _axis_from_name.search(name)
    if m: return m.group(1).lower()
    return None

def _infer_base_label(h, counters):
    name  = (h.get('name') or '').strip()
    ftype = h.get('type') or ''
    if _imu_in_type.search(ftype) or _imu_in_name.search(name):
        m = _digits_any.search(name)
        if not m:
            counters['imu'] += 1
            return f"IMU {counters['imu']}"
        return f"IMU {m.group(1)}"
    if _emg_in_type.search(ftype) or (_emg_in_name.search(name) and not _imu_in_type.search(ftype)):
        m = _digits_any.search(name)
        if not m:
            counters['emg'] += 1
            return f"EMG {counters['emg']}"
        return f"EMG {m.group(1)}"
    base = re.sub(r"[^0-9A-Za-z_]+", "_", name.replace(" ", "_")) or "channel"
    return base


# ── strict / gap-free wide CSV writer ────────────────────────────────────────

class _WideCSVGroup:
    def __init__(self, outfile_base: Path, rate_hz: float,
                 channels: list, strict: bool = True):
        self.strict  = bool(strict)
        rate_tag     = f"{int(round(rate_hz))}Hz"
        stem         = outfile_base.stem
        self.path    = outfile_base.with_name(f"{stem}_{rate_tag}.csv").with_suffix(".csv")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        cols = ["unix_ts"]
        self._col_specs: List[Tuple[int, List[str]]] = []
        for ch in channels:
            base = ch['name']
            ft   = ch['full_type'].lower()
            axis = _axis_suffix({'id': ch.get('id'), 'name': ch['name']})

            if 'vector3.rot' in ch['full_type']:
                sub = [f"{base}_q0", f"{base}_q1", f"{base}_q2", f"{base}_q3"]
            elif 'vector3' in ft:
                if 'accel' in ft:
                    sub = [f"{base}_ax", f"{base}_ay", f"{base}_az"]
                elif 'gyro' in ft or 'angular' in ft:
                    sub = [f"{base}_gx", f"{base}_gy", f"{base}_gz"]
                elif 'mag' in ft:
                    sub = [f"{base}_mx", f"{base}_my", f"{base}_mz"]
                else:
                    sub = [f"{base}_x", f"{base}_y", f"{base}_z"]
            elif any(k in ft for k in ('accel','gyro','angular','mag')) and axis:
                sub = [f"{base}_{axis}"]
            else:
                sub = [base]

            self._col_specs.append((ch['index'], sub))
            cols.extend(sub)

        self.f = self.path.open("w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(cols)
        self.f.flush()

        self.rate    = float(rate_hz)
        self.step    = 1.0 / self.rate
        self.buffers: Dict[int, deque] = {ch['index']: deque() for ch in channels}
        self.n_rows  = 0
        self.epoch0: Optional[float] = None

    def set_epoch0(self, epoch_s: float):
        if self.epoch0 is None:
            self.epoch0 = float(epoch_s)

    def feed_channel(self, ch_index: int, samples_tuple_list: list):
        if samples_tuple_list:
            self.buffers[ch_index].extend(samples_tuple_list)

    def _min_buffer_len(self) -> int:
        return min(len(buf) for buf in self.buffers.values()) if self.buffers else 0

    def drain_rows(self):
        if self.strict:
            k = self._min_buffer_len()
            while k > 0:
                ts  = (self.epoch0 + self.n_rows * self.step
                       if self.epoch0 else time.time())
                row = [ts]
                for ch_index, sub in self._col_specs:
                    tup = self.buffers[ch_index].popleft()
                    row.extend(list(tup) if isinstance(tup, (list, tuple)) else [tup])
                self.w.writerow(row)
                self.n_rows += 1
                k -= 1
            if self.n_rows % 50 == 0:
                self.f.flush()
        else:
            while any(len(b) > 0 for b in self.buffers.values()):
                ts  = (self.epoch0 + self.n_rows * self.step
                       if self.epoch0 else time.time())
                row = [ts]
                for ch_index, sub in self._col_specs:
                    if self.buffers[ch_index]:
                        tup = self.buffers[ch_index].popleft()
                        row.extend(list(tup) if isinstance(tup, (list, tuple)) else [tup])
                    else:
                        row.extend([""] * len(sub))
                self.w.writerow(row)
                self.n_rows += 1
            self.f.flush()

    def close(self):
        try:   self.f.flush()
        finally: self.f.close()


# ── recorder ─────────────────────────────────────────────────────────────────

@dataclass
class NoraxonRecorder:
    ip:                    str             = "127.0.0.1"
    port:                  int             = 9221
    sensors:               str             = ""
    imu_only:              bool            = False
    strict:                bool            = True
    poll_timeout_connect:  float           = 0.5
    poll_timeout_read:     float           = 0.5
    outfile_base:          Optional[str]   = None

    _thread:   Optional[threading.Thread] = field(default=None, init=False)
    _stop:     threading.Event            = field(default_factory=threading.Event, init=False)
    _plots_on: bool                       = field(default=False, init=False)
    _writers:  Optional[Dict]             = field(default=None, init=False)
    _idx2rate: Dict[int, int]             = field(default_factory=dict, init=False)
    _idx2meta: Dict[int, Any]             = field(default_factory=dict, init=False)
    _cfg:      Optional[Dict]             = field(default=None, init=False)

    # ── GUI ring buffer ───────────────────────────────────────────────────────
    # Accumulates (host_t, *scalar_values) tuples for all scalar EMG channels.
    # Shape per row: (1 + n_scalar_channels,)
    # Populated in _run(); read by GUI via get_buffer().
    _gui_buf:      deque = field(default_factory=lambda: deque(maxlen=20000), init=False)
    _gui_buf_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _gui_ch_order: list = field(default_factory=list, init=False)  # channel indices in order

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        self._stop.clear()
        self._gui_buf.clear()
        self._gui_ch_order = []
        self._cfg     = self._init_stream()
        base          = Path(self.outfile_base or "emg_noraxon.csv")
        self._writers = self._make_writers(base, self._cfg, strict=self.strict)
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, save_csv_path: Optional[str] = None):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        saved_path = None
        if self._writers:
            for w in self._writers.values():
                w.close()
                saved_path = str(w.path)
        return saved_path

    def start_preview(self):
        self._stop.clear()
        self._gui_buf.clear()
        self._gui_ch_order = []
        self._cfg     = self._init_stream()
        self._writers = None
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_preview(self):
        return self.stop(save_csv_path=None)

    def start_plotting(self): self._plots_on = True
    def stop_plotting(self):  self._plots_on = False

    # ── GUI buffer read ───────────────────────────────────────────────────────

    def get_buffer(self):
        """
        Returns (t_arr, x_arr) where:
          t_arr : (N,)      host timestamps float64
          x_arr : (N, C)    one column per scalar EMG channel, float32
        Returns (None, None) if buffer is empty.
        """
        with self._gui_buf_lock:
            if not self._gui_buf:
                return None, None
            rows = list(self._gui_buf)
            self._gui_buf.clear()

        arr   = np.array(rows, dtype=np.float64)   # (N, 1+C)
        t_arr = arr[:, 0]
        x_arr = arr[:, 1:].astype(np.float32)
        return t_arr, x_arr

    # ── internals ─────────────────────────────────────────────────────────────

    def _init_stream(self):
        server_url  = f"http://{self.ip}:{self.port}"
        headers_url = f"{server_url}/headers"
        enable_url  = f"{server_url}/enable/"
        disable_url = f"{server_url}/disable/"

        r = SESSION.get(headers_url, timeout=3)
        r.raise_for_status()
        meta    = r.json()
        headers = meta.get('headers', [])
        if not headers:
            raise RuntimeError(
                "No sources at /headers. "
                "Start MR HTTP Streaming on an active measurement.")

        def is_imu(h):
            return bool(_imu_in_type.search(h.get('type') or ''))

        if self.imu_only:
            headers = [h for h in headers if is_imu(h)]

        sensor_set = set()
        s = (self.sensors or "").strip()
        if s:
            for part in s.split(","):
                part = part.strip()
                if not part: continue
                if "-" in part:
                    a, b = part.split("-", 1)
                    sensor_set.update(range(min(int(a),int(b)), max(int(a),int(b))+1))
                else:
                    sensor_set.add(int(part))

        sel = []
        if sensor_set:
            for h in headers:
                name = h.get('name', '')
                m    = _digits_any.search(name)
                sn   = int(m.group(1)) if m else None
                if sn in sensor_set:
                    t = h.get('type') or ''
                    if self.imu_only:
                        if is_imu(h): sel.append(h['index'])
                    else:
                        if re.search(
                            r'emg|accel|gyro|mag|quaternion|vector3|pos|orientation|angular',
                            t, re.I):
                            sel.append(h['index'])
        else:
            sel = [h['index'] for h in headers]

        try:
            for h in meta['headers']:
                SESSION.get(f"{disable_url}{h['index']}", timeout=2)
        except Exception:
            pass
        for i in sel:
            try:
                SESSION.get(f"{enable_url}{i}", timeout=2)
            except Exception:
                pass

        counters = {'emg': 0, 'imu': 0}
        channels = []
        for h in headers:
            if h['index'] not in sel: continue
            base = _infer_base_label(h, counters)
            channels.append({
                'name':        base,
                'id':          h.get('id'),
                'type':        _normalize_type(h['type']),
                'full_type':   h['type'],
                'sample_rate': float(h['samplerate']),
                'units':       h['units'],
                'index':       h['index'],
            })

        # Sort channels by the numeric sensor number in the name so that
        # column order is always EMG 1, EMG 2, … regardless of server order.
        def _sort_key(ch):
            m = _digits_any.search(ch['name'])
            return int(m.group(1)) if m else 9999
        channels.sort(key=_sort_key)

        return {
            'server_url':   server_url,
            'channels':     channels,
            'selected_idx': set(sel),
        }

    def _make_writers(self, base_outfile: Path, cfg: dict, strict: bool):
        groups = defaultdict(list)
        for ch in cfg['channels']:
            key = int(round(ch['sample_rate']))
            groups[key].append(ch)
        writers = {
            rate: _WideCSVGroup(base_outfile, rate, chs, strict=strict)
            for rate, chs in groups.items()
        }
        self._idx2rate = {ch['index']: int(round(ch['sample_rate']))
                         for ch in cfg['channels']}
        self._idx2meta = {ch['index']: ch for ch in cfg['channels']}
        return writers

    def _run(self):
        # Figure out which channel indices are scalar (1 value per sample tick)
        # and record their order for the GUI buffer.
        # Build scalar_indices sorted by sensor number so GUI buffer
        # columns are always in EMG 1, 2, 3… order.
        scalar_ch = [
            ch for ch in self._cfg['channels']
            if 'vector3' not in ch['full_type'].lower()
        ]
        def _ch_sort_key(ch):
            m = _digits_any.search(ch['name'])
            return int(m.group(1)) if m else 9999
        scalar_ch.sort(key=_ch_sort_key)
        scalar_indices = [ch['index'] for ch in scalar_ch]
        self._gui_ch_order = scalar_indices

        last_data_ts      = time.time()
        timeout_no_data   = 10.0

        # per-channel mini-deque for GUI: accumulate until we can emit a row
        # key = channel_index, value = deque of scalar floats
        gui_pending: Dict[int, deque] = {i: deque() for i in scalar_indices}

        try:
            while not self._stop.is_set():
                pkt = None
                try:
                    r = SESSION.get(
                        f"{self._cfg['server_url']}/samples",
                        timeout=(self.poll_timeout_connect, self.poll_timeout_read)
                    )
                    r.raise_for_status()
                    pkt = r.json()
                except ReadTimeout:
                    pass
                except (ConnectionError, HTTPError):
                    time.sleep(0.2)

                if not pkt or not pkt.get("channels"):
                    if time.time() - last_data_ts > timeout_no_data:
                        raise RuntimeError("No MR3 samples for too long.")
                    continue

                last_data_ts = time.time()
                pkt_epoch    = _coerce_epoch(pkt.get("timestamp"))
                host_now     = time.time()

                if self._writers:
                    for w in self._writers.values():
                        w.set_epoch0(pkt_epoch)

                for ch in pkt.get("channels", []):
                    idx = ch.get("index")
                    if idx not in self._cfg["selected_idx"]:
                        continue

                    meta    = self._idx2meta.get(idx)
                    ft      = meta['full_type'].lower() if meta else ''
                    samples = ch.get("samples", []) or []

                    # ── CSV writing ──────────────────────────────────────────
                    if self._writers:
                        rate = self._idx2rate.get(idx)
                        if rate and rate in self._writers:
                            tup_list = []
                            if 'vector3.rot' in (meta['full_type'] if meta else ''):
                                take = len(samples) - (len(samples) % 3)
                                for j in range(0, take, 3):
                                    x, y, z = samples[j:j+3]
                                    q0 = max(0.0, 1.0-(x*x+y*y+z*z))**0.5
                                    tup_list.append((q0, x, y, z))
                            elif 'vector3' in ft:
                                take = len(samples) - (len(samples) % 3)
                                for j in range(0, take, 3):
                                    tup_list.append(
                                        (samples[j], samples[j+1], samples[j+2]))
                            else:
                                tup_list = [(s,) for s in samples]
                            if tup_list:
                                self._writers[rate].feed_channel(idx, tup_list)

                    # ── GUI buffer: scalar channels only ─────────────────────
                    if idx in gui_pending and 'vector3' not in ft:
                        for s in samples:
                            gui_pending[idx].append(float(s))

                # drain CSV
                if self._writers:
                    for w in self._writers.values():
                        w.drain_rows()

                # ── emit GUI rows ────────────────────────────────────────────
                # Emit one row per scalar sample tick, using host_now as timestamp.
                # We emit as many rows as the shortest channel buffer allows.
                if scalar_indices and gui_pending:
                    n_emit = min(len(gui_pending[i]) for i in scalar_indices)
                    if n_emit:
                        # Spread timestamps evenly across this poll interval (50ms typ.)
                        # Use the packet epoch as anchor.
                        rate_hz = self._idx2rate.get(scalar_indices[0], 2000)
                        dt      = 1.0 / rate_hz
                        t0      = pkt_epoch - (n_emit - 1) * dt
                        rows    = []
                        for k in range(n_emit):
                            t_k = t0 + k * dt
                            row = [t_k]
                            for i in scalar_indices:
                                row.append(gui_pending[i].popleft())
                            rows.append(row)
                        with self._gui_buf_lock:
                            self._gui_buf.extend(rows)

        finally:
            pass   # close handled in stop()


# ── standalone CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Noraxon MR3 HTTP streamer → gap-free CSV writer")
    ap.add_argument("--ip",       default="127.0.0.1")
    ap.add_argument("--port",     type=int, default=9221)
    ap.add_argument("--outfile",  default="noraxon_emg.csv")
    ap.add_argument("--duration", type=float, default=10)
    ap.add_argument("--sensors",  default="")
    ap.add_argument("--imu-only", action="store_true")
    ap.add_argument("--strict",   action="store_true", default=True)
    args = ap.parse_args()

    rec = NoraxonRecorder(ip=args.ip, port=args.port,
                          sensors=args.sensors,
                          imu_only=args.imu_only,
                          strict=args.strict,
                          outfile_base=args.outfile)
    print(f"Starting Noraxon recording → {args.outfile} …")
    rec.start()
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        saved = rec.stop()
        print(f"Saved CSV → {saved}")
