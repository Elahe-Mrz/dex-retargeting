#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroCapture  –  EMG + FSR Data Collector
==========================================
Fixes / improvements in this version
  • Signals now appear: pump loop uses absolute time.time() stamps so the
    rolling-window mask always works.  Falls back gracefully if the recorder
    exposes get_latest_samples() / get_buffer() / _ring_buffer / _data_buf.
  • Stop button is in a sticky footer – always visible, never scrolled away.
  • Instruction panel lives at the top of the right column (large, bold,
    colour-coded) – easy to read from across the room.
  • Unguided mode now writes events.csv with session start + stop timestamps.
  • All other previous improvements retained (FSR optional, save toggle,
    combined embedded plot, dark theme, thread-safe GUI calls).
"""

import tkinter as tk
from tkinter import ttk, filedialog
import threading
import queue
import time
import os
from datetime import datetime
import csv
import json
import tempfile
import shutil
import traceback
import multiprocessing
import signal
import ctypes
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# ── your collectors (must be importable) ─────────────────────────────────────
from mindrove_recorder import MindroveRecorder
from fsr_logger import FSRLogger
from noraxon_recorder import NoraxonRecorder


# ═════════════════════════════════════════════════════════════════════════════
#  Theme constants
# ═════════════════════════════════════════════════════════════════════════════
BG        = "#0d1117"
BG2       = "#161b22"
BG3       = "#21262d"
BG4       = "#30363d"
ACCENT    = "#58a6ff"
ACCENT2   = "#bc8cff"
SUCCESS   = "#3fb950"
WARNING   = "#d29922"
DANGER    = "#f85149"
FG        = "#e6edf3"
FG2       = "#8b949e"
FG3       = "#484f58"
BORDER    = "#30363d"

FONT_MONO  = ("Consolas", 9)
FONT_UI    = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_HEAD  = ("Segoe UI Semibold", 11)

PLOT_BG    = "#0d1117"
EMG_COLORS = ["#58a6ff","#bc8cff","#3fb950","#d29922","#f85149",
              "#39c5cf","#ff7b72","#a5d6ff"]
FSR_COLOR  = "#d29922"
PLOT_HISTORY_S = 6.0


# ═════════════════════════════════════════════════════════════════════════════
#  Trial plan
# ═════════════════════════════════════════════════════════════════════════════
def build_trial_plan(selected_grasp, rest_s=30,
                     forces=("Light","Medium","Strong"),
                     reps_per_force=5, duration_s=15):
    trials = [{"grasp":"Open Hand","force":"Relaxed","duration":float(rest_s)}]
    for f in forces:
        for _ in range(reps_per_force):
            trials.append({"grasp":selected_grasp,"force":f,"duration":float(duration_s)})
            trials.append({"grasp":"Open Hand","force":"Relaxed","duration":5.0})
    return trials


# ═════════════════════════════════════════════════════════════════════════════
#  Thread-safe rolling buffer
# ═════════════════════════════════════════════════════════════════════════════
class RollingBuffer:
    def __init__(self, n_channels=1, maxlen=6000):
        self._lock   = threading.Lock()
        self.n_ch    = n_channels
        self.maxlen  = maxlen
        self._t = np.empty(0, dtype=np.float64)
        self._x = np.empty((0, n_channels), dtype=np.float32)

    def push(self, t_arr, x_arr):
        t_arr = np.asarray(t_arr, dtype=np.float64).ravel()
        x_arr = np.asarray(x_arr, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        with self._lock:
            # If channel count changed (or first push), reset x buffer
            if x_arr.shape[1] != self.n_ch or self._x.shape[1] != x_arr.shape[1]:
                self.n_ch = x_arr.shape[1]
                self._x   = np.empty((0, self.n_ch), dtype=np.float32)
            self._t = np.concatenate([self._t, t_arr])[-self.maxlen:]
            self._x = (np.vstack([self._x, x_arr]) if self._x.shape[0]
                       else x_arr)[-self.maxlen:]

    def snapshot(self):
        with self._lock:
            return self._t.copy(), self._x.copy()

    def clear(self):
        with self._lock:
            self._t = np.empty(0, dtype=np.float64)
            self._x = np.empty((0, self.n_ch), dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  NOTE: Both FSRLogger and NoraxonRecorder now expose get_buffer() which
#  drains their internal deque and returns (t_arr, x_arr).  The GUI pump
#  calls this directly – no guessing needed.
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
#  Main GUI
# ═════════════════════════════════════════════════════════════════════════════
class DataCollectorGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("NeuroCapture  ·  EMG + FSR")
        self.root.geometry("1380x920")
        self.root.configure(bg=BG)
        self.root.minsize(1100, 780)

        self._apply_theme()

        # threading
        self.msg_queue      = queue.Queue()
        # ── unified streaming thread (runs during preview AND recording) ──
        self.stream_thread  = None          # single always-on device thread
        self.stream_stop    = threading.Event()
        # ── save gate: set() = actively recording to disk ────────────────
        self.save_gate      = threading.Event()
        # legacy aliases kept so nothing else breaks
        self.worker_thread  = None
        self.worker_stop    = threading.Event()
        self.preview_thread = None
        self.preview_stop   = threading.Event()
        self.preview_tmpdir = None

        # session
        self.session_ts      = None
        self.session_dir     = None
        self.events_fp       = None
        self.events_writer   = None
        self.events_file     = None
        self.session_start_t = None

        # devices
        self.emg = None
        self.fsr = None

        # retargeting processes
        self.retarg_producer   = None
        self.retarg_consumer   = None
        self.retarg_queue      = None
        self.retarg_save_event = None

        # shared frame buffers (written by consumer, read by GUI)
        # Layout: [flag(1 byte), W(4), H(4), C(4), pixels(W*H*C)]
        # flag=1 means a new frame is ready
        _SHM_MAX = 1 + 5 + 320 * 240 * 3           # fits 320×240 RGB
        self.retarg_cam_shm   = multiprocessing.Array(ctypes.c_uint8, _SHM_MAX)
        self.retarg_robot_shm = multiprocessing.Array(ctypes.c_uint8, _SHM_MAX)
        self._video_after_id  = None
        self._cam_photo       = None   # keep reference to avoid GC
        self._robot_photo     = None

        # plot
        self.emg_buf        = None
        self.fsr_buf        = None
        self._plot_after_id = None
        self._emg_lines     = []
        self._fsr_lines     = []   # one Line2D per FSR channel
        self._ax_emg        = None
        self._ax_fsr        = None
        self._ax_emg_list   = []   # one Axes per EMG channel
        self._n_emg_axes    = 1    # current subplot count
        self._canvas        = None
        self._fig           = None
        self._plot_running  = False

        # guided trials
        self.trials             = []
        self.trial_idx          = -1
        self.trial_running      = False
        self.trial_start_host_s = None
        self.refresh_ms         = 80

        # defaults
        self.force_levels     = ("Light","Medium","Strong")
        self.reps_per_force   = 5
        self.force_duration_s = 15
        self.rest_s           = 30
        self.nx_ip_default    = "127.0.0.1"
        self.nx_port_default  = 9221

        self._build_ui()
        self.root.after(100, self._drain_msgs)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── theme ─────────────────────────────────────────────────────────────────
    def _apply_theme(self):
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except Exception:
            pass

        st.configure(".", background=BG2, foreground=FG, font=FONT_UI,
                     bordercolor=BORDER, relief="flat")
        st.configure("TFrame",       background=BG2)
        st.configure("TLabel",       background=BG2, foreground=FG)
        st.configure("TEntry",       fieldbackground=BG3, foreground=FG,
                     insertcolor=FG, bordercolor=BORDER)
        st.configure("TCombobox",    fieldbackground=BG3, foreground=FG,
                     selectbackground=ACCENT, selectforeground="#fff",
                     arrowcolor=FG, bordercolor=BORDER)
        st.configure("TCheckbutton", background=BG2, foreground=FG,
                     focuscolor=ACCENT, indicatorcolor=BG3, indicatorbackground=BG3)
        st.map("TCheckbutton", indicatorcolor=[("selected", ACCENT)])
        st.configure("TRadiobutton", background=BG2, foreground=FG, focuscolor=ACCENT)
        st.configure("TScrollbar",   background=BG3, troughcolor=BG2, arrowcolor=FG2)
        st.configure("TProgressbar", troughcolor=BG3, background=ACCENT,
                     bordercolor=BG3, lightcolor=ACCENT, darkcolor=ACCENT)

    # ═══════════════════════════════════════════════════════════════════════════
    #  UI build
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── title bar ──────────────────────────────────────────────────────────
        tbar = tk.Frame(self.root, bg=BG, height=52)
        tbar.pack(fill="x")
        tbar.pack_propagate(False)
        tk.Label(tbar, text="⬡ NeuroCapture", bg=BG, fg=ACCENT,
                 font=("Segoe UI Bold", 15)).pack(side="left", padx=18, pady=10)
        tk.Label(tbar, text="EMG + FSR Data Collector",
                 bg=BG, fg=FG2, font=FONT_UI).pack(side="left")
        self.status_badge = tk.Label(tbar, text="● IDLE",
                 bg=BG, fg=FG2, font=("Segoe UI Semibold", 10))
        self.status_badge.pack(side="right", padx=18)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        # ── sticky footer (always-visible buttons) ─────────────────────────────
        footer = tk.Frame(self.root, bg=BG, height=64)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        tk.Frame(footer, bg=BORDER, height=1).pack(fill="x", side="top")

        btn_area = tk.Frame(footer, bg=BG)
        btn_area.place(relx=0.5, rely=0.55, anchor="center")

        def _mk(parent, text, bg, fg, cmd, state="normal"):
            return tk.Button(parent, text=text, bg=bg, fg=fg,
                             disabledforeground=FG3,
                             font=("Segoe UI Semibold", 10),
                             relief="flat", padx=20, pady=7,
                             cursor="hand2", command=cmd, state=state,
                             activebackground=bg, activeforeground=fg)

        self.preview_btn     = _mk(btn_area, "▷  Preview",        ACCENT2, "#fff", self.start_preview)
        self.end_preview_btn = _mk(btn_area, "■  End Preview",     BG3,     FG,    self.stop_preview, "disabled")
        spacer               = tk.Frame(btn_area, bg=BG, width=28)
        self.start_btn       = _mk(btn_area, "⏺  Start Recording", ACCENT,  "#fff", self.start_collection)
        self.stop_btn        = _mk(btn_area, "⏹  Stop",            DANGER,  "#fff", self.stop_collection, "disabled")

        for w in [self.preview_btn, self.end_preview_btn, spacer,
                  self.start_btn, self.stop_btn]:
            w.pack(side="left", padx=4)

        # ── main paned area ────────────────────────────────────────────────────
        pane = tk.PanedWindow(self.root, orient="horizontal",
                              bg=BG, sashwidth=5, sashrelief="flat")
        pane.pack(fill="both", expand=True)

        left = tk.Frame(pane, bg=BG2, width=400)
        left.pack_propagate(False)
        pane.add(left, minsize=340)

        right = tk.Frame(pane, bg=BG)
        pane.add(right, minsize=560)

        self._build_left(left)
        self._build_right(right)

        self._rebuild_trials_from_ui()
        self._apply_guided_visibility()
        self._toggle_source_fields()

    # ── left scrollable panel ─────────────────────────────────────────────────
    def _build_left(self, parent):
        canvas = tk.Canvas(parent, bg=BG2, highlightthickness=0)
        vsb    = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG2)
        wid   = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(wid, width=e.width))

        # SESSION
        self._section(inner, "SESSION")
        f = self._card(inner)
        f.columnconfigure(1, weight=1)

        self._lbl(f, "Save dir", 0)
        self.save_dir_var = tk.StringVar(value="./Data")
        row0 = tk.Frame(f, bg=BG2)
        row0.grid(row=0, column=1, sticky="we", padx=6, pady=3)
        self._entry(row0, self.save_dir_var).pack(side="left", fill="x", expand=True)
        tk.Button(row0, text="…", bg=BG3, fg=FG, relief="flat",
                  padx=8, command=self._choose_dir).pack(side="right", padx=(3, 0))

        self._lbl(f, "Base name", 1)
        self.base_name_var = tk.StringVar(value="session")
        self._entry(f, self.base_name_var).grid(row=1, column=1, sticky="we", padx=6, pady=3)

        # EMG SOURCE
        self._section(inner, "EMG SOURCE")
        fsrc = self._card(inner)
        fsrc.columnconfigure(1, weight=1)

        self.emg_source_var = tk.StringVar(value="noraxon")
        src_row = tk.Frame(fsrc, bg=BG2)
        src_row.grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=4)
        for val, lbl in [("mindrove", "Mindrove"), ("noraxon", "Noraxon MR3 (HTTP)")]:
            tk.Radiobutton(src_row, text=lbl, variable=self.emg_source_var, value=val,
                           bg=BG2, fg=FG, selectcolor=BG3,
                           activebackground=BG2, activeforeground=ACCENT,
                           command=self._toggle_source_fields
                           ).pack(side="left", padx=(0, 14))

        self.nx_frame = tk.LabelFrame(fsrc, text=" Noraxon options ",
                                      bg=BG2, fg=FG2, font=FONT_SMALL,
                                      bd=1, relief="groove")
        self.nx_frame.grid(row=1, column=0, columnspan=2,
                           sticky="we", padx=6, pady=(0, 6))
        self.nx_frame.columnconfigure(1, weight=1)
        self.nx_frame.columnconfigure(3, weight=1)

        def nxl(r, c, t):
            tk.Label(self.nx_frame, text=t, bg=BG2, fg=FG2, font=FONT_UI
                     ).grid(row=r, column=c, sticky="w", padx=(6, 2), pady=3)

        def nxe(r, c, var, w=12):
            e = tk.Entry(self.nx_frame, textvariable=var, bg=BG3, fg=FG,
                         insertbackground=FG, relief="flat", bd=4,
                         font=FONT_UI, width=w)
            e.grid(row=r, column=c, sticky="we", padx=(0, 8), pady=3)
            return e

        nxl(0, 0, "IP")
        self.nx_ip_var = tk.StringVar(value=self.nx_ip_default)
        nxe(0, 1, self.nx_ip_var)
        nxl(0, 2, "Port")
        self.nx_port_var = tk.IntVar(value=self.nx_port_default)
        nxe(0, 3, self.nx_port_var, 7)
        nxl(1, 0, "Sensors (e.g. 1,2-4)")
        self.nx_sensors_var = tk.StringVar(value="")
        nxe(1, 1, self.nx_sensors_var, 20).grid(
            row=1, column=1, columnspan=3, sticky="we", padx=(0, 6), pady=3)

        cr = tk.Frame(self.nx_frame, bg=BG2)
        cr.grid(row=2, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 4))
        self.nx_imu_only_var = tk.BooleanVar(value=False)
        self.nx_strict_var   = tk.BooleanVar(value=True)
        for var, lbl in [(self.nx_imu_only_var, "IMU only"),
                         (self.nx_strict_var, "Strict (gap-free)")]:
            tk.Checkbutton(cr, text=lbl, variable=var, bg=BG2, fg=FG,
                           selectcolor=BG3, activebackground=BG2
                           ).pack(side="left", padx=(0, 14))

        # FSR
        self._section(inner, "FSR (FORCE SENSOR)")
        ffsr = self._card(inner)
        ffsr.columnconfigure(1, weight=1)

        fr0 = tk.Frame(ffsr, bg=BG2)
        fr0.grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 2))
        self.fsr_enabled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(fr0, text="Enable FSR", variable=self.fsr_enabled_var,
                       bg=BG2, fg=FG, selectcolor=BG3, activebackground=BG2,
                       command=self._toggle_fsr_fields).pack(side="left")
        tk.Label(fr0, text="  (uncheck if cable not connected)",
                 bg=BG2, fg=FG3, font=FONT_SMALL).pack(side="left")

        self._lbl(ffsr, "Serial port", 1)
        self.port_var = tk.StringVar(value="/dev/ttyUSB0")
        self.fsr_port_entry = self._entry(ffsr, self.port_var)
        self.fsr_port_entry.grid(row=1, column=1, sticky="we", padx=6, pady=3)

        fr2 = tk.Frame(ffsr, bg=BG2)
        fr2.grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 6))
        self.fsr_save_var = tk.BooleanVar(value=True)
        self.fsr_save_chk = tk.Checkbutton(fr2, text="Save FSR data to disk",
                                           variable=self.fsr_save_var,
                                           bg=BG2, fg=FG, selectcolor=BG3,
                                           activebackground=BG2)
        self.fsr_save_chk.pack(side="left")

        # SESSION CONFIG
        self._section(inner, "SESSION CONFIG")
        fcfg = self._card(inner)
        fcfg.columnconfigure(1, weight=1)

        self.guided_var = tk.BooleanVar(value=True)
        tk.Checkbutton(fcfg, text="Guided mode (auto-advance trials)",
                       variable=self.guided_var, bg=BG2, fg=FG,
                       selectcolor=BG3, activebackground=BG2,
                       command=self._apply_guided_visibility
                       ).grid(row=0, column=0, columnspan=2, sticky="w",
                              padx=6, pady=(6, 2))

        self._lbl(fcfg, "Gesture", 1)
        self.gesture_var = tk.StringVar(value="Power Grasp")
        cb = ttk.Combobox(fcfg, textvariable=self.gesture_var, state="readonly",
                          values=["Power Grasp", "Pinch", "Spherical", "Tripod"],
                          width=20)
        cb.grid(row=1, column=1, sticky="w", padx=6, pady=3)
        cb.bind("<<ComboboxSelected>>", lambda e: self._rebuild_trials_from_ui())

        qr = tk.Frame(fcfg, bg=BG2)
        qr.grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))
        for name in ["Power Grasp", "Pinch", "Spherical", "Tripod"]:
            tk.Button(qr, text=name.split()[0], bg=BG3, fg=FG,
                      relief="flat", padx=10, pady=3, font=FONT_SMALL,
                      command=lambda n=name: self._set_gesture(n)
                      ).pack(side="left", padx=2)

        tk.Label(fcfg,
                 text="Plan: 30s Rest → Light/Medium/Strong × 5 reps × 15s",
                 bg=BG2, fg=ACCENT, font=FONT_SMALL
                 ).grid(row=3, column=0, columnspan=2, sticky="w",
                        padx=6, pady=(0, 6))

        # UPCOMING TRIALS
        self._section(inner, "UPCOMING TRIALS")
        ftlist = self._card(inner)
        self.trials_list = tk.Text(ftlist, height=12, bg=BG3, fg=FG2,
                                   font=FONT_MONO, relief="flat", bd=4,
                                   wrap="none")
        tsb = ttk.Scrollbar(ftlist, command=self.trials_list.yview)
        self.trials_list.configure(yscrollcommand=tsb.set)
        tsb.pack(side="right", fill="y", padx=(0, 3))
        self.trials_list.pack(fill="both", expand=True, padx=(6, 0), pady=4)

        # RETARGETING
        self._section(inner, "RETARGETING (optional)")
        fret = self._card(inner)
        fret.columnconfigure(1, weight=1)

        ret_top = tk.Frame(fret, bg=BG2)
        ret_top.grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 2))
        self.retarg_enabled_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ret_top, text="Enable retargeting recording",
                       variable=self.retarg_enabled_var,
                       bg=BG2, fg=FG, selectcolor=BG3, activebackground=BG2,
                       command=self._toggle_retarg_fields).pack(side="left")

        self._lbl(fret, "Robot name", 1)
        self.retarg_robot_var = tk.StringVar(value="ability")
        self.retarg_robot_entry = self._entry(fret, self.retarg_robot_var, width=18)
        self.retarg_robot_entry.grid(row=1, column=1, sticky="we", padx=6, pady=3)

        self._lbl(fret, "Retarget type", 2)
        self.retarg_type_var = tk.StringVar(value="vector")
        self.retarg_type_entry = self._entry(fret, self.retarg_type_var, width=18)
        self.retarg_type_entry.grid(row=2, column=1, sticky="we", padx=6, pady=3)

        self._lbl(fret, "Hand type", 3)
        self.retarg_hand_var = tk.StringVar(value="right")
        self.retarg_hand_combo = ttk.Combobox(fret, textvariable=self.retarg_hand_var,
            state="readonly", values=["right", "left"], width=10)
        self.retarg_hand_combo.grid(row=3, column=1, sticky="w", padx=6, pady=3)

        self._lbl(fret, "Camera", 4)
        self.retarg_cam_var = tk.StringVar(value="realsense")
        self.retarg_cam_entry = self._entry(fret, self.retarg_cam_var, width=18)
        tk.Label(fret, text="(realsense / 0 / /dev/...)",
                 bg=BG2, fg=FG3, font=FONT_SMALL).grid(row=4, column=2, sticky="w")
        self.retarg_cam_entry.grid(row=4, column=1, sticky="we", padx=6, pady=3)

        tk.Label(fret, text="Saves: angles CSV + video to session folder",
                 bg=BG2, fg=ACCENT, font=FONT_SMALL
                 ).grid(row=5, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 6))

        # store refs for enable/disable toggling
        self._retarg_widgets = [
            self.retarg_robot_entry, self.retarg_type_entry,
            self.retarg_hand_combo,  self.retarg_cam_entry,
        ]
        self._toggle_retarg_fields()  # apply initial disabled state

    # ── right panel ───────────────────────────────────────────────────────────
    def _build_right(self, parent):
        parent.rowconfigure(0, weight=0)   # instruction banner
        parent.rowconfigure(1, weight=0)   # video panels (camera + robot)
        parent.rowconfigure(2, weight=3)   # plot
        parent.rowconfigure(3, weight=0)   # separator
        parent.rowconfigure(4, weight=1)   # console
        parent.columnconfigure(0, weight=1)

        # INSTRUCTION BANNER
        self.instr_frame = tk.Frame(parent, bg=BG3)
        self.instr_frame.grid(row=0, column=0, sticky="nsew")
        self.instr_frame.columnconfigure(0, weight=1)

        self.trial_badge = tk.Label(self.instr_frame,
                                    text="", bg=BG3, fg=FG2,
                                    font=("Segoe UI", 10))
        self.trial_badge.grid(row=0, column=0, sticky="w", padx=16, pady=(10, 0))

        self.instruction = tk.Label(self.instr_frame,
                                    text="Press  ⏺ Start Recording  to begin.",
                                    bg=BG3, fg=FG,
                                    font=("Segoe UI Bold", 28),
                                    wraplength=820, justify="center",
                                    pady=6)
        self.instruction.grid(row=1, column=0, sticky="ew", padx=16)

        self.force_label = tk.Label(self.instr_frame,
                                    text="", bg=BG3, fg=WARNING,
                                    font=("Segoe UI Semibold", 18))
        self.force_label.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 4))

        prog_row = tk.Frame(self.instr_frame, bg=BG3)
        prog_row.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 12))
        self.progress = ttk.Progressbar(prog_row, orient="horizontal",
                                        mode="determinate", maximum=100)
        self.progress.pack(fill="x", expand=True, side="left")
        self.timer_label = tk.Label(prog_row, text="",
                                    bg=BG3, fg=FG2, font=FONT_MONO, width=14)
        self.timer_label.pack(side="right", padx=(8, 0))
        self.progress_row = prog_row

        tk.Frame(parent, bg=BORDER, height=1).grid(row=0, column=0, sticky="sew")

        # VIDEO PANELS (camera feed + sapien robot) — hidden until retargeting starts
        self._video_frame = tk.Frame(parent, bg=BG, height=200)
        self._video_frame.grid(row=1, column=0, sticky="nsew")
        self._video_frame.columnconfigure(0, weight=1)
        self._video_frame.columnconfigure(1, weight=1)
        self._video_frame.grid_remove()   # hidden by default

        # Camera feed canvas
        cam_wrap = tk.Frame(self._video_frame, bg=BG)
        cam_wrap.grid(row=0, column=0, sticky="nsew", padx=(4,2), pady=4)
        tk.Label(cam_wrap, text="CAMERA", bg=BG, fg=FG3,
                 font=("Consolas", 7)).pack(anchor="w", padx=4)
        self._cam_canvas = tk.Canvas(cam_wrap, bg=BG2,
                                     highlightthickness=0, width=320, height=180)
        self._cam_canvas.pack(fill="both", expand=True)

        # Sapien robot canvas
        robot_wrap = tk.Frame(self._video_frame, bg=BG)
        robot_wrap.grid(row=0, column=1, sticky="nsew", padx=(2,4), pady=4)
        tk.Label(robot_wrap, text="ROBOT VIEW", bg=BG, fg=FG3,
                 font=("Consolas", 7)).pack(anchor="w", padx=4)
        self._robot_canvas = tk.Canvas(robot_wrap, bg=BG2,
                                       highlightthickness=0, width=320, height=180)
        self._robot_canvas.pack(fill="both", expand=True)

        # PLOT
        plot_frame = tk.Frame(parent, bg=PLOT_BG)
        plot_frame.grid(row=2, column=0, sticky="nsew")

        self._fig = Figure(facecolor=PLOT_BG, tight_layout={"pad": 1.2})
        self._build_plot_axes()
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().configure(bg=PLOT_BG, highlightthickness=0)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        self._ax_emg.text(0.5, 0.5, "No signal — start Preview or Recording",
                          transform=self._ax_emg.transAxes,
                          ha="center", va="center",
                          color=FG2, fontsize=11, alpha=0.5)
        self._canvas.draw_idle()

        # SEPARATOR
        tk.Frame(parent, bg=BORDER, height=1).grid(row=3, column=0, sticky="ew")

        # LOG CONSOLE
        log_frame = tk.Frame(parent, bg=BG)
        log_frame.grid(row=4, column=0, sticky="nsew")
        log_frame.rowconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)
        tk.Label(log_frame, text=" LOG", bg=BG, fg=FG2,
                 font=("Consolas", 8)).grid(row=0, column=0, sticky="w",
                                            padx=8, pady=(4, 1))
        self.console = tk.Text(log_frame, bg=BG2, fg=FG2, font=FONT_MONO,
                               relief="flat", bd=0, wrap="word", height=6)
        csb = ttk.Scrollbar(log_frame, command=self.console.yview)
        self.console.configure(yscrollcommand=csb.set)
        csb.grid(row=1, column=1, sticky="ns")
        self.console.grid(row=1, column=0, sticky="nsew",
                          padx=(8, 0), pady=(0, 6))

    # ── plot axes ──────────────────────────────────────────────────────────────
    def _build_plot_axes(self, n_emg_ch=1):
        """
        Build axes layout:
          - One subplot per EMG channel (stacked, shared x-axis)
          - One FSR subplot below (if FSR enabled), also shared x
        n_emg_ch  : number of EMG channels (determines row count)
        """
        self._fig.clear()
        use_fsr = getattr(self, "fsr_enabled_var", None)
        use_fsr = use_fsr.get() if use_fsr else True

        n_emg_ch = max(1, int(n_emg_ch))
        n_rows   = n_emg_ch + (1 if use_fsr else 0)

        # height ratios: each EMG row = 3 units, FSR row = dynamically scaled to prevent squashing
        ratios = [3] * n_emg_ch + ([max(4.0, 1.5 * n_emg_ch)] if use_fsr else [])
        gs = gridspec.GridSpec(n_rows, 1, figure=self._fig,
                               height_ratios=ratios,
                               hspace=0.04)

        # EMG axes
        self._ax_emg_list = []
        for i in range(n_emg_ch):
            if i == 0:
                ax = self._fig.add_subplot(gs[i])
            else:
                ax = self._fig.add_subplot(gs[i],
                                           sharex=self._ax_emg_list[0])
            self._ax_emg_list.append(ax)
            ax.set_facecolor(PLOT_BG)
            ax.tick_params(colors=FG2, labelsize=7)
            ax.tick_params(axis="x",
                           labelbottom=(i == n_emg_ch - 1 and not use_fsr))
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.grid(True, color=BORDER, lw=0.4, alpha=0.6)
            col = EMG_COLORS[i % len(EMG_COLORS)]
            ax.set_ylabel(f"CH{i+1}", color=col, fontsize=7, rotation=0,
                          labelpad=24)
            ax.yaxis.set_label_position("left")
            if i == 0:
                ax.set_title("Live EMG", color=FG, fontsize=9, pad=5)

        # Keep single _ax_emg pointing to first for compat
        self._ax_emg = self._ax_emg_list[0] if self._ax_emg_list else None

        # FSR axis
        if use_fsr:
            self._ax_fsr = self._fig.add_subplot(
                gs[n_emg_ch],
                sharex=self._ax_emg_list[0] if self._ax_emg_list else None)
            self._ax_fsr.set_facecolor(PLOT_BG)
            self._ax_fsr.tick_params(colors=FG2, labelsize=7)
            for sp in self._ax_fsr.spines.values():
                sp.set_edgecolor(BORDER)
            self._ax_fsr.grid(True, color=BORDER, lw=0.4, alpha=0.6)
            self._ax_fsr.set_ylabel("Force (N)", color=FSR_COLOR,
                                    fontsize=7)
            self._ax_fsr.set_xlabel("Time (s)", color=FG2, fontsize=8)
        else:
            self._ax_fsr = None
            if self._ax_emg_list:
                self._ax_emg_list[-1].set_xlabel("Time (s)", color=FG2,
                                                  fontsize=8)

        self._emg_lines  = []   # one Line2D per EMG channel
        self._fsr_lines  = []   # one Line2D per FSR channel
        self._n_emg_axes = n_emg_ch

    # ── small helpers ─────────────────────────────────────────────────────────
    def _section(self, parent, text):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x")
        tk.Label(f, text=f"  {text}", bg=BG, fg=FG3,
                 font=("Consolas", 8)).pack(side="left", padx=(8, 0), pady=(8, 2))
        tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x",
                                               expand=True, padx=(6, 0))

    def _card(self, parent):
        f = tk.Frame(parent, bg=BG2)
        f.pack(fill="x", padx=10, pady=(0, 4))
        return f

    def _lbl(self, parent, text, row):
        tk.Label(parent, text=text, bg=BG2, fg=FG2, font=FONT_UI
                 ).grid(row=row, column=0, sticky="w", padx=(6, 0), pady=3)

    def _entry(self, parent, var, width=22):
        return tk.Entry(parent, textvariable=var, bg=BG3, fg=FG,
                        insertbackground=FG, relief="flat", bd=4,
                        font=FONT_UI, width=width)

    def _set_status(self, text, color=FG2):
        self.status_badge.config(text=f"●  {text}", fg=color)

    def _toggle_source_fields(self):
        nx = self.emg_source_var.get() == "noraxon"
        st = "normal" if nx else "disabled"
        for w in self.nx_frame.winfo_children():
            try:
                w.configure(state=st)
            except tk.TclError:
                pass

    def _toggle_fsr_fields(self):
        en = self.fsr_enabled_var.get()
        st = "normal" if en else "disabled"
        self.fsr_port_entry.configure(state=st)
        self.fsr_save_chk.configure(state=st)
        self._build_plot_axes(n_emg_ch=getattr(self, "_n_emg_axes", 1))
        if self._canvas:
            self._canvas.draw_idle()

    def _toggle_retarg_fields(self):
        en = self.retarg_enabled_var.get()
        st = "normal" if en else "disabled"
        for w in self._retarg_widgets:
            try: w.configure(state=st)
            except tk.TclError: pass

    # ── Embedded video feed ───────────────────────────────────────────────────
    def _start_video_feed(self):
        """Show video panel and start polling shared frame buffers."""
        self._video_frame.grid()
        self._poll_video_frames()

    def _stop_video_feed(self):
        self._video_frame.grid_remove()
        if self._video_after_id:
            self.root.after_cancel(self._video_after_id)
            self._video_after_id = None
        self._cam_photo   = None
        self._robot_photo = None

    def _poll_video_frames(self):
        """Called every 40 ms on the main thread to blit new frames into canvases."""
        self._blit_shm(self.retarg_cam_shm,   self._cam_canvas,   "_cam_photo")
        self._blit_shm(self.retarg_robot_shm, self._robot_canvas, "_robot_photo")
        self._video_after_id = self.root.after(40, self._poll_video_frames)

    def _blit_shm(self, shm, canvas, photo_attr):
        """
        Read one frame from shared memory and display it on canvas.
        Uses ctypes.memmove for fast pixel copy — no Python list overhead.
        """
        try:
            buf = shm.get_obj()
            if buf[0] != 1:       # flag byte — 0 = no new frame
                return
            buf[0] = 0            # clear flag immediately
            w = buf[1] | (buf[2] << 8)
            h = buf[3] | (buf[4] << 8)
            c = buf[5]
            n = w * h * c
            if n <= 0 or 6 + n > len(buf):
                return

            # Fast zero-copy read via memmove into a numpy array
            arr = np.empty(n, dtype=np.uint8)
            ctypes.memmove(
                arr.ctypes.data,
                ctypes.addressof(buf) + 6,
                n
            )
            arr = arr.reshape((h, w, c))

            # Resize to fit canvas preserving aspect ratio
            cw = canvas.winfo_width()  or 320
            ch = canvas.winfo_height() or 180
            scale = min(cw / max(w, 1), ch / max(h, 1))
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            img   = Image.fromarray(arr).resize((nw, nh), Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            setattr(self, photo_attr, photo)
            canvas.delete("all")
            canvas.create_image(cw // 2, ch // 2, anchor="center", image=photo)
        except Exception:
            pass

    def _set_gesture(self, name):
        self.gesture_var.set(name)
        self._rebuild_trials_from_ui()

    def _choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.save_dir_var.get() or ".")
        if d:
            self.save_dir_var.set(d)

    def _log(self, msg):
        self.msg_queue.put(msg)

    def _drain_msgs(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                ts  = time.strftime("%H:%M:%S")
                self.console.insert("end", f"[{ts}]  {msg}\n")
                self.console.see("end")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_msgs)

    def _rebuild_trials_from_ui(self):
        sel = self.gesture_var.get().strip() or "Power Grasp"
        self.trials = build_trial_plan(
            sel, rest_s=self.rest_s,
            forces=self.force_levels,
            reps_per_force=self.reps_per_force,
            duration_s=self.force_duration_s)
        self._refresh_trial_preview()

    def _refresh_trial_preview(self):
        if not hasattr(self, "trials_list"):
            return
        self.trials_list.delete("1.0", "end")
        self.trials_list.tag_configure("rest",  foreground=FG2)
        self.trials_list.tag_configure("trial", foreground=FG)
        self.trials_list.tag_configure("idx",   foreground=ACCENT)
        for i, tr in enumerate(self.trials):
            tag = "rest" if tr["grasp"] == "Open Hand" else "trial"
            self.trials_list.insert("end", f" {i+1:02d} ", "idx")
            self.trials_list.insert(
                "end",
                f"{'REST' if tr['grasp']=='Open Hand' else tr['grasp']:<14}"
                f"  {tr['force']:<8}  {tr['duration']:>4.0f}s\n",
                tag)

    def _apply_guided_visibility(self):
        guided = self.guided_var.get()
        if guided:
            self.trial_badge.grid()
            self.force_label.grid()
            self.progress_row.grid()
        else:
            self.trial_badge.grid_remove()
            self.force_label.grid_remove()
            self.progress_row.grid_remove()

    # ═══════════════════════════════════════════════════════════════════════════
    #  Live plot  (main-thread only, 100 ms refresh)
    # ═══════════════════════════════════════════════════════════════════════════
    def _start_plot_loop(self):
        if self._plot_running:
            return
        self._plot_running = True
        self._build_plot_axes()
        self._canvas.draw_idle()
        self._schedule_plot()

    def _stop_plot_loop(self):
        self._plot_running = False
        if self._plot_after_id:
            self.root.after_cancel(self._plot_after_id)
            self._plot_after_id = None

    def _schedule_plot(self):
        if not self._plot_running:
            return
        self._update_plot()
        self._plot_after_id = self.root.after(100, self._schedule_plot)

    def _update_plot(self):
        try:
            now   = time.time()
            t_min = now - PLOT_HISTORY_S

            # ── EMG ──────────────────────────────────────────────────────────
            if self.emg_buf is not None:
                t_arr, x_arr = self.emg_buf.snapshot()
                if t_arr.size:
                    mask   = t_arr >= t_min
                    t_plot = t_arr[mask] - now
                    x_plot = x_arr[mask]
                    if x_plot.ndim == 1:
                        x_plot = x_plot.reshape(-1, 1)
                    n_ch = x_plot.shape[1]

                    # Rebuild axes if channel count changed (first data arrives)
                    if n_ch != getattr(self, "_n_emg_axes", 1) or len(self._emg_lines) == 0:
                        self._build_plot_axes(n_emg_ch=n_ch)
                        self._canvas.draw_idle()

                    # Create one Line2D per channel in its own subplot
                    while len(self._emg_lines) < n_ch:
                        ch_idx = len(self._emg_lines)
                        col    = EMG_COLORS[ch_idx % len(EMG_COLORS)]
                        ax     = self._ax_emg_list[ch_idx]
                        ln,    = ax.plot([], [], color=col, lw=0.9, alpha=0.9)
                        self._emg_lines.append(ln)

                    for ch in range(n_ch):
                        self._emg_lines[ch].set_data(t_plot, x_plot[:, ch])
                        ax  = self._ax_emg_list[ch]
                        ax.set_xlim(-PLOT_HISTORY_S, 0)
                        col = x_plot[:, ch]
                        if col.size:
                            lo, hi = col.min(), col.max()
                            pad = max((hi - lo) * 0.1, 1e-4)
                            ax.set_ylim(lo - pad, hi + pad)

            # ── FSR ───────────────────────────────────────────────────────────
            if self._ax_fsr is not None and self.fsr_buf is not None:
                t_arr, x_arr = self.fsr_buf.snapshot()
                if t_arr.size:
                    mask   = t_arr >= t_min
                    t_plot = t_arr[mask] - now
                    x_plot = x_arr[mask]
                    if x_plot.ndim == 1:
                        x_plot = x_plot.reshape(-1, 1)
                    n_fch = x_plot.shape[1]

                    FSR_CH_COLORS = [FSR_COLOR, "#06b6d4", "#a3e635"]
                    FSR_CH_LABELS = ["Middle", "Index", "Thumb"]
                    while len(self._fsr_lines) < n_fch:
                        idx = len(self._fsr_lines)
                        col = FSR_CH_COLORS[idx % len(FSR_CH_COLORS)]
                        lbl = FSR_CH_LABELS[idx] if idx < len(FSR_CH_LABELS) else f"Ch{idx}"
                        ln, = self._ax_fsr.plot([], [], color=col, lw=1.3, label=lbl)
                        self._fsr_lines.append(ln)
                        self._ax_fsr.legend(
                            fontsize=7, loc="upper left",
                            facecolor=PLOT_BG, labelcolor=FG2,
                            framealpha=0.6, edgecolor=BORDER)

                    for fc in range(n_fch):
                        self._fsr_lines[fc].set_data(t_plot, x_plot[:, fc])

                    self._ax_fsr.set_xlim(-PLOT_HISTORY_S, 0)
                    if x_plot.size:
                        lo, hi = x_plot.min(), x_plot.max()
                        pad = max((hi - lo) * 0.1, 1.0)
                        self._ax_fsr.set_ylim(lo - pad, hi + pad)

            self._canvas.draw_idle()
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════════════
    #  Device helpers
    # ═══════════════════════════════════════════════════════════════════════════
    def _init_emg(self, save: bool):
        src = self.emg_source_var.get()
        if src == "noraxon":
            nx_base = (os.path.join(self.session_dir, "noraxon")
                       if save and self.session_dir else None)
            self.emg = NoraxonRecorder(
                ip=self.nx_ip_var.get().strip(),
                port=int(self.nx_port_var.get() or self.nx_port_default),
                sensors=self.nx_sensors_var.get().strip(),
                imu_only=bool(self.nx_imu_only_var.get()),
                strict=bool(self.nx_strict_var.get()),
                outfile_base=nx_base)
        else:
            self.emg = MindroveRecorder()

        if not save and hasattr(self.emg, "start_preview"):
            self.emg.start_preview()
        else:
            self.emg.start()

        self.emg_last_t = time.time()
        self.emg_buf    = RollingBuffer(n_channels=1, maxlen=8000)
        self._log(f"EMG ({src}) started")

    def _teardown_emg(self, save: bool):
        try:
            if self.emg:
                if save and self.session_dir:
                    path = self.emg.stop(
                        save_csv_path=os.path.join(self.session_dir, "emg"))
                    if path:
                        self._log(f"EMG saved → {path}")
                else:
                    try:
                        self.emg.stop(save_csv_path=None)
                    except TypeError:
                        self.emg.stop()
        except Exception as e:
            self._log(f"EMG stop error: {e}")
        self.emg     = None
        self.emg_buf = None

    def _init_fsr(self, save: bool):
        if not self.fsr_enabled_var.get():
            self._log("FSR disabled — skipping.")
            return
        port  = self.port_var.get().strip() or "/dev/ttyUSB0"
        s_dir = (self.session_dir if (save and self.session_dir)
                 else (self.preview_tmpdir or ""))
        try:
            self.fsr = FSRLogger(port=port, baudrate=115200,
                                 save_dir=s_dir, base_name="fsr")
            if not save and hasattr(self.fsr, "start_streaming"):
                self.fsr.start_streaming()
            else:
                self.fsr.start_logging()
            self.fsr_last_t = time.time()
            self.fsr_buf    = RollingBuffer(n_channels=3, maxlen=4000)
            self._log(f"FSR started on {port}  (save={'yes' if save else 'no'})")
        except Exception as e:
            self._log(f"FSR init FAILED ({e}) — continuing without FSR.")
            self.fsr     = None
            self.fsr_buf = None

    def _teardown_fsr(self, save: bool):
        try:
            if self.fsr:
                if save and self.session_dir:
                    path = self.fsr.stop_logging()
                    if path:
                        self._log(f"FSR saved → {path}")
                else:
                    if hasattr(self.fsr, "stop_streaming"):
                        self.fsr.stop_streaming()
                    elif hasattr(self.fsr, "stop_logging"):
                        p = self.fsr.stop_logging()
                        if p:
                            try:
                                os.remove(p)
                            except Exception:
                                pass
        except Exception as e:
            self._log(f"FSR stop error: {e}")
        self.fsr     = None
        self.fsr_buf = None

    # ── sample pump  (worker thread, every 50 ms) ─────────────────────────────
    def _pump(self):
        """
        Both FSRLogger and NoraxonRecorder expose get_buffer() which drains
        their internal deque and returns (t_arr, x_arr) – or (None, None).
        """
        if self.emg is not None and self.emg_buf is not None:
            try:
                t, x = self.emg.get_buffer()
                if t is not None and len(t):
                    self.emg_buf.push(t, x)
            except Exception:
                pass

        if self.fsr is not None and self.fsr_buf is not None:
            try:
                t, x = self.fsr.get_buffer()
                if t is not None and len(t):
                    self.fsr_buf.push(t, x)
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════════════
    #  Preview
    # ═══════════════════════════════════════════════════════════════════════════
    # ─────────────────────────────────────────────────────────────────────────
    # Unified streaming (preview = stream, recording = stream + save)
    # ─────────────────────────────────────────────────────────────────────────
    def _streaming_active(self):
        return self.stream_thread and self.stream_thread.is_alive()

    def start_preview(self):
        """Start devices streaming (no saving).  Retargeting also starts here."""
        if self._streaming_active():
            return
        if self.worker_thread and self.worker_thread.is_alive():
            self._log("Recording in progress — stop first.")
            return

        self.stream_stop.clear()
        self.save_gate.clear()        # not saving yet

        self.preview_btn.config(state="disabled")
        self.end_preview_btn.config(state="normal")
        self.start_btn.config(state="normal")   # allow Start while streaming
        self._set_status("STREAMING", ACCENT2)
        self.instruction.config(
            text="Streaming… press  ⏺ Start Recording  when ready.", fg=FG2)
        self.force_label.config(text="")
        self._log("Streaming started (no data saved yet).")

        self.stream_thread = threading.Thread(
            target=self._run_stream, daemon=True)
        self.stream_thread.start()

        # Start retargeting alongside streaming
        if self.retarg_enabled_var.get():
            self.root.after(200, self._start_retargeting)

    def stop_preview(self):
        """Stop all streaming (and recording if active)."""
        if self.save_gate.is_set():
            # finalise recording first
            self._finalise_recording()
        if not self._streaming_active():
            return
        self._log("Stopping stream…")
        self.end_preview_btn.config(state="disabled")
        self.stream_stop.set()
        # retargeting stops when stream stops
        self.root.after(0, self._stop_retargeting)

    def _run_stream(self):
        """Single device thread: streams always, writes to disk when save_gate is set."""
        try:
            self._init_emg(save=False)
            self._init_fsr(save=False)
            self.root.after(0, self._start_plot_loop)
            while not self.stream_stop.is_set():
                self._pump()
                time.sleep(0.05)
        except Exception as e:
            self._log(f"Stream ERROR: {e}\n{traceback.format_exc()}")
        finally:
            self.root.after(0, self._stop_plot_loop)
            # If recording was active when stream was killed, finalise
            if self.save_gate.is_set():
                self.save_gate.clear()
                self._teardown_emg(save=True)
                self._teardown_fsr(save=True)
                self._flush_save_files()
            else:
                self._teardown_emg(save=False)
                self._teardown_fsr(save=False)
            self.stream_thread = None
            self._log("Stream ended.")
            self.root.after(0, self._on_stream_ended)

    def _on_stream_ended(self):
        self.preview_btn.config(state="normal")
        self.start_btn.config(state="normal")
        self.end_preview_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        self._set_status("IDLE")
        self.instruction.config(
            text="Press  ▷ Preview  or  ⏺ Start Recording  to begin.", fg=FG)
        self.force_label.config(text="")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Recording
    # ═══════════════════════════════════════════════════════════════════════════
    def start_collection(self):
        # If not streaming yet, start the stream first (non-blocking)
        if not self._streaming_active():
            self._log("Stream not active — starting stream first…")
            self.start_preview()
            # Give devices 2 s to initialise before opening save files
            self.root.after(2000, self._begin_saving)
            return
        # Already streaming → begin saving immediately
        self._begin_saving()

    def _begin_saving(self):
        """Open save files and flip the save gate. Called on main thread."""
        if self.save_gate.is_set():
            return   # already saving
        if self.worker_thread and self.worker_thread.is_alive():
            return

        self._rebuild_trials_from_ui()

        self.save_dir  = self.save_dir_var.get().strip() or "./Data"
        base           = self.gesture_var.get().replace(" ", "_") or "session"
        self.base_name = base
        os.makedirs(self.save_dir, exist_ok=True)

        self.session_ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = os.path.join(self.save_dir,
                                        f"{base}_{self.session_ts}")
        os.makedirs(self.session_dir, exist_ok=True)

        # Tell retargeting to start saving into this folder
        if self.retarg_save_event is not None:
            self.retarg_save_event.set()
            self._log("Retargeting: saving started.")

        meta = dict(
            base_name=self.base_name,
            session_timestamp=self.session_ts,
            guided=bool(self.guided_var.get()),
            selected_gesture=self.gesture_var.get(),
            force_levels=list(self.force_levels),
            reps_per_force=self.reps_per_force,
            duration_per_rep_s=self.force_duration_s,
            rest_s=self.rest_s,
            trial_plan=self.trials,
            emg_source=self.emg_source_var.get(),
            fsr_enabled=bool(self.fsr_enabled_var.get()),
            fsr_saved=bool(self.fsr_save_var.get()),
        )
        if self.emg_source_var.get() == "noraxon":
            meta["noraxon"] = dict(
                ip=self.nx_ip_var.get().strip(),
                port=int(self.nx_port_var.get() or self.nx_port_default),
                sensors=self.nx_sensors_var.get().strip(),
                imu_only=bool(self.nx_imu_only_var.get()),
                strict=bool(self.nx_strict_var.get()))
        try:
            with open(os.path.join(self.session_dir, "session_meta.json"), "w") as fp:
                json.dump(meta, fp, indent=2)
        except Exception as e:
            self._log(f"session_meta.json error: {e}")

        # events.csv — always created for both guided and unguided
        events_path        = os.path.join(self.session_dir, "events.csv")
        self.events_fp     = open(events_path, "w", newline="")
        self.events_writer = csv.writer(self.events_fp)
        if self.guided_var.get():
            self.events_writer.writerow([
                "trial_index", "grasp", "force",
                "t_start_host_s", "t_end_host_s", "duration_s"])
        else:
            self.events_writer.writerow(
                ["event", "t_host_s", "datetime_utc"])
        self.events_file = events_path
        self._log(f"Events → {self.events_file}")

        self.worker_stop.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.preview_btn.config(state="disabled")
        self._set_status("RECORDING ●", DANGER)
        self._log(f"Session → {self.session_dir}")

        # Save gate is set inside _run_collectors once devices confirm ready
        self.worker_thread = threading.Thread(
            target=self._run_collectors, daemon=True)
        self.worker_thread.start()

        if self.guided_var.get():
            self.trial_idx     = -1
            self.trial_running = False
            self.root.after(400, self._advance_trial)
        else:
            self.instruction.config(text="Recording in progress…", fg=SUCCESS)
            self.force_label.config(text="Press  ⏹ Stop  when done.")

    def stop_collection(self):
        """Signal the save worker to stop; it handles teardown and file save."""
        self._log("Stop recording requested.")
        self.stop_btn.config(state="disabled")
        self.worker_stop.set()          # signals _run_collectors to exit
        self._finalise_recording()      # stops retargeting save gate

    def _finalise_recording(self):
        """Stop retargeting save gate. EMG/FSR handled by _run_collectors."""
        if self.retarg_save_event is not None:
            self.retarg_save_event.clear()
            self._log("Retargeting: saving stopped.")

    def _flush_save_files(self):
        """Close events file and any open save handles (EMG/FSR handle their own files)."""
        try:
            if self.events_fp:
                self.events_fp.flush()
                self.events_fp.close()
                self.events_fp = None
                self._log(f"Events saved → {self.events_file}")
        except Exception as e:
            self._log(f"Events flush error: {e}")

    def _run_collectors(self):
        """
        Hot-swap recorders from stream-only → save mode.

        Strategy:
          1. Stop the current stream-only recorder (discards its unsaved buffer).
          2. Restart it with save=True into session_dir.
          3. Set save_gate so the pump keeps feeding the plot buffer.
          4. On stop: stop the saving recorder, restart it in stream-only mode
             so the plot keeps working after the session ends.
        """
        guided      = self.guided_var.get()
        do_save_fsr = (bool(self.fsr_save_var.get()) and
                       bool(self.fsr_enabled_var.get()))
        try:
            # ── Hot-swap EMG: stop stream, reopen with save=True ──────────
            self._teardown_emg(save=False)
            self._init_emg(save=True)

            # ── Hot-swap FSR ──────────────────────────────────────────────
            if do_save_fsr:
                self._teardown_fsr(save=False)
                self._init_fsr(save=True)
            # (if FSR save disabled, keep existing stream-only FSR running)

            # ── Activate save gate ────────────────────────────────────────
            self.save_gate.set()
            self._log("Recorders hot-swapped → saving mode.")

            self.session_start_t = time.time()
            if not guided and self.events_writer:
                self.events_writer.writerow([
                    "session_start",
                    f"{self.session_start_t:.6f}",
                    datetime.utcnow().isoformat()])
                self.events_fp.flush()

            # ── Wait for stop signal ──────────────────────────────────────
            while not self.worker_stop.is_set():
                time.sleep(0.1)

        except Exception as e:
            self._log(f"ERROR in save worker: {e}\n{traceback.format_exc()}")
        finally:
            session_end_t = time.time()

            if not guided and self.events_writer:
                try:
                    self.events_writer.writerow([
                        "session_stop",
                        f"{session_end_t:.6f}",
                        datetime.utcnow().isoformat()])
                except Exception:
                    pass

            # ── Stop saving recorders, write files ────────────────────────
            self.save_gate.clear()
            self._teardown_emg(save=True)      # flushes EMG CSV
            if do_save_fsr:
                self._teardown_fsr(save=True)  # flushes FSR CSV

            # ── Restart in stream-only mode so plot keeps working ─────────
            try:
                self._init_emg(save=False)
            except Exception as e:
                self._log(f"EMG stream restart error: {e}")
            try:
                self._init_fsr(save=False)
            except Exception as e:
                self._log(f"FSR stream restart error: {e}")

            self._flush_save_files()   # events CSV
            self._log("Session saved. Streaming resumed.")
            self.root.after(0, lambda: (
                self.start_btn.config(state="normal"),
                self.stop_btn.config(state="disabled"),
                self._set_status("STREAMING", ACCENT2),
                self.instruction.config(
                    text="Saved.  Press  ⏺ Start Recording  for another run.",
                    fg=SUCCESS),
                self.force_label.config(text=""),
            ))

    # ═══════════════════════════════════════════════════════════════════════════
    #  Guided trial logic
    # ═══════════════════════════════════════════════════════════════════════════
    def _advance_trial(self):
        if not self.guided_var.get() or self.worker_stop.is_set():
            return

        if self.trial_running:
            self.trial_running = False
            t_end = time.time()
            tr    = self.trials[self.trial_idx]
            dur   = max(0.0, t_end - (self.trial_start_host_s or t_end))
            if self.events_writer:
                self.events_writer.writerow([
                    self.trial_idx + 1, tr["grasp"], tr["force"],
                    f"{self.trial_start_host_s:.6f}",
                    f"{t_end:.6f}", f"{dur:.3f}"])
                if self.events_fp:
                    self.events_fp.flush()

        self.trial_idx += 1
        if self.trial_idx >= len(self.trials):
            self._log("All trials complete — stopping.")
            self.stop_collection()
            return

        tr                      = self.trials[self.trial_idx]
        self.trial_running      = True
        self.trial_start_host_s = time.time()

        is_rest = (tr["grasp"] == "Open Hand")
        total   = len(self.trials)

        self.trial_badge.config(text=f"Trial {self.trial_idx+1} / {total}")

        if is_rest:
            self.instruction.config(text="REST  —  Open Hand", fg=FG2)
            self.force_label.config(text="Relax completely", fg=FG2)
        else:
            self.instruction.config(text=tr["grasp"], fg=FG)
            self.force_label.config(
                text=f"Force:  {tr['force']}",
                fg={"Light": SUCCESS, "Medium": WARNING,
                    "Strong": DANGER}.get(tr["force"], FG))

        self._update_progress()

    def _update_progress(self):
        if (not self.guided_var.get() or
                not self.trial_running or
                self.worker_stop.is_set()):
            return
        tr      = self.trials[self.trial_idx]
        elapsed = time.time() - self.trial_start_host_s
        remain  = max(0.0, tr["duration"] - elapsed)
        pct     = min(100.0, 100.0 * elapsed / max(0.001, tr["duration"]))

        self.progress["value"] = pct
        self.timer_label.config(text=f"{elapsed:04.1f} / {tr['duration']:.0f}s")

        if remain <= 0.0:
            self.root.after(200, self._advance_trial)
        else:
            self.root.after(self.refresh_ms, self._update_progress)

    # ── retargeting process control ──────────────────────────────────────────
    def _start_retargeting(self):
        """Spawn producer + consumer retargeting processes into session_dir."""
        if not self.retarg_enabled_var.get():
            return
        try:
            from dex_retargeting.constants import (
                RobotName, RetargetingType, HandType, get_default_config_path)
            from dex_retargeting.retargeting_config import RetargetingConfig
            from retargeting_teleop_2 import produce_frame, start_retargeting
            from pathlib import Path
        except ImportError as e:
            self._log(f"Retargeting import failed: {e} — skipping.")
            return

        # Map string -> enum
        try:
            robot_name   = RobotName[self.retarg_robot_var.get().strip()]
            retarg_type  = RetargetingType[self.retarg_type_var.get().strip()]
            hand_type    = HandType[self.retarg_hand_var.get().strip()]
        except KeyError as e:
            self._log(f"Retargeting config error: {e}")
            return

        cam = self.retarg_cam_var.get().strip()
        # Convert numeric camera index string → int
        try:    cam = int(cam)
        except ValueError: pass  # keep as string ("realsense" or path)

        config_path = get_default_config_path(robot_name, retarg_type, hand_type)
        robot_dir   = str(
            Path(__file__).absolute().parent.parent.parent
            / "assets" / "robots" / "hands")

        self.retarg_queue      = multiprocessing.Queue(maxsize=1000)
        self.retarg_save_event = multiprocessing.Event()  # clear = stream only

        # Default output dir: a 'retargeting' subfolder inside save_dir
        # (session_dir doesn't exist yet at preview time; saving into it
        #  is triggered later via retarg_save_event + dir passed separately)
        retarg_base_dir = self.save_dir_var.get().strip() or "./Data"

        def _consumer_entry(q, rd, cp, base_dir, save_event,
                            cam_shm, robot_shm):
            import os
            os.environ["NEUROCAPTURE_BASE_DIR"] = base_dir
            start_retargeting(q, rd, cp, save_event=save_event,
                              cam_shm=cam_shm, robot_shm=robot_shm)

        self.retarg_producer = multiprocessing.Process(
            target=produce_frame,
            args=(self.retarg_queue, cam),
            daemon=True)
        self.retarg_consumer = multiprocessing.Process(
            target=_consumer_entry,
            args=(self.retarg_queue, robot_dir, str(config_path),
                  retarg_base_dir, self.retarg_save_event,
                  self.retarg_cam_shm, self.retarg_robot_shm),
            daemon=True)

        self.retarg_producer.start()
        self.retarg_consumer.start()
        self._log(f"Retargeting streaming (robot={self.retarg_robot_var.get()}, "
                  f"hand={self.retarg_hand_var.get()}, cam={cam}) — "
                  f"not saving yet.")
        # Show embedded video panels
        self.root.after(0, self._start_video_feed)

    def _stop_retargeting(self):
        """Gracefully terminate retargeting processes."""
        for proc in (self.retarg_consumer, self.retarg_producer):
            if proc and proc.is_alive():
                proc.terminate()
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.kill()
        self.retarg_producer = None
        self.retarg_consumer = None
        if self.retarg_queue is not None:
            try: self.retarg_queue.close()
            except Exception: pass
            self.retarg_queue = None
        self._log("Retargeting stopped.")
        self.root.after(0, self._stop_video_feed)

    # ── close ─────────────────────────────────────────────────────────────────
    def _on_close(self):
        if self.preview_thread and self.preview_thread.is_alive():
            self.stop_preview()
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_collection()
            self.root.after(700, self.root.destroy)
        else:
            self._stop_retargeting()
            self.root.destroy()


# ═════════════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    DataCollectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
