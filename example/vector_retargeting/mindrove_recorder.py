# # import numpy as np
# # import pandas as pd
# # import time
# # from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
# # from mindrove.data_filter import DataFilter
# # import datetime

# # class MindroveRecorder:
# #     def __init__(self, board_id=BoardIds.MINDROVE_WIFI_BOARD, duration=None):
# #         self.board_id = board_id
# #         self.duration = duration
# #         self.board = None
# #         self.emg_channels = None
# #         self.accel_channels = None
# #         self.gyro_channels = None
# #         self.timestamp_channel = None

# #     def start(self):
# #         BoardShim.enable_dev_board_logger()
# #         params = MindRoveInputParams()  # Add connection details here if needed
# #         self.board = BoardShim(self.board_id, params)

# #         self.emg_channels = BoardShim.get_exg_channels(self.board_id)
# #         self.accel_channels = BoardShim.get_accel_channels(self.board_id)
# #         self.gyro_channels = BoardShim.get_gyro_channels(self.board_id)
# #         self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)

# #         self.board.prepare_session()
# #         self.board.start_stream()
# #         self.start_time = time.time()

# #         print("✅ Mindrove stream started.")

# #     def stop(self, save_csv_path=None):
# #         if self.board is None:
# #             raise RuntimeError("Stream not started. Call `start()` first.")

# #         self.board.stop_stream()
# #         data = self.board.get_board_data()
# #         self.board.release_session()

# #         print("🛑 Stream stopped and session released.")

# #         # Convert to DataFrame
# #         df = pd.DataFrame(np.transpose(data))
# #         column_rename_mapping = {
# #             0: 'CH1', 1: 'CH2', 2: 'CH3', 3: 'CH4',
# #             4: 'CH5', 5: 'CH6', 6: 'CH7', 7: 'CH8',
# #             20: 'AccX', 21: 'AccY', 22: 'AccZ',
# #             23: 'GyX', 24: 'GyY', 25: 'GyZ',
# #             27: 'Timestamp'
# #         }
# #         df = df.rename(columns=column_rename_mapping)
# #         columns_to_write = [ 'CH1','CH2','CH3', 'CH4','CH5','CH6','CH7','CH8','AccX','AccY','AccZ','GyX','GyY','GyZ','Timestamp']


# #         if save_csv_path:
# #             now = datetime.datetime.now()
# #             timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
# #             filename = f"{save_csv_path}_{timestamp}.csv"
# #             df[columns_to_write].to_csv(filename, index=False)
# #             # print(f"💾 Data saved to {filename}")

# #         return filename


# import numpy as np
# import pandas as pd
# import time
# from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
# from mindrove.data_filter import DataFilter
# import datetime
# import matplotlib.pyplot as plt
# from collections import deque
# import threading
# # MindRove filters
# from mindrove.data_filter import DataFilter, DetrendOperations, FilterTypes

# class MindroveRecorder:
#     def __init__(self, board_id=BoardIds.MINDROVE_WIFI_BOARD, duration=None):
#         self.board_id = board_id
#         self.duration = duration
#         self.board = None
#         self.emg_channels = None
#         self.accel_channels = None
#         self.gyro_channels = None
#         self.timestamp_channel = None

#         self.latest_values = [0.0] * 8  # Assuming 8 EMG channels
#         self.plotting = False
#         self._plot_thread = None
#         self._reading_thread = None
#         self._keep_reading = False

#     def start(self):
#         BoardShim.enable_dev_board_logger()
#         params = MindRoveInputParams()  # Customize if needed
#         self.board = BoardShim(self.board_id, params)

#         self.emg_channels = BoardShim.get_exg_channels(self.board_id)
#         self.accel_channels = BoardShim.get_accel_channels(self.board_id)
#         self.gyro_channels = BoardShim.get_gyro_channels(self.board_id)
#         self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)

#         self.board.prepare_session()
#         self.board.start_stream()
#         self.start_time = time.time()

#         self._keep_reading = True
#         self._reading_thread = threading.Thread(target=self._update_latest_data)
#         self._reading_thread.daemon = True
#         self._reading_thread.start()

#         print("✅ Mindrove stream started.")

#     def _update_latest_data(self):
#         while self._keep_reading:
#             try:
#                 data = self.board.get_current_board_data(1)
#                 if data.shape[1] > 0:
#                     emg_data = data[self.emg_channels, -1]
#                     self.latest_values = emg_data.tolist()
#             except Exception as e:
#                 print("⚠️ EMG read error:", e)
#             time.sleep(0.01)  # Adjust to match sampling rate


#     def stop(self, save_csv_path=None):
#         if self.board is None:
#             raise RuntimeError("Stream not started. Call `start()` first.")

#         self._keep_reading = False
#         if self._reading_thread:
#             self._reading_thread.join()

#         self.board.stop_stream()
#         data = self.board.get_board_data()
#         self.board.release_session()

#         print("🛑 Stream stopped and session released.")

#         # Convert to DataFrame
#         df = pd.DataFrame(np.transpose(data))
#         column_rename_mapping = {
#             0: 'CH1', 1: 'CH2', 2: 'CH3', 3: 'CH4',
#             4: 'CH5', 5: 'CH6', 6: 'CH7', 7: 'CH8',
#             20: 'AccX', 21: 'AccY', 22: 'AccZ',
#             23: 'GyX', 24: 'GyY', 25: 'GyZ',
#             27: 'Timestamp'
#         }
#         df = df.rename(columns=column_rename_mapping)
#         columns_to_write = ['Timestamp','CH1','CH2','CH3','CH4','CH5','CH6','CH7','CH8','AccX','AccY','AccZ','GyX','GyY','GyZ']
#         sampling_rate=500
#         for channel in range(8):
#             print ("channel",channel)
    
#             DataFilter.detrend(df[channel], DetrendOperations.CONSTANT.value)
#             DataFilter.perform_bandpass(df[channel], sampling_rate, 51.0, 100.0, 2,
#                                         FilterTypes.BUTTERWORTH.value, 0)
#             DataFilter.perform_bandstop(df[channel], sampling_rate, 48.0, 52.0, 2, #remove popwerline interference
#                                         FilterTypes.BUTTERWORTH.value, 0)
#             DataFilter.perform_bandstop(df[channel], sampling_rate, 58.0, 62.0, 2, #remove popwerline interference
#                                         FilterTypes.BUTTERWORTH.value, 0)
#             # data[channel] = scaler.fit_transform(data[channel].reshape(-1, 1)).flatten()
    
#         df = pd.DataFrame(data.T)
#         # return df

#         if save_csv_path:
#             now = datetime.datetime.now()
#             timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
#             filename = f"{save_csv_path}_{timestamp}.csv"
#             df[columns_to_write].to_csv(filename, index=False)
#             return filename

#     def start_plotting(self):
#         self.plotting = True
#         self._plot_thread = threading.Thread(target=self._plot_loop)
#         self._plot_thread.daemon = True
#         self._plot_thread.start()

#     def _plot_loop(self):
#         max_len = 200
#         emg_labels = ["CH{}".format(i+1) for i in range(8)]
#         data = [deque([0]*max_len, maxlen=max_len) for _ in range(8)]

#         plt.ion()
#         fig, ax = plt.subplots()
#         lines = [ax.plot([], [])[0] for _ in range(8)]
#         ax.set_xlim(0, max_len)
#         ax.set_ylim(-500000, 500000)  
#         ax.set_title("Real-Time EMG")
#         ax.legend(emg_labels)

#         while self.plotting and self._keep_reading:
#             for i in range(8):
#                 data[i].append(self.latest_values[i])
#                 lines[i].set_xdata(range(len(data[i])))
#                 lines[i].set_ydata(data[i])
#             ax.relim()
#             ax.autoscale_view()
#             fig.canvas.draw()
#             fig.canvas.flush_events()
#             time.sleep(0.05)


# mindrove_recorder.py
import numpy as np
import pandas as pd
import time
import datetime
import threading
from collections import deque

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import DataFilter, DetrendOperations, FilterTypes

class MindroveRecorder:
    def __init__(self, board_id=BoardIds.MINDROVE_WIFI_BOARD, duration=None):
        self.board_id = board_id
        self.duration = duration
        self.board = None

        self.emg_channels = None
        self.accel_channels = None
        self.gyro_channels = None
        self.timestamp_channel = None

        # live view (optional)
        self.latest_values = [0.0] * 8
        self.plotting = False
        self._plot_thread = None

        self._reading_thread = None
        self._keep_reading = False

    # ------------ streaming ------------
    def start(self):
        BoardShim.enable_dev_board_logger()
        params = MindRoveInputParams()
        self.board = BoardShim(self.board_id, params)

        self.emg_channels = BoardShim.get_exg_channels(self.board_id)   # indices in SDK buffer
        self.accel_channels = BoardShim.get_accel_channels(self.board_id)
        self.gyro_channels = BoardShim.get_gyro_channels(self.board_id)
        self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)

        self.board.prepare_session()
        self.board.start_stream()
        self.start_time = time.time()

        # background sampler for live preview
        self._keep_reading = True
        self._reading_thread = threading.Thread(target=self._update_latest_data, daemon=True)
        self._reading_thread.start()

        print("✅ Mindrove stream started.")

    def _update_latest_data(self):
        while self._keep_reading:
            try:
                data = self.board.get_current_board_data(1)
                if data.shape[1] > 0:
                    emg_data = data[self.emg_channels, -1]
                    self.latest_values = emg_data.tolist()
            except Exception as e:
                print("⚠️ EMG read error:", e)
            time.sleep(0.01)

    # ------------ preprocessing ------------
    @staticmethod
    def _preprocess_emg_block(df: pd.DataFrame, emg_cols, fs: int) -> pd.DataFrame:
        """
        Returns a *new* dataframe with the same columns as df, but EMG columns filtered in-place.
        - Detrend (DC removal)
        - Band-pass 51–100 Hz (Butterworth, order 2)
        - Notch 48–52 Hz and 58–62 Hz (order 2)
        """
        if not emg_cols:
            return df.copy()

        if fs < 250:
            # Too low to safely band-pass up to 100 Hz → return unchanged
            print(f"[EMG] Sampling rate {fs} Hz too low for 51–100 Hz band-pass. Skipping EMG filtering.")
            return df.copy()

        out = df.copy()
        for ch in emg_cols:
            sig = out[ch].to_numpy(dtype=np.float64, copy=True)

            # Fill NaNs by linear interpolation (both ends)
            if np.isnan(sig).any():
                s = pd.Series(sig)
                sig = s.interpolate(limit_direction="both").to_numpy(dtype=np.float64)

            # Detrend
            DataFilter.detrend(sig, DetrendOperations.CONSTANT.value)

            # Band-pass
            DataFilter.perform_bandpass(
                sig, int(fs), 51.0, 100.0, 2, FilterTypes.BUTTERWORTH.value, 0
            )

            # Notches
            DataFilter.perform_bandstop(sig, int(fs), 48.0, 52.0, 2, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(sig, int(fs), 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)

            out[ch] = sig
        return out

    # ------------ stop & save ------------
    def stop(self, save_csv_path=None):
        if self.board is None:
            raise RuntimeError("Stream not started. Call start() first.")

        self._keep_reading = False
        if self._reading_thread:
            self._reading_thread.join(timeout=1.0)

        self.board.stop_stream()
        data = self.board.get_board_data()    # shape: (channels, samples)
        self.board.release_session()

        print("🛑 Stream stopped and session released.")

        # Build DataFrame (transpose to samples x channels)
        df = pd.DataFrame(np.transpose(data))

        # Rename columns for convenience (MindRove WiFi layout)
        rename = {
            0: 'CH1', 1: 'CH2', 2: 'CH3', 3: 'CH4',
            4: 'CH5', 5: 'CH6', 6: 'CH7', 7: 'CH8',
            20: 'AccX', 21: 'AccY', 22: 'AccZ',
            23: 'GyX', 24: 'GyY', 25: 'GyZ',
            27: 'Timestamp'
        }
        df = df.rename(columns=rename)

        # Choose column order (present columns only)
        emg_cols = [c for c in [f"CH{i}" for i in range(1, 9)] if c in df.columns]
        imu_cols = [c for c in ['AccX', 'AccY', 'AccZ', 'GyX', 'GyY', 'GyZ'] if c in df.columns]
        cols_order = (['Timestamp'] + emg_cols + imu_cols)
        cols_order = [c for c in cols_order if c in df.columns]  # keep existing only
        df = df[cols_order]

        # Compute sampling rate from SDK (more reliable than estimating)
        fs = BoardShim.get_sampling_rate(self.board_id)
        print(f"[EMG] MindRove sampling rate: {fs} Hz")

        # Preprocess EMG → filtered dataframe with same column names
        df_pre = self._preprocess_emg_block(df, emg_cols, fs)

        # Save
        raw_file = pre_file = None
        if save_csv_path:
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            raw_file = f"{save_csv_path}_{now}.csv"
            pre_file = f"{save_csv_path}_pre_{now}.csv"

            df.to_csv(raw_file, index=False)
            df_pre.to_csv(pre_file, index=False)

            print(f"💾 Saved raw EMG/IMU → {raw_file}")
            print(f"💾 Saved preprocessed EMG/IMU → {pre_file}")

        return {"raw": raw_file, "preprocessed": pre_file}

    # ------------ optional live plotting (unchanged) ------------

    def start_plotting(self, ax=None):
        """If ax is provided, plot into that axis; otherwise make our own figure."""
        self._plot_ax = ax
        self.plotting = True
        self._plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
        self._plot_thread.start()

    def stop_plotting(self):
        self.plotting = False
        self._plot_ax = None  # release reference

    def _plot_loop(self):
        import matplotlib.pyplot as plt
        from collections import deque
        max_len = 200
        emg_labels = [f"CH{i}" for i in range(1, 9)]
        data = [deque([0]*max_len, maxlen=max_len) for _ in range(8)]

        # pick axis / figure
        if self._plot_ax is None:
            plt.ion()
            fig, ax = plt.subplots()
        else:
            ax = self._plot_ax
            fig = ax.figure

        lines = [ax.plot([], [])[0] for _ in range(8)]
        ax.set_xlim(0, max_len)
        ax.set_ylim(-500000, 500000)
        ax.set_title("Real-Time EMG")
        ax.legend(emg_labels, loc="upper right", ncol=4)

        while self.plotting and self._keep_reading:
            for i in range(8):
                data[i].append(self.latest_values[i])
                lines[i].set_xdata(range(len(data[i])))
                lines[i].set_ydata(data[i])
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.05)

    # --- preview aliases (no file writing) ---
    def start_preview(self):
        # same as start(), but the GUI will not pass a save path on stop
        return self.start()

    def stop_preview(self):
        # stop without saving
        return self.stop(save_csv_path=None)
