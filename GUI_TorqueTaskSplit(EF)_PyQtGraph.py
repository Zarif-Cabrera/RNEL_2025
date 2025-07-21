import sys
import time
import nidaqmx
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from sklearn.linear_model import LinearRegression
import pandas as pd

def sin_func(t, A, f, phi, offset):
    return A * np.sin(2 * np.pi * f * t + phi) + offset

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time DAQ Plotting")
        self.layout = QtWidgets.QVBoxLayout(self)

        # --- Flexion duration input ---
        flexion_layout = QtWidgets.QHBoxLayout()
        flexion_label = QtWidgets.QLabel("Flexion Duration (s):")
        self.flexion_duration_box = QtWidgets.QLineEdit(self)
        self.flexion_duration_box.setText("5")
        flexion_layout.addWidget(flexion_label)
        flexion_layout.addWidget(self.flexion_duration_box)
        self.layout.addLayout(flexion_layout)

        # --- Number of cycles input ---
        cycles_layout = QtWidgets.QHBoxLayout()
        cycles_label = QtWidgets.QLabel("Number of Cycles:")
        self.cycles_box = QtWidgets.QLineEdit(self)
        self.cycles_box.setText("3")
        cycles_layout.addWidget(cycles_label)
        cycles_layout.addWidget(self.cycles_box)
        self.layout.addLayout(cycles_layout)

        # DAQ device input
        daq_layout = QtWidgets.QHBoxLayout()
        daq_label = QtWidgets.QLabel("Daq Device:")
        self.daq_input = QtWidgets.QLineEdit(self)
        self.daq_input.setPlaceholderText("Enter DAQ device (e.g., Dev1)")
        daq_layout.addWidget(daq_label)
        daq_layout.addWidget(self.daq_input)
        self.layout.addLayout(daq_layout)

        # Buttons
        self.btn_layout = QtWidgets.QHBoxLayout()
        self.baseline_btn = QtWidgets.QPushButton("Set Flexion Baseline", self)
        self.baseline_btn.clicked.connect(self.set_flexion_baseline)
        self.btn_layout.addWidget(self.baseline_btn)

        self.ext_baseline_btn = QtWidgets.QPushButton("Set Extension Baseline", self)
        self.ext_baseline_btn.clicked.connect(self.set_extension_baseline)
        self.btn_layout.addWidget(self.ext_baseline_btn)

        self.mvc_btn = QtWidgets.QPushButton("Set Flexion MVC", self)
        self.mvc_btn.clicked.connect(self.set_flexion_mvc)
        self.btn_layout.addWidget(self.mvc_btn)

        self.ext_mvc_btn = QtWidgets.QPushButton("Set Extension MVC", self)
        self.ext_mvc_btn.clicked.connect(self.set_extension_mvc)
        self.btn_layout.addWidget(self.ext_mvc_btn)

        self.task_btn = QtWidgets.QPushButton("Start Task", self)
        self.task_btn.clicked.connect(self.start_task)
        self.btn_layout.addWidget(self.task_btn)

        self.layout.addLayout(self.btn_layout)

        # Plot widget
        self.plot_widget = pg.PlotWidget(title="Real-Time Torque Plot")
        self.layout.addWidget(self.plot_widget)
        self.curve = self.plot_widget.plot(pen='b')
        self.dot = pg.ScatterPlotItem(size=12, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        self.target_curve = self.plot_widget.plot(pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
        self.flexion_curve = self.plot_widget.plot(pen=pg.mkPen('g', width=2, style=QtCore.Qt.DotLine))

        # Data storage as pandas DataFrames
        self.df_task = pd.DataFrame(columns=["time", "position", "torque", "voltage"])
        self.df_baseline = pd.DataFrame(columns=["time", "torque"])
        self.df_flexionMVC = pd.DataFrame(columns=["time", "torque"])
        self.df_ext_baseline = pd.DataFrame(columns=["time", "torque"])
        self.df_extensionMVC = pd.DataFrame(columns=["time", "torque"])

        self.window_size = 50
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.start_time = None
        self.acquiring = False
        self.trialNum = 1

    def startDAQ(self):
        try:
            if hasattr(self, 'input_task') and self.input_task is not None:
                self.input_task.stop()
                self.input_task.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'output_task') and self.output_task is not None:
                self.output_task.stop()
                self.output_task.close()
        except Exception:
            pass

        dev = self.daq_input.text()
        self.input_task = nidaqmx.Task()
        self.input_task.ai_channels.add_ai_voltage_chan(
            f"{dev}/ai3", terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
        self.input_task.ai_channels.add_ai_voltage_chan(
            f"{dev}/ai2", terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
        self.input_task.ai_channels.add_ai_voltage_chan(
            f"{dev}/ai4", terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
        self.output_task = nidaqmx.Task()
        self.output_task.ao_channels.add_ao_voltage_chan(
            f"{dev}/ao0", min_val=-10.0, max_val=10.0)
        self.input_task.start()
        self.output_task.start()

    def stopDAQ(self):
        try:
            self.input_task.stop()
            self.input_task.close()
        except Exception:
            pass
        try:
            self.output_task.stop()
            self.output_task.close()
        except Exception:
            pass

    def set_flexion_baseline(self):
        self.startDAQ()
        baseline_data = []
        Ts = 0.01
        try:
            duration = float(self.flexion_duration_box.text())
        except Exception:
            duration = 5
        self.start_time = time.time()
        while (time.time() - self.start_time) < duration:
            value = self.input_task.read(number_of_samples_per_channel=10)
            t = time.time() - self.start_time
            torque = np.mean(value[1])
            baseline_data.append([t, torque])
            QtWidgets.QApplication.processEvents()
            time.sleep(Ts)
        self.stopDAQ()
        self.df_baseline = pd.DataFrame(baseline_data, columns=["time", "torque"])
        print("Flexion Baseline set.")

    def set_extension_baseline(self):
        self.startDAQ()
        baseline_data = []
        Ts = 0.01
        try:
            duration = float(self.flexion_duration_box.text())
        except Exception:
            duration = 5
        self.start_time = time.time()
        while (time.time() - self.start_time) < duration:
            value = self.input_task.read(number_of_samples_per_channel=10)
            t = time.time() - self.start_time
            torque = np.mean(value[1])  # or whichever channel is extension torque
            baseline_data.append([t, torque])
            QtWidgets.QApplication.processEvents()
            time.sleep(Ts)
        self.stopDAQ()
        self.df_ext_baseline = pd.DataFrame(baseline_data, columns=["time", "torque"])
        # Fit sinusoid to extension baseline if needed (optional)
        print("Extension Baseline set.")

    def set_flexion_mvc(self):
        self.startDAQ()
        mvc_data = []
        Ts = 0.01
        try:
            duration = float(self.flexion_duration_box.text())
        except Exception:
            duration = 5
        self.start_time = time.time()
        while (time.time() - self.start_time) < duration:
            value = self.input_task.read(number_of_samples_per_channel=10)
            t = time.time() - self.start_time
            torque = np.mean(value[1])
            mvc_data.append([t, torque])
            QtWidgets.QApplication.processEvents()
            time.sleep(Ts)
        self.stopDAQ()
        self.df_flexionMVC = pd.DataFrame(mvc_data, columns=["time", "torque"])
        print("Flexion MVC set.")

    def set_extension_mvc(self):
        self.startDAQ()
        mvc_data = []
        Ts = 0.01
        try:
            duration = float(self.flexion_duration_box.text())
        except Exception:
            duration = 5
        self.start_time = time.time()
        while (time.time() - self.start_time) < duration:
            value = self.input_task.read(number_of_samples_per_channel=10)
            t = time.time() - self.start_time
            torque = np.mean(value[1])  # or whichever channel is extension torque
            mvc_data.append([t, torque])
            QtWidgets.QApplication.processEvents()
            time.sleep(Ts)
        self.stopDAQ()
        self.df_extensionMVC = pd.DataFrame(mvc_data, columns=["time", "torque"])
        print("Extension MVC set.")

    def start_task(self):
        self.acquiring = True
        self.df_task = pd.DataFrame(columns=["time", "position", "torque", "voltage"])
        self.start_time = time.time()
        self.startDAQ()
        try:
            self.flexion_duration = float(self.flexion_duration_box.text())
        except Exception:
            self.flexion_duration = 5
        try:
            self.num_cycles = int(self.cycles_box.text())
        except Exception:
            self.num_cycles = 3
        self.task_total_time = self.flexion_duration * 2 * self.num_cycles
        print(f"Starting task for {self.task_total_time} seconds.")
        self.timer.start(10)  # update every 10 ms

    def update_plot(self):
        if not self.acquiring:
            return
        value = self.input_task.read(number_of_samples_per_channel=10)
        t = time.time() - self.start_time
        position = np.mean(value[0])
        torque = np.mean(value[1])
        voltage = np.mean(value[2])
        ao0 = 9  # Example: output value (change as needed)
        self.output_task.write(ao0)
        # --- Append to DataFrame ---
        new_row = pd.DataFrame([[t, position, torque, voltage]], columns=self.df_task.columns)
        if self.df_task.empty:
            self.df_task = new_row
        else:
            self.df_task = pd.concat([self.df_task, new_row], ignore_index=True)

        # Moving window for plotting
        if len(self.df_task) < self.window_size:
            x_window = self.df_task["time"].values
            y_window = self.df_task["torque"].values
        else:
            x_window = self.df_task["time"].values[-self.window_size:]
            y_window = self.df_task["torque"].values[-self.window_size:]

        self.curve.setData(x_window, y_window)
        self.dot.setData([x_window[-1]], [y_window[-1]])
        self.plot_widget.setXRange(x_window[0], x_window[-1])

        # Example: plot target sinusoid if baseline is set
        if hasattr(self, "baseline_reg") and self.baseline_reg is not None and hasattr(self, "baseline_freq") and self.baseline_freq is not None:
            X = np.column_stack((
                np.sin(2 * np.pi * self.baseline_freq * np.array(x_window)),
                np.cos(2 * np.pi * self.baseline_freq * np.array(x_window))
            ))
            y_target = self.baseline_reg.predict(X)
            self.target_curve.setData(x_window, y_target)

        # Plot flexion MVC if available
        if hasattr(self, "df_flexionMVC") and not self.df_flexionMVC.empty:
            self.flexion_curve.setData(self.df_flexionMVC["time"].values, self.df_flexionMVC["torque"].values)

        # Stop after task_total_time seconds
        if t > self.task_total_time:
            self.timer.stop()
            self.stopDAQ()
            self.acquiring = False
            # --- Save all data to CSV ---
            self.df_task.to_csv(f"DAQ_Task_Trial{self.trialNum}.csv", index=False)
            print("Task complete. Data saved.")
            self.trialNum += 1

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


        # # Fit sinusoid to baseline
        # t_vals = self.df_baseline["time"].values
        # y_vals = self.df_baseline["torque"].values
        # dt = np.mean(np.diff(t_vals))
        # N = len(t_vals)
        # yf = rfft(y_vals)
        # xf = rfftfreq(N, dt)
        # dominant_freq = xf[np.argmax(np.abs(yf[1:])) + 1]
        # X = np.column_stack((
        #     np.sin(2 * np.pi * dominant_freq * t_vals),
        #     np.cos(2 * np.pi * dominant_freq * t_vals)
        # ))
        # reg = LinearRegression()
        # reg.fit(X, y_vals)
        # self.baseline_reg = reg
        # self.baseline_freq = dominant_freq
