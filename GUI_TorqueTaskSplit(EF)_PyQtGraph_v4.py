import sys
import time
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
import nidaqmx
import nidaqmx.constants
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Fixed Y-axis bounds for all plots
Y_AXIS_BOUND = 0.3

#Note: click Humac Task first since it takes a few milliseconds to start

class MVCCurveFitter:
    """Embedded MVC curve fitting class for real-time prediction"""
    def __init__(self, velocity_threshold=0.03):
        self.velocity_threshold = velocity_threshold
        self.df_raw = None
        self.df_filtered = None
        self.best_model = None
        self.best_degree = None
        self.best_r2 = None
        self.best_rmse = None
        
    def load_data_from_dataframe(self, df, phase_type="flexion"):
        """Load MVC data from DataFrame"""
        self.df_raw = df.copy()
        self.filter_by_velocity(phase_type)
        return len(self.df_filtered) > 0
    
    def filter_by_velocity(self, phase_type="flexion"):
        """Filter DataFrame based on phase-specific velocity criteria"""
        if 'velocity' not in self.df_raw.columns:
            self.df_filtered = self.df_raw.copy()
            return
            
        initial_count = len(self.df_raw)
        
        # Phase-specific filtering logic
        if phase_type.lower() == 'flexion':
            self.df_filtered = self.df_raw[self.df_raw['velocity'] <= -self.velocity_threshold]
        elif phase_type.lower() == 'extension':
            self.df_filtered = self.df_raw[self.df_raw['velocity'] >= self.velocity_threshold]
        else:
            self.df_filtered = self.df_raw[
                (self.df_raw['velocity'] <= -self.velocity_threshold) | 
                (self.df_raw['velocity'] >= self.velocity_threshold)
            ]
        
        final_count = len(self.df_filtered)
        print(f"Velocity filtering ({phase_type}): {initial_count} -> {final_count} points")
        
        # Normalize time to start at 0
        if len(self.df_filtered) > 0:
            time_raw = self.df_filtered["time"].values
            self.df_filtered = self.df_filtered.copy()
            self.df_filtered["time_normalized"] = time_raw - time_raw[0]
    
    def fit_polynomial_models(self, max_degree=5):  # Reduced from 8 to 5 for stability
        """Fit polynomial models and find the best one"""
        if self.df_filtered is None or len(self.df_filtered) == 0:
            return False
            
        X = self.df_filtered["time_normalized"].values.reshape(-1, 1)
        y = self.df_filtered["torque"].values
        
        best_r2 = -np.inf
        
        # Use more conservative degree limits to prevent extrapolation disasters
        min_degree = 1  # Start from linear
        max_safe_degree = min(max_degree, len(self.df_filtered) // 5, 4)  # Very conservative
        
        for degree in range(min_degree, max_safe_degree + 1):
            try:
                poly_model = Pipeline([
                    ('poly', PolynomialFeatures(degree=degree)),
                    ('linear', LinearRegression())
                ])
                
                poly_model.fit(X, y)
                y_pred = poly_model.predict(X)
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                if r2 > best_r2:
                    best_r2 = r2
                    self.best_model = poly_model
                    self.best_degree = degree
                    self.best_r2 = r2
                    self.best_rmse = rmse
                    
            except Exception as e:
                continue
        
        success = self.best_model is not None
        if success:
            print(f"Best curve fit: Polynomial degree {self.best_degree}, R² = {self.best_r2:.4f}")
        return success
    
    def predict(self, time_normalized):
        """Predict torque values for given normalized time points"""
        if self.best_model is None:
            return np.zeros_like(time_normalized)
        
        time_reshaped = np.array(time_normalized).reshape(-1, 1)
        return self.best_model.predict(time_reshaped)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time DAQ Plotting v4 - Curve Fitting")
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
        
        # Set larger font sizes for axis labels and numbers
        font = QtGui.QFont()
        font.setPointSize(24)
        self.plot_widget.getAxis('left').setStyle(tickFont=font)
        self.plot_widget.getAxis('bottom').setStyle(tickFont=font)
        self.plot_widget.getAxis('left').setPen(pg.mkPen(width=2))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(width=2))
        
        # Configure time axis for cleaner tick spacing (0.5 second intervals)
        self.plot_widget.getAxis('bottom').setTickSpacing(major=1, minor=0.5)

        
        self.layout.addWidget(self.plot_widget)
        # Add MVC curve first (bottom layer)
        self.mvc_curve = self.plot_widget.plot(pen=pg.mkPen('g', width=5, style=QtCore.Qt.DotLine))
        # Add blue curve last (top layer) so it appears over MVC line
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=4))
        self.dot = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)

        # Data storage as pandas DataFrames
        self.df_task = pd.DataFrame(columns=["time", "position", "torque", "uncorrected_torque", "voltage", "velocity"])
        self.df_baseline = pd.DataFrame(columns=["time", "torque", "velocity"])
        self.df_flexionMVC = pd.DataFrame(columns=["time", "torque", "velocity"])
        self.df_ext_baseline = pd.DataFrame(columns=["time", "torque", "velocity"])
        self.df_extensionMVC = pd.DataFrame(columns=["time", "torque", "velocity"])

        self.window_size = 500
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.start_time = None
        self.acquiring = False
        self.trialNum = 1
        
        # Add variables for baseline/MVC data collection
        self.collection_mode = None
        self.collection_data = []
        self.collection_duration = 0
        
        # Look-ahead parameters for centered real-time display
        self.look_ahead_time = 1.0
        self.center_position = 0.5
        
        # Performance optimization variables
        self.plot_update_counter = 0
        self.plot_update_interval = 2
        self.data_buffer = []
        self.buffer_size = 10

        # Simple moving average for smooth plotting
        self.smoothing_window_size = 3  # Number of points for moving average
        self.torque_history = []  # Store recent torque values for smoothing
        self.smoothed_torque = 0.0  # Current smoothed torque value

        # Curve fitting models (replacing sinusoidal fit parameters)
        self.flexion_curve_fitter = None
        self.extension_curve_fitter = None
        self.mvc_fitted = False
        
        # Baseline curve fitting models
        self.flexion_baseline_fitter = None
        self.extension_baseline_fitter = None
        self.flexion_baseline_fitted = False
        self.extension_baseline_fitted = False
        
        # Sampling rate tracking for real-time display
        self.sampling_rates = []
        self.last_sample_time = None

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
            f"{dev}/ai0", terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
        self.input_task.ai_channels.add_ai_voltage_chan(
            f"{dev}/ai3", terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
        self.input_task.ai_channels.add_ai_voltage_chan(
            f"{dev}/ai2", terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
        self.input_task.ai_channels.add_ai_voltage_chan(
            f"{dev}/ai5", terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
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
        try:
            duration = float(self.flexion_duration_box.text())
        except Exception:
            duration = 5
        
        # Setup for collection
        self.collection_mode = 'flexion_baseline'
        self.collection_data = []
        self.collection_duration = duration
        self.acquiring = True
        self.start_time = None
        
        # Setup plot with default range for collection
        self.plot_widget.clear()
        self.plot_widget.setTitle("Setting Flexion Baseline")
        
        # Use fixed y-range for baseline collection
        self.plot_widget.setYRange(-Y_AXIS_BOUND, Y_AXIS_BOUND, padding=0)
        self.plot_widget.disableAutoRange(axis='y')
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=4), name='Baseline Torque')
        self.dot = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(5)

    def set_extension_baseline(self):
        try:
            duration = float(self.flexion_duration_box.text())
        except Exception:
            duration = 5
        
        # Setup for collection
        self.collection_mode = 'extension_baseline'
        self.collection_data = []
        self.collection_duration = duration
        self.acquiring = True
        self.start_time = None
        
        # Setup plot with default range for collection
        self.plot_widget.clear()
        self.plot_widget.setTitle("Setting Extension Baseline")
        
        # Use fixed y-range for baseline collection
        self.plot_widget.setYRange(-Y_AXIS_BOUND, Y_AXIS_BOUND, padding=0)
        self.plot_widget.disableAutoRange(axis='y')
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=4), name='Baseline Torque')
        self.dot = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(5)

    def set_flexion_mvc(self):
        try:
            duration = float(self.flexion_duration_box.text())
        except Exception:
            duration = 5
        
        # Setup for collection
        self.collection_mode = 'flexion_mvc'
        self.collection_data = []
        self.collection_duration = duration
        self.acquiring = True
        self.start_time = None
        
        # Setup plot with default range for MVC collection
        self.plot_widget.clear()
        self.plot_widget.setTitle("Setting Flexion MVC")
        
        # Use fixed y-range for MVC collection
        self.plot_widget.setYRange(-Y_AXIS_BOUND, Y_AXIS_BOUND, padding=0)
        self.plot_widget.disableAutoRange(axis='y')
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=4), name='MVC Torque')
        self.dot = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(5)

    def set_extension_mvc(self):
        try:
            duration = float(self.flexion_duration_box.text())
        except Exception:
            duration = 5
        
        # Setup for collection
        self.collection_mode = 'extension_mvc'
        self.collection_data = []
        self.collection_duration = duration
        self.acquiring = True
        self.start_time = None
        
        # Setup plot with default range for MVC collection
        self.plot_widget.clear()
        self.plot_widget.setTitle("Setting Extension MVC")
        
        # Use fixed y-range for MVC collection
        self.plot_widget.setYRange(-Y_AXIS_BOUND, Y_AXIS_BOUND, padding=0)
        self.plot_widget.disableAutoRange(axis='y')
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=4), name='MVC Torque')
        self.dot = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(5)

    def start_task(self):
        print(f"DEBUG: Starting task, current trial number: {self.trialNum}")
        self.acquiring = True
        self.df_task = pd.DataFrame(columns=["time", "position", "torque", "uncorrected_torque", "voltage", "velocity"])
        self.start_time = None
        self.collection_mode = None
        
        # Reset sampling rate tracking
        self.sampling_rates = []
        self.last_sample_time = None
        
        # Reset performance optimization variables for task
        self.plot_update_counter = 0
        self.data_buffer = []
        
        # Reset smoothing variables for new task
        self.torque_history = []
        self.smoothed_torque = 0.0
        
        # Setup plot for task with FIXED y-range
        self.plot_widget.clear()
        self.plot_widget.setTitle("Real-Time Torque Plot")
        
        # Calculate y-range using fixed bounds
        y_max = Y_AXIS_BOUND
        y_min = -Y_AXIS_BOUND
        
        print(f"Setting fixed y-range: [{y_min:.2f}, {y_max:.2f}]")
        
        # Set fixed Y range and disable auto-range for performance
        self.plot_widget.setYRange(y_min, y_max, padding=0)
        self.plot_widget.disableAutoRange(axis='y')
        
        # Additional performance optimizations
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.hideButtons()
        
        # Add MVC curve first (bottom layer)
        self.mvc_curve = self.plot_widget.plot(pen=pg.mkPen('r', width=7, style=QtCore.Qt.DotLine), name='Combined MVC')
        # Add blue curve last (top layer) so it appears over MVC line
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=4), name='Measured Torque')
        self.dot = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        
        self.startDAQ()
        try:
            self.flexion_duration = float(self.flexion_duration_box.text())
        except Exception:
            self.flexion_duration = 5
        try:
            self.num_cycles = float(self.cycles_box.text())
        except Exception:
            self.num_cycles = 3
        self.task_total_time = self.flexion_duration * 2 * self.num_cycles
        print(f"Starting task for {self.task_total_time} seconds.")
        self.timer.start(5)

    def update_plot(self):
        if not self.acquiring:
            return
        
        # Set start time on first data point
        if self.start_time is None:
            self.start_time = time.time()
            
        # Optimized data acquisition
        value = self.input_task.read(number_of_samples_per_channel=20)
        
        # Use real wall-clock time for actual sampling timestamps
        t = time.time() - self.start_time
        
        # Calculate instantaneous sampling rate (only during task for performance)
        if self.collection_mode is None:
            current_sample_time = time.time()
            if self.last_sample_time is not None:
                sample_interval = current_sample_time - self.last_sample_time
                if sample_interval > 0:
                    instantaneous_rate = 1.0 / sample_interval
                    self.sampling_rates.append(instantaneous_rate)
            self.last_sample_time = current_sample_time
    
        # Process all samples in the chunk for maximum data capture
        velocity = np.mean(value[0])  # ai0 - velocity
        position = np.mean(value[1])  # ai3 - position
        torque = -np.mean(value[2])  # ai2 - torque (MULTIPLY BY -1 TO FLIP SIGN)
        voltage = np.mean(value[3])  # ai5 - voltage
        
        # Handle different collection modes
        if self.collection_mode is not None:
            # This is baseline or MVC collection - simple processing
            self.collection_data.append([t, torque, velocity])
            
            # Check if collection duration is complete
            if t >= self.collection_duration:
                self.finish_collection()
                return
                
            # Update plots less frequently during collection for speed
            self.plot_update_counter += 1
            if self.plot_update_counter >= self.plot_update_interval:
                self._update_collection_plot()
                self.plot_update_counter = 0
        else:
            # This is the main task - optimized processing
            ao0 = 9  # Example: output value (change as needed)
            self.output_task.write(ao0)
            
            # Fast baseline correction with sanity check
            baseline_correction = self.get_baseline_corrected_value(t, torque_value=torque)
            
            # Sanity check: prevent extreme baseline corrections
            if abs(baseline_correction) > 10.0:  # Reasonable limit for torque baseline
                print(f"Warning: Extreme baseline correction {baseline_correction:.2f} at time {t:.2f}s, clamping to ±1.0")
                baseline_correction = np.sign(baseline_correction) * 1.0
            
            corrected_torque = torque - baseline_correction
            
            # Apply simple moving average for smooth plotting
            self.torque_history.append(corrected_torque)
            
            # Keep only the last N values for moving average
            if len(self.torque_history) > self.smoothing_window_size:
                self.torque_history.pop(0)
            
            # Calculate smoothed torque value
            self.smoothed_torque = np.mean(self.torque_history)
            
            # Buffer data for batch processing (store both raw and smoothed values)
            self.data_buffer.append([t, position, self.smoothed_torque, torque, voltage, velocity])
            
            # Process buffer when it reaches target size for efficiency
            if len(self.data_buffer) >= self.buffer_size:
                self._process_data_buffer()
            
            # Update plots much less frequently during task for maximum speed
            self.plot_update_counter += 1
            if self.plot_update_counter >= self.plot_update_interval:
                self._update_task_plots(t, self.smoothed_torque)
                self.plot_update_counter = 0

            # Stop after task_total_time seconds
            if t > self.task_total_time:
                self.finish_task()

    def _process_data_buffer(self):
        """Process buffered data in batch for efficiency"""
        if not self.data_buffer:
            return
            
        # Convert buffer to DataFrame and append in one operation
        buffer_df = pd.DataFrame(self.data_buffer, columns=self.df_task.columns)
        if self.df_task.empty:
            self.df_task = buffer_df
        else:
            self.df_task = pd.concat([self.df_task, buffer_df], ignore_index=True)
        
        # Clear buffer
        self.data_buffer = []

    def _update_collection_plot(self):
        """Optimized plot update for collection phases"""
        if len(self.collection_data) > 1:
            times = [row[0] for row in self.collection_data]
            torques = [row[1] for row in self.collection_data]
            
            # Moving window for plotting
            if len(times) < self.window_size:
                x_window = times
                y_window = torques
            else:
                x_window = times[-self.window_size:]
                y_window = torques[-self.window_size:]
            
            self.curve.setData(x_window, y_window)
            self.dot.setData([x_window[-1]], [y_window[-1]])
            self.plot_widget.setXRange(x_window[0], x_window[-1])

    def _update_task_plots(self, current_time, current_smoothed_torque=None):
        """Optimized plot update for task phase using curve fitting"""
        # Ensure any buffered data is processed first
        if self.data_buffer:
            self._process_data_buffer()
            
        if len(self.df_task) == 0:
            return
            
        # Moving window for plotting - simplified
        if len(self.df_task) < self.window_size:
            x_window = self.df_task["time"].values
            y_window = self.df_task["torque"].values
        else:
            x_window = self.df_task["time"].values[-self.window_size:]
            y_window = self.df_task["torque"].values[-self.window_size:]

        # Plot curve fitting prediction only if MVC data is available and fitted
        if (self.flexion_curve_fitter is not None and self.extension_curve_fitter is not None and self.mvc_fitted):
            
            # Simplified MVC plotting for performance
            window_start = current_time - (self.look_ahead_time * self.center_position)
            window_end = current_time + (self.look_ahead_time * (1 - self.center_position))
            
            # Reduced resolution for real-time performance
            time_points = np.linspace(window_start, window_end, 25)
            
            # Use GUI-specified duration for consistent cycle timing (normalized phases)
            time_offset = self.flexion_duration  # Always use GUI duration for consistent cycles
            cycle_period = 2 * time_offset  # flexion_duration + extension_duration (same length)
            
            # Determine phase for each time point
            cycle_times = time_points % cycle_period
            flexion_mask = cycle_times <= time_offset
            extension_mask = ~flexion_mask
            
            y_combined = np.zeros_like(time_points)
            
            # Use curve fitting for flexion phase
            if np.any(flexion_mask):
                flexion_phase_times = cycle_times[flexion_mask]
                
                # Normalize flexion phase times to the filtered MVC duration, then stretch to GUI duration
                flexion_df = self.flexion_curve_fitter.df_filtered
                max_flex_time = flexion_df["time_normalized"].max()
                
                # Stretch the phase times to match the recorded MVC duration for curve prediction
                normalized_flex_times = (flexion_phase_times / self.flexion_duration) * max_flex_time
                # Use modulo to repeat the MVC pattern cyclically
                cyclic_flex_times = normalized_flex_times % max_flex_time
                
                flexion_mvc_values = self.flexion_curve_fitter.predict(cyclic_flex_times)
                
                # Apply baseline correction if available (with same stretching approach)
                if self.flexion_baseline_fitter is not None:
                    baseline_df = self.flexion_baseline_fitter.df_filtered
                    max_base_time = baseline_df["time_normalized"].max()
                    # Stretch baseline times to match recorded baseline duration
                    normalized_base_times = (flexion_phase_times / self.flexion_duration) * max_base_time
                    cyclic_base_times = normalized_base_times % max_base_time
                    baseline_corrections = self.flexion_baseline_fitter.predict(cyclic_base_times)
                    flexion_mvc_values -= baseline_corrections
                
                y_combined[flexion_mask] = flexion_mvc_values
            
            # Use curve fitting for extension phase
            if np.any(extension_mask):
                extension_phase_times = cycle_times[extension_mask] - time_offset
                
                # Normalize extension phase times to the filtered MVC duration, then stretch to GUI duration
                extension_df = self.extension_curve_fitter.df_filtered
                max_ext_time = extension_df["time_normalized"].max()
                
                # Stretch the phase times to match the recorded MVC duration for curve prediction
                normalized_ext_times = (extension_phase_times / self.flexion_duration) * max_ext_time
                # Use modulo to repeat the MVC pattern cyclically
                cyclic_ext_times = normalized_ext_times % max_ext_time
                
                extension_mvc_values = self.extension_curve_fitter.predict(cyclic_ext_times)
                
                # Apply baseline correction if available (with same stretching approach)
                if self.extension_baseline_fitter is not None:
                    baseline_df = self.extension_baseline_fitter.df_filtered
                    max_base_time = baseline_df["time_normalized"].max()
                    # Stretch baseline times to match recorded baseline duration
                    normalized_base_times = (extension_phase_times / self.flexion_duration) * max_base_time
                    cyclic_base_times = normalized_base_times % max_base_time
                    baseline_corrections = self.extension_baseline_fitter.predict(cyclic_base_times)
                    extension_mvc_values -= baseline_corrections
                
                y_combined[extension_mask] = extension_mvc_values
            
            # Update MVC curve
            self.mvc_curve.setData(time_points, y_combined)
            
            # Update measured data curve with look-ahead windowing
            if len(self.df_task) > 1:
                current_x = x_window[-1]
                x_range_start = current_x - (self.look_ahead_time * self.center_position)
                x_range_end = current_x + (self.look_ahead_time * (1 - self.center_position))
                
                self.curve.setData(x_window, y_window)
                self.dot.setData([x_window[-1]], [y_window[-1]])
                
                # Use smoothed torque for the current dot position if available
                if current_smoothed_torque is not None:
                    self.dot.setData([current_time], [current_smoothed_torque])
                
                self.plot_widget.setXRange(x_range_start, x_range_end)
            else:
                self.curve.setData(x_window, y_window)
                self.dot.setData([x_window[-1]], [y_window[-1]])
                
                # Use smoothed torque for the current dot position if available
                if current_smoothed_torque is not None:
                    self.dot.setData([current_time], [current_smoothed_torque])
                
                self.plot_widget.setXRange(x_window[0], x_window[-1])
        else:
            # No MVC data - simple plotting
            self.curve.setData(x_window, y_window)
            self.dot.setData([x_window[-1]], [y_window[-1]])
            
            # Use smoothed torque for the current dot position if available
            if current_smoothed_torque is not None:
                self.dot.setData([current_time], [current_smoothed_torque])
            
            self.plot_widget.setXRange(x_window[0], x_window[-1])

    def finish_collection(self):
        """Finish baseline or MVC data collection"""
        self.timer.stop()
        self.stopDAQ()
        self.acquiring = False
        
        # Save data to appropriate DataFrame
        if self.collection_mode == 'flexion_baseline':
            self.df_baseline = pd.DataFrame(self.collection_data, columns=["time", "torque", "velocity"])
            print("Flexion Baseline set.")
            # Save baseline data to CSV
            self.df_baseline.to_csv("FlexionBaseline_Data_v4.csv", index=False)
            print("Flexion baseline data saved to FlexionBaseline_Data_v4.csv")
            # Calculate baseline curve fit
            self.calculate_flexion_baseline_fit()
            # Plot baseline analysis
            self.plot_baseline_analysis()
        elif self.collection_mode == 'extension_baseline':
            self.df_ext_baseline = pd.DataFrame(self.collection_data, columns=["time", "torque", "velocity"])
            print("Extension Baseline set.")
            # Save extension baseline data to CSV
            self.df_ext_baseline.to_csv("ExtensionBaseline_Data_v4.csv", index=False)
            print("Extension baseline data saved to ExtensionBaseline_Data_v4.csv")
            # Calculate baseline curve fit
            self.calculate_extension_baseline_fit()
            # Plot baseline analysis
            self.plot_baseline_analysis()
        elif self.collection_mode == 'flexion_mvc':
            self.df_flexionMVC = pd.DataFrame(self.collection_data, columns=["time", "torque", "velocity"])
            print("Flexion MVC set.")
            # Save flexion MVC data to CSV
            self.df_flexionMVC.to_csv("FlexionMVC_Data_v4.csv", index=False)
            print("Flexion MVC data saved to FlexionMVC_Data_v4.csv")
            # Check if both MVCs are available and calculate curve fit
            if hasattr(self, "df_extensionMVC") and not self.df_extensionMVC.empty:
                self.calculate_mvc_fit()
        elif self.collection_mode == 'extension_mvc':
            self.df_extensionMVC = pd.DataFrame(self.collection_data, columns=["time", "torque", "velocity"])
            print("Extension MVC set.")
            # Save extension MVC data to CSV
            self.df_extensionMVC.to_csv("ExtensionMVC_Data_v4.csv", index=False)
            print("Extension MVC data saved to ExtensionMVC_Data_v4.csv")
            # Check if both MVCs are available and calculate curve fit
            if hasattr(self, "df_flexionMVC") and not self.df_flexionMVC.empty:
                self.calculate_mvc_fit()
        
        # Reset collection variables
        self.collection_mode = None
        self.collection_data = []
        self.plot_widget.setTitle("Real-Time Torque Plot")

    def finish_task(self):
        """Finish the main task"""
        self.timer.stop()
        self.stopDAQ()
        self.acquiring = False
        
        # Process any remaining buffered data
        if self.data_buffer:
            self._process_data_buffer()
        
        # Save task data
        print(f"DEBUG: Current trial number before saving: {self.trialNum}")
        self.df_task.to_csv(f"DAQ_Task_Trial{self.trialNum}_v4.csv", index=False)
        print(f"Task data saved to DAQ_Task_Trial{self.trialNum}_v4.csv")
        self.trialNum += 1

        # Display sampling rate analysis
        self.display_sampling_rate_analysis()
        
        # Plot torque correction comparison
        self.plot_torque_correction_comparison()

    def display_sampling_rate_analysis(self):
        """Display real-time sampling rate analysis during task"""
        print(f"DEBUG: Sampling rate analysis using trial number: {self.trialNum}")
        if len(self.sampling_rates) > 0:
            sampling_rates_array = np.array(self.sampling_rates)
            mean_rate = np.mean(sampling_rates_array)
            std_rate = np.std(sampling_rates_array)
            min_rate = np.min(sampling_rates_array)
            max_rate = np.max(sampling_rates_array)
            
            print(f"\nSampling Rate Analysis:")
            print(f"  Mean: {mean_rate:.2f} Hz")
            print(f"  Std Dev: {std_rate:.2f} Hz")
            print(f"  Min: {min_rate:.2f} Hz")
            print(f"  Max: {max_rate:.2f} Hz")
            
            # Approximate time axis for CSV
            time_axis = np.arange(len(sampling_rates_array)) * 0.005  # Approximate 5ms intervals
        
        # Save sampling rate data to CSV
        sampling_data = pd.DataFrame({
            'sample_number': range(len(sampling_rates_array)),
            'time_approx': time_axis,
            'sampling_rate_hz': sampling_rates_array
        })
        sampling_data.to_csv(f"SamplingRate_Trial{self.trialNum}_v4.csv", index=False)
        print(f"Sampling rate data saved to SamplingRate_Trial{self.trialNum}_v4.csv")

    def filter_by_velocity(self, df, velocity_threshold=0.03, data_type="", phase_type=""):
        """Filter DataFrame based on phase-specific velocity criteria"""
        if 'velocity' not in df.columns:
            print(f"Warning: No velocity column found in {data_type}. Skipping velocity filtering.")
            return df
            
        initial_count = len(df)
        
        # Phase-specific filtering logic
        if phase_type.lower() == 'flexion':
            filtered_df = df[df['velocity'] <= -velocity_threshold]
            filter_description = f"velocity > -{velocity_threshold}"
        elif phase_type.lower() == 'extension':
            filtered_df = df[df['velocity'] >= velocity_threshold]
            filter_description = f"velocity < {velocity_threshold}"
        else:
            filtered_df = df[
                (df['velocity'] <= -velocity_threshold) | 
                (df['velocity'] >= velocity_threshold)
            ]
            filter_description = f"-{velocity_threshold} < velocity < {velocity_threshold}"
        
        final_count = len(filtered_df)
        removed_count = initial_count - final_count
        removed_percentage = (removed_count / initial_count) * 100 if initial_count > 0 else 0
        
        print(f"{data_type} velocity filtering results:")
        print(f"  Initial data points: {initial_count}")
        print(f"  Final data points: {final_count}")
        print(f"  Removed data points: {removed_count} ({removed_percentage:.1f}%)")
        print(f"  Filter criterion: removed where {filter_description}")
        print(f"  Phase type: {phase_type if phase_type else 'symmetric'}")
        
        return filtered_df    
        
    def calculate_mvc_fit(self):
        """Calculate and store curve fit parameters for flexion and extension MVC data separately"""
        if (hasattr(self, "df_flexionMVC") and not self.df_flexionMVC.empty and 
            hasattr(self, "df_extensionMVC") and not self.df_extensionMVC.empty):
            
            print("\nCalculating MVC curve fits...")
            
            # Create and fit flexion curve model
            self.flexion_curve_fitter = MVCCurveFitter()
            flexion_success = self.flexion_curve_fitter.load_data_from_dataframe(self.df_flexionMVC, "flexion")
            if flexion_success:
                flexion_fit_success = self.flexion_curve_fitter.fit_polynomial_models()
                if not flexion_fit_success:
                    print("Warning: Flexion curve fitting failed")
            else:
                print("Warning: Failed to load flexion MVC data for curve fitting")
            
            # Create and fit extension curve model
            self.extension_curve_fitter = MVCCurveFitter()
            extension_success = self.extension_curve_fitter.load_data_from_dataframe(self.df_extensionMVC, "extension")
            if extension_success:
                extension_fit_success = self.extension_curve_fitter.fit_polynomial_models()
                if not extension_fit_success:
                    print("Warning: Extension curve fitting failed")
            else:
                print("Warning: Failed to load extension MVC data for curve fitting")
            
            # Mark as fitted if both succeeded
            self.mvc_fitted = (flexion_success and extension_success and 
                             flexion_fit_success and extension_fit_success)
            
            if self.mvc_fitted:
                print("MVC curve fitting completed successfully!")
                # Plot combined MVC analysis
                self.plot_combined_mvc_analysis()
            else:
                print("MVC curve fitting incomplete - some models failed")

    def calculate_flexion_baseline_fit(self):
        """Calculate and store curve fit parameters for flexion baseline data"""
        if hasattr(self, "df_baseline") and not self.df_baseline.empty:
            print("Calculating flexion baseline curve fit...")
            
            self.flexion_baseline_fitter = MVCCurveFitter()
            success = self.flexion_baseline_fitter.load_data_from_dataframe(self.df_baseline, "flexion")
            if success:
                self.flexion_baseline_fitted = self.flexion_baseline_fitter.fit_polynomial_models()
                if self.flexion_baseline_fitted:
                    print("Flexion baseline curve fitting completed!")
                else:
                    print("Warning: Flexion baseline curve fitting failed")
            else:
                print("Warning: Failed to load flexion baseline data for curve fitting")

    def calculate_extension_baseline_fit(self):
        """Calculate and store curve fit parameters for extension baseline data"""
        if hasattr(self, "df_ext_baseline") and not self.df_ext_baseline.empty:
            print("Calculating extension baseline curve fit...")
            
            self.extension_baseline_fitter = MVCCurveFitter()
            success = self.extension_baseline_fitter.load_data_from_dataframe(self.df_ext_baseline, "extension")
            if success:
                self.extension_baseline_fitted = self.extension_baseline_fitter.fit_polynomial_models()
                if self.extension_baseline_fitted:
                    print("Extension baseline curve fitting completed!")
                else:
                    print("Warning: Extension baseline curve fitting failed")
            else:
                print("Warning: Failed to load extension baseline data for curve fitting")

    def plot_baseline_analysis(self):
        """Plot flexion and extension baseline data with their curve fits"""
        # Check if we have at least one baseline
        has_flexion = hasattr(self, "df_baseline") and not self.df_baseline.empty and self.flexion_baseline_fitted
        has_extension = hasattr(self, "df_ext_baseline") and not self.df_ext_baseline.empty and self.extension_baseline_fitted
        
        if not (has_flexion or has_extension):
            print("No baseline data available for plotting.")
            return
            
        # Create the matplotlib plot
        plt.figure(figsize=(12, 6))
        
        plot_count = 0
        if has_flexion:
            plot_count += 1
        if has_extension:
            plot_count += 1
            
        current_plot = 1
        
        # Plot flexion baseline if available
        if has_flexion:
            plt.subplot(1, plot_count, current_plot)
            
            # Get filtered data
            df_filtered = self.flexion_baseline_fitter.df_filtered
            time_norm = df_filtered["time_normalized"].values
            torque_actual = df_filtered["torque"].values
            torque_pred = self.flexion_baseline_fitter.predict(time_norm)
            
            plt.plot(time_norm, torque_actual, 'b-', linewidth=2, alpha=0.7, label='Filtered Data')
            plt.plot(time_norm, torque_pred, 'r--', linewidth=3, 
                    label=f'Curve Fit (degree {self.flexion_baseline_fitter.best_degree})')
            plt.xlabel('Normalized Time (s)')
            plt.ylabel('Torque')
            plt.title(f'Flexion Baseline Curve Fit\nR² = {self.flexion_baseline_fitter.best_r2:.4f}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            current_plot += 1
            
        # Plot extension baseline if available
        if has_extension:
            plt.subplot(1, plot_count, current_plot)
            
            # Get filtered data
            df_filtered = self.extension_baseline_fitter.df_filtered
            time_norm = df_filtered["time_normalized"].values
            torque_actual = df_filtered["torque"].values
            torque_pred = self.extension_baseline_fitter.predict(time_norm)
            
            plt.plot(time_norm, torque_actual, 'b-', linewidth=2, alpha=0.7, label='Filtered Data')
            plt.plot(time_norm, torque_pred, 'r--', linewidth=3, 
                    label=f'Curve Fit (degree {self.extension_baseline_fitter.best_degree})')
            plt.xlabel('Normalized Time (s)')
            plt.ylabel('Torque')
            plt.title(f'Extension Baseline Curve Fit\nR² = {self.extension_baseline_fitter.best_r2:.4f}')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print analysis information
        print(f"\nBaseline Analysis Results:")
        if has_flexion:
            print(f"  Flexion: Polynomial degree {self.flexion_baseline_fitter.best_degree}, "
                  f"R² = {self.flexion_baseline_fitter.best_r2:.4f}")
        if has_extension:
            print(f"  Extension: Polynomial degree {self.extension_baseline_fitter.best_degree}, "
                  f"R² = {self.extension_baseline_fitter.best_r2:.4f}")

    def get_baseline_corrected_value(self, time_val, torque_value=None, is_flexion_phase=None):
        """Calculate baseline curve value for subtraction at given time using normalized time stretching"""
        # Determine which baseline to use based on torque sign if available
        if torque_value is not None:
            use_flexion_baseline = torque_value > 0
        elif is_flexion_phase is not None:
            use_flexion_baseline = is_flexion_phase
        else:
            use_flexion_baseline = True  # Default to flexion
        
        if use_flexion_baseline and self.flexion_baseline_fitted:
            # Normalize time to consistent duration, then stretch to match recorded baseline duration
            baseline_df = self.flexion_baseline_fitter.df_filtered
            max_training_time = baseline_df["time_normalized"].max()
            
            # Create consistent cycle timing based on GUI duration
            cycle_time = time_val % self.flexion_duration
            # Stretch to match recorded baseline duration for curve prediction
            normalized_time = (cycle_time / self.flexion_duration) * max_training_time
            baseline_value = self.flexion_baseline_fitter.predict(np.array([normalized_time]))[0]
            
            return baseline_value
            
        elif not use_flexion_baseline and self.extension_baseline_fitted:
            # Same stretching approach for extension baseline
            baseline_df = self.extension_baseline_fitter.df_filtered
            max_training_time = baseline_df["time_normalized"].max()
            
            # Create consistent cycle timing based on GUI duration
            cycle_time = time_val % self.flexion_duration
            # Stretch to match recorded baseline duration for curve prediction
            normalized_time = (cycle_time / self.flexion_duration) * max_training_time
            baseline_value = self.extension_baseline_fitter.predict(np.array([normalized_time]))[0]
            
            return baseline_value
        else:
            return 0.0  # No baseline correction available

    def plot_combined_mvc_analysis(self):
        """Plot individual MVC data with separate curve fits and their combination"""
        if (hasattr(self, "df_flexionMVC") and not self.df_flexionMVC.empty and 
            hasattr(self, "df_extensionMVC") and not self.df_extensionMVC.empty and
            self.mvc_fitted):
            
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Flexion MVC with curve fit
            plt.subplot(2, 2, 1)
            df_flex_filtered = self.flexion_curve_fitter.df_filtered
            time_norm_flex = df_flex_filtered["time_normalized"].values
            torque_actual_flex = df_flex_filtered["torque"].values
            torque_pred_flex = self.flexion_curve_fitter.predict(time_norm_flex)
            
            plt.plot(self.df_flexionMVC["time"].values, self.df_flexionMVC["torque"].values, 'b-', alpha=0.3, linewidth=1, label='Raw Data')
            plt.plot(df_flex_filtered["time"].values, torque_actual_flex, 'b-', linewidth=2, alpha=0.7, label='Filtered Data')
            plt.plot(df_flex_filtered["time"].values, torque_pred_flex, 'r--', linewidth=3, 
                    label=f'Curve Fit (degree {self.flexion_curve_fitter.best_degree})')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque')
            plt.title(f'Flexion MVC Curve Fit\nR² = {self.flexion_curve_fitter.best_r2:.4f}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 2: Extension MVC with curve fit
            plt.subplot(2, 2, 2)
            df_ext_filtered = self.extension_curve_fitter.df_filtered
            time_norm_ext = df_ext_filtered["time_normalized"].values
            torque_actual_ext = df_ext_filtered["torque"].values
            torque_pred_ext = self.extension_curve_fitter.predict(time_norm_ext)
            
            plt.plot(self.df_extensionMVC["time"].values, self.df_extensionMVC["torque"].values, 'g-', alpha=0.3, linewidth=1, label='Raw Data')
            plt.plot(df_ext_filtered["time"].values, torque_actual_ext, 'g-', linewidth=2, alpha=0.7, label='Filtered Data')
            plt.plot(df_ext_filtered["time"].values, torque_pred_ext, 'r--', linewidth=3, 
                    label=f'Curve Fit (degree {self.extension_curve_fitter.best_degree})')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque')
            plt.title(f'Extension MVC Curve Fit\nR² = {self.extension_curve_fitter.best_r2:.4f}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 3: Combined cycle prediction
            plt.subplot(2, 1, 2)
            
            # Get the actual normalized durations from the curve fitters
            flexion_max_time = self.flexion_curve_fitter.df_filtered["time_normalized"].max()
            extension_max_time = self.extension_curve_fitter.df_filtered["time_normalized"].max()
            
            # Use GUI-specified duration for consistent cycle timing (like in the real-time task)
            gui_duration = float(self.flexion_duration_box.text()) if hasattr(self, 'flexion_duration_box') else 5.0
            cycle_period = 2 * gui_duration  # flexion + extension (same duration each)
            
            # Generate 2.5 cycles for visualization
            total_time = 2.5 * cycle_period
            time_points = np.linspace(0, total_time, int(total_time * 50))  # 50 Hz resolution
            
            y_combined = np.zeros_like(time_points)
            
            for i, t in enumerate(time_points):
                cycle_time = t % cycle_period
                if cycle_time <= gui_duration:
                    # Flexion phase - stretch normalized time to match recorded MVC duration
                    normalized_flex_time = (cycle_time / gui_duration) * flexion_max_time
                    y_combined[i] = self.flexion_curve_fitter.predict(np.array([normalized_flex_time]))[0]
                else:
                    # Extension phase - stretch normalized time to match recorded MVC duration
                    extension_phase_time = cycle_time - gui_duration
                    normalized_ext_time = (extension_phase_time / gui_duration) * extension_max_time
                    y_combined[i] = self.extension_curve_fitter.predict(np.array([normalized_ext_time]))[0]
            
            plt.plot(time_points, y_combined, 'r-', linewidth=3, label='Combined MVC Curve Prediction')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # Add individual curves for comparison (first cycle only)
            flexion_time_points = np.linspace(0, gui_duration, int(gui_duration * 50))
            flexion_normalized = (flexion_time_points / gui_duration) * flexion_max_time
            flexion_values = self.flexion_curve_fitter.predict(flexion_normalized)
            plt.plot(flexion_time_points, flexion_values, 'b--', linewidth=2, alpha=0.7, label='Individual Flexion Curve')
            
            extension_time_points = np.linspace(gui_duration, 2*gui_duration, int(gui_duration * 50))
            extension_phase_time = extension_time_points - gui_duration
            extension_normalized = (extension_phase_time / gui_duration) * extension_max_time
            extension_values = self.extension_curve_fitter.predict(extension_normalized)
            plt.plot(extension_time_points, extension_values, 'g--', linewidth=2, alpha=0.7, label='Individual Extension Curve')
            
            plt.xlabel('Time (s)')
            plt.ylabel('Torque')
            plt.title(f'Combined MVC Cycle Prediction (Curve Fitting)\nGUI Duration: {gui_duration}s, Flexion Recorded: {flexion_max_time:.2f}s, Extension Recorded: {extension_max_time:.2f}s')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nMVC Curve Fitting Analysis Results:")
            print(f"  Flexion: Polynomial degree {self.flexion_curve_fitter.best_degree}, "
                  f"R² = {self.flexion_curve_fitter.best_r2:.4f}")
            print(f"  Extension: Polynomial degree {self.extension_curve_fitter.best_degree}, "
                  f"R² = {self.extension_curve_fitter.best_r2:.4f}")
            print(f"  GUI Duration: {gui_duration}s per phase")
            print(f"  Recorded Flexion Duration: {flexion_max_time:.2f}s")
            print(f"  Recorded Extension Duration: {extension_max_time:.2f}s")
            print(f"  Time Stretching Factors: Flexion {flexion_max_time/gui_duration:.2f}x, Extension {extension_max_time/gui_duration:.2f}x")

    def plot_torque_correction_comparison(self):
        """Plot comparison between uncorrected and corrected torque data with MVC reference"""
        if hasattr(self, "df_task") and not self.df_task.empty:
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Uncorrected vs Corrected Torque with MVC line
            plt.subplot(2, 1, 1)
            
            # Plot measured torque data
            plt.plot(self.df_task["time"].values, self.df_task["uncorrected_torque"].values, 'b-', alpha=0.7, linewidth=2, label='Uncorrected Torque')
            plt.plot(self.df_task["time"].values, self.df_task["torque"].values, 'r-', linewidth=2, label='Baseline Corrected Torque')
            
            # Add MVC line if available
            if (hasattr(self, 'flexion_curve_fitter') and self.flexion_curve_fitter is not None and 
                hasattr(self, 'extension_curve_fitter') and self.extension_curve_fitter is not None and 
                self.mvc_fitted):
                
                # Generate MVC line using the same approach as real-time task
                time_points = self.df_task["time"].values
                mvc_values = np.zeros_like(time_points)
                
                # Get GUI duration and curve fitting parameters
                gui_duration = self.flexion_duration if hasattr(self, 'flexion_duration') else 5.0
                cycle_period = 2 * gui_duration
                
                flexion_max_time = self.flexion_curve_fitter.df_filtered["time_normalized"].max()
                extension_max_time = self.extension_curve_fitter.df_filtered["time_normalized"].max()
                
                for i, t in enumerate(time_points):
                    cycle_time = t % cycle_period
                    if cycle_time <= gui_duration:
                        # Flexion phase - stretch normalized time to match recorded MVC duration
                        normalized_flex_time = (cycle_time / gui_duration) * flexion_max_time
                        mvc_val = self.flexion_curve_fitter.predict(np.array([normalized_flex_time]))[0]
                        
                        # Apply baseline correction if available
                        if hasattr(self, 'flexion_baseline_fitter') and self.flexion_baseline_fitter is not None:
                            baseline_df = self.flexion_baseline_fitter.df_filtered
                            max_base_time = baseline_df["time_normalized"].max()
                            normalized_base_time = (cycle_time / gui_duration) * max_base_time
                            baseline_correction = self.flexion_baseline_fitter.predict(np.array([normalized_base_time]))[0]
                            mvc_val -= baseline_correction
                        
                        mvc_values[i] = mvc_val
                    else:
                        # Extension phase - stretch normalized time to match recorded MVC duration
                        extension_phase_time = cycle_time - gui_duration
                        normalized_ext_time = (extension_phase_time / gui_duration) * extension_max_time
                        mvc_val = self.extension_curve_fitter.predict(np.array([normalized_ext_time]))[0]
                        
                        # Apply baseline correction if available
                        if hasattr(self, 'extension_baseline_fitter') and self.extension_baseline_fitter is not None:
                            baseline_df = self.extension_baseline_fitter.df_filtered
                            max_base_time = baseline_df["time_normalized"].max()
                            normalized_base_time = (extension_phase_time / gui_duration) * max_base_time
                            baseline_correction = self.extension_baseline_fitter.predict(np.array([normalized_base_time]))[0]
                            mvc_val -= baseline_correction
                        
                        mvc_values[i] = mvc_val
                
                plt.plot(time_points, mvc_values, 'k--', linewidth=3, alpha=0.8, label='Target MVC Curve')
            
            plt.xlabel('Time (s)')
            plt.ylabel('Torque')
            plt.title('Torque Correction Comparison with MVC Reference')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Baseline correction values
            plt.subplot(2, 1, 2)
            correction_values = self.df_task["uncorrected_torque"].values - self.df_task["torque"].values
            plt.plot(self.df_task["time"].values, correction_values, 'g-', linewidth=2, label='Baseline Correction Applied')
            plt.xlabel('Time (s)')
            plt.ylabel('Correction Value')
            plt.title('Baseline Correction Values Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        else:
            print("No task data available for plotting torque correction comparison.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
