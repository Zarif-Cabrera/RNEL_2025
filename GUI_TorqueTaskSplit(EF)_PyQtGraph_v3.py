import sys
import time
import nidaqmx
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Fixed Y-axis bounds for all plots
Y_AXIS_BOUND = 1

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time DAQ Plotting v3 - Optimized Performance")
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
        font.setPointSize(12)  # Larger font for axis numbers
        self.plot_widget.getAxis('left').setStyle(tickFont=font)
        self.plot_widget.getAxis('bottom').setStyle(tickFont=font)
        self.plot_widget.getAxis('left').setPen(pg.mkPen(width=2))  # Thicker axis lines
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(width=2))
        
        self.layout.addWidget(self.plot_widget)
        # Add MVC curve first (bottom layer)
        self.mvc_curve = self.plot_widget.plot(pen=pg.mkPen('g', width=5, style=QtCore.Qt.DotLine))  # Thicker MVC curve
        # Add blue curve last (top layer) so it appears over MVC line
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=4))  # Thicker main curve
        self.dot = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('b'))  # Larger dot
        self.plot_widget.addItem(self.dot)

        # Data storage as pandas DataFrames
        self.df_task = pd.DataFrame(columns=["time", "position", "torque", "uncorrected_torque", "voltage", "velocity"])
        self.df_baseline = pd.DataFrame(columns=["time", "torque", "velocity"])
        self.df_flexionMVC = pd.DataFrame(columns=["time", "torque", "velocity"])
        self.df_ext_baseline = pd.DataFrame(columns=["time", "torque", "velocity"])
        self.df_extensionMVC = pd.DataFrame(columns=["time", "torque", "velocity"])

        self.window_size = 100
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.start_time = None
        self.acquiring = False
        self.trialNum = 1
        
        # Add variables for baseline/MVC data collection
        self.collection_mode = None  # 'flexion_baseline', 'extension_baseline', 'flexion_mvc', 'extension_mvc'
        self.collection_data = []
        self.collection_duration = 0
        
        # Look-ahead parameters for centered real-time display
        self.look_ahead_time = 1.0  # seconds of preview time
        self.center_position = 0.5  # position of current point (0.75 = 75% from left edge)
        
        # Performance optimization variables
        self.plot_update_counter = 0
        self.plot_update_interval = 2  # Only update plots every 2 data samples
        self.data_buffer = []  # Buffer for batch processing
        self.buffer_size = 10  # Process data in batches

        # Stored sinusoidal fit parameters (calculated once after MVC collection)
        self.flexion_freq = None
        self.flexion_reg = None
        self.extension_freq = None
        self.extension_reg = None
        self.mvc_fitted = False
        
        # Baseline sinusoidal fit parameters
        self.flexion_baseline_freq = None
        self.flexion_baseline_reg = None
        self.extension_baseline_freq = None
        self.extension_baseline_reg = None
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
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=4), name='Baseline Torque')  # Thicker baseline curve
        self.dot = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('b'))  # Larger dot
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(5)  # Same 10ms interval as task

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
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=4), name='Baseline Torque')  # Thicker baseline curve
        self.dot = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('b'))  # Larger dot
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(5)  # Same 10ms interval as task

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
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=4), name='MVC Torque')  # Thicker MVC curve
        self.dot = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('b'))  # Larger dot
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(5)  # Same 8ms interval as task

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
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=4), name='MVC Torque')  # Thicker MVC curve
        self.dot = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('b'))  # Larger dot
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(5)  # Same 8ms interval as task

    def start_task(self):
        self.acquiring = True
        self.df_task = pd.DataFrame(columns=["time", "position", "torque", "uncorrected_torque", "voltage", "velocity"])
        self.start_time = None  # Will be set on first data point
        self.collection_mode = None  # Reset collection mode for task
        
        # Reset sampling rate tracking
        self.sampling_rates = []
        self.last_sample_time = None
        
        # Reset performance optimization variables for task
        self.plot_update_counter = 0
        self.data_buffer = []
        
        # Setup plot for task with FIXED y-range based on MVC data
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
        self.plot_widget.setMouseEnabled(x=True, y=False)  # Allow x-pan but disable y interactions
        self.plot_widget.hideButtons()  # Hide auto-range button
        
        # Add MVC curve first (bottom layer)
        self.mvc_curve = self.plot_widget.plot(pen=pg.mkPen('r', width=7, style=QtCore.Qt.DotLine), name='Combined MVC')  # Even thicker MVC curve
        # Add blue curve last (top layer) so it appears over MVC line
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=4), name='Measured Torque')  # Thicker main curve
        self.dot = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('b'))  # Larger dot
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
        self.timer.start(5)  # update every 1 ms

    def update_plot(self):
        if not self.acquiring:
            return
        
        # Set start time on first data point
        if self.start_time is None:
            self.start_time = time.time()
            
        # Optimized data acquisition - read larger chunks less frequently
        value = self.input_task.read(number_of_samples_per_channel=20)  # Increased from 10 to 20
        
        # Use real wall-clock time for actual sampling timestamps
        t = time.time() - self.start_time
        
        # Calculate instantaneous sampling rate (only during task for performance)
        if self.collection_mode is None:  # Only track during task
            current_sample_time = time.time()
            if self.last_sample_time is not None:
                interval = current_sample_time - self.last_sample_time
                instantaneous_rate = 1.0 / interval if interval > 0 else 0
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
            self.collection_data.append([t, torque, velocity])  # torque is now flipped, velocity added
            
            # Check if collection duration is complete
            if t >= self.collection_duration:
                self.finish_collection()
                return
                
            # Update plots less frequently during collection for speed
            self.plot_update_counter += 1
            if self.plot_update_counter >= self.plot_update_interval:
                self.plot_update_counter = 0
                self._update_collection_plot()
        else:
            # This is the main task - optimized processing
            ao0 = 9  # Example: output value (change as needed)
            self.output_task.write(ao0)
            
            # Fast baseline correction
            baseline_correction = self.get_baseline_corrected_value(t, torque_value=torque)
            corrected_torque = torque - baseline_correction
            
            # Buffer data for batch processing - torque is already flipped, velocity added
            self.data_buffer.append([t, position, corrected_torque, torque, voltage, velocity])
            
            # Process buffer when it reaches target size for efficiency
            if len(self.data_buffer) >= self.buffer_size:
                self._process_data_buffer()
            
            # Update plots much less frequently during task for maximum speed
            self.plot_update_counter += 1
            if self.plot_update_counter >= self.plot_update_interval:
                self.plot_update_counter = 0
                self._update_task_plots(t)

            # Stop after task_total_time seconds
            if t > self.task_total_time:
                # Process any remaining buffered data
                if self.data_buffer:
                    self._process_data_buffer()
                    
                self.timer.stop()
                # Set output to 0 before stopping
                self.output_task.write(0)
                
                # Read voltage from ai5 after setting output to 0
                try:
                    final_voltage_reading = self.input_task.read()
                    final_voltage = final_voltage_reading[3]  # ai5 is now the fourth channel (index 3)
                    print(f"\nFinal voltage reading from ai5 after setting ao0 to 0: {final_voltage:.6f} V")
                except Exception as e:
                    print(f"Error reading final voltage: {e}")
                
                self.stopDAQ()
                self.acquiring = False
                
                # Display sampling rate analysis
                self.display_sampling_rate_analysis()
                
                # Plot torque correction comparison
                self.plot_torque_correction_comparison()
                
                # --- Save all data to CSV ---
                self.df_task.to_csv(f"DAQ_Task_Trial{self.trialNum}_v3.csv", index=False)
                print("Task complete. Data saved.")
                self.trialNum += 1

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

    def _update_task_plots(self, current_time):
        """Optimized plot update for task phase"""
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

        # Plot sinusoid only if MVC data is available - optimized
        if (hasattr(self, "df_flexionMVC") and not self.df_flexionMVC.empty and 
            hasattr(self, "df_extensionMVC") and not self.df_extensionMVC.empty and
            self.mvc_fitted):
            
            # Simplified MVC plotting for performance
            window_start = current_time - (self.look_ahead_time * self.center_position)
            window_end = current_time + (self.look_ahead_time * (1 - self.center_position))
            
            # Reduced resolution for real-time performance
            time_points = np.linspace(window_start, window_end, 25)  # Reduced from 50 to 25
            
            # Fast sinusoidal calculation
            flexion_times = self.df_flexionMVC["time"].values
            time_offset = flexion_times[-1] if len(flexion_times) > 0 else 0
            cycle_period = 2 * time_offset
            
            cycle_times = time_points % cycle_period
            flexion_mask = cycle_times <= time_offset
            extension_mask = ~flexion_mask
            
            y_combined = np.zeros_like(time_points)
            
            # Optimized flexion calculation
            if np.any(flexion_mask):
                flexion_phase_times = cycle_times[flexion_mask]
                X_flexion = np.column_stack((
                    np.sin(2 * np.pi * self.flexion_freq * flexion_phase_times),
                    np.cos(2 * np.pi * self.flexion_freq * flexion_phase_times)
                ))
                flexion_mvc_values = self.flexion_reg.predict(X_flexion)
                
                # Fast baseline correction
                if self.flexion_baseline_fitted:
                    baseline_corrections = np.array([
                        self.get_baseline_corrected_value(t, is_flexion_phase=True) 
                        for t in flexion_phase_times
                    ])
                    flexion_mvc_values -= baseline_corrections
                
                y_combined[flexion_mask] = flexion_mvc_values
            
            # Optimized extension calculation
            if np.any(extension_mask):
                extension_phase_times = cycle_times[extension_mask] - time_offset
                X_extension = np.column_stack((
                    np.sin(2 * np.pi * self.extension_freq * extension_phase_times),
                    np.cos(2 * np.pi * self.extension_freq * extension_phase_times)
                ))
                extension_mvc_values = self.extension_reg.predict(X_extension)
                
                # Fast baseline correction
                if self.extension_baseline_fitted:
                    baseline_corrections = np.array([
                        self.get_baseline_corrected_value(t, is_flexion_phase=False) 
                        for t in extension_phase_times
                    ])
                    extension_mvc_values -= baseline_corrections
                
                y_combined[extension_mask] = extension_mvc_values
            
            # Update MVC curve
            self.mvc_curve.setData(time_points, y_combined)
            
            # Simplified windowing for task data
            if len(self.df_task) > 1:
                task_times = self.df_task["time"].values
                task_torques = self.df_task["torque"].values
                
                # Simple windowing without complex filtering
                window_mask = (task_times >= window_start) & (task_times <= window_end)
                if np.any(window_mask):
                    windowed_times = task_times[window_mask]
                    windowed_torques = task_torques[window_mask]
                    self.curve.setData(windowed_times, windowed_torques)
                    self.dot.setData([current_time], [task_torques[-1]])
                    self.plot_widget.setXRange(window_start, window_end)
                else:
                    # Fallback
                    self.curve.setData(x_window, y_window)
                    self.dot.setData([x_window[-1]], [y_window[-1]])
                    self.plot_widget.setXRange(x_window[0], x_window[-1])
            else:
                self.curve.setData(x_window, y_window)
                self.dot.setData([x_window[-1]], [y_window[-1]])
                self.plot_widget.setXRange(x_window[0], x_window[-1])
        else:
            # No MVC data - simple plotting
            self.curve.setData(x_window, y_window)
            self.dot.setData([x_window[-1]], [y_window[-1]])
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
            self.df_baseline.to_csv("FlexionBaseline_Data_v3.csv", index=False)
            print("Flexion baseline data saved to FlexionBaseline_Data_v3.csv")
            # Calculate baseline sinusoidal fit
            self.calculate_flexion_baseline_fit()
            # Plot baseline analysis
            self.plot_baseline_analysis()
        elif self.collection_mode == 'extension_baseline':
            self.df_ext_baseline = pd.DataFrame(self.collection_data, columns=["time", "torque", "velocity"])
            print("Extension Baseline set.")
            # Save extension baseline data to CSV
            self.df_ext_baseline.to_csv("ExtensionBaseline_Data_v3.csv", index=False)
            print("Extension baseline data saved to ExtensionBaseline_Data_v3.csv")
            # Calculate baseline sinusoidal fit
            self.calculate_extension_baseline_fit()
            # Plot baseline analysis
            self.plot_baseline_analysis()
        elif self.collection_mode == 'flexion_mvc':
            self.df_flexionMVC = pd.DataFrame(self.collection_data, columns=["time", "torque", "velocity"])
            print("Flexion MVC set.")
            # Save flexion MVC data to CSV
            self.df_flexionMVC.to_csv("FlexionMVC_Data_v3.csv", index=False)
            print("Flexion MVC data saved to FlexionMVC_Data_v3.csv")
            # Check if both MVCs are available and calculate sinusoidal fit
            if hasattr(self, "df_extensionMVC") and not self.df_extensionMVC.empty:
                self.calculate_mvc_fit()
                self.plot_combined_mvc_analysis()
        elif self.collection_mode == 'extension_mvc':
            self.df_extensionMVC = pd.DataFrame(self.collection_data, columns=["time", "torque", "velocity"])
            print("Extension MVC set.")
            # Save extension MVC data to CSV
            self.df_extensionMVC.to_csv("ExtensionMVC_Data_v3.csv", index=False)
            print("Extension MVC data saved to ExtensionMVC_Data_v3.csv")
            # Check if both MVCs are available and calculate sinusoidal fit
            if hasattr(self, "df_flexionMVC") and not self.df_flexionMVC.empty:
                self.calculate_mvc_fit()
                self.plot_combined_mvc_analysis()
        
        # Reset collection variables
        self.collection_mode = None
        self.collection_data = []
        self.plot_widget.setTitle("Real-Time Torque Plot")  # Reset title

    def display_sampling_rate_analysis(self):
        """Display real-time sampling rate analysis during task"""
        if len(self.sampling_rates) > 0:
            sampling_rates_array = np.array(self.sampling_rates)
            
            # Calculate statistics
            mean_rate = np.mean(sampling_rates_array)
            std_rate = np.std(sampling_rates_array)
            min_rate = np.min(sampling_rates_array)
            max_rate = np.max(sampling_rates_array)
            
            # Print statistics
            print(f"\nTask Sampling Rate Analysis:")
            print(f"Window size: {self.window_size} points")
            print(f"Display optimization: {'Enabled' if self.window_size > 100 else 'Disabled'}")
            print(f"Mean sampling rate: {mean_rate:.1f} Hz")
            print(f"Standard deviation: {std_rate:.2f} Hz")
            print(f"Min rate: {min_rate:.1f} Hz")
            print(f"Max rate: {max_rate:.1f} Hz")
            print(f"Target rate: 50.0 Hz")
            print(f"Rate stability: {((50.0 - std_rate)/50.0)*100:.1f}%")
            print(f"Performance impact: {((50.0 - mean_rate)/50.0)*100:.1f}% reduction from target")
            
            # Create a time series plot of sampling rates
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Sampling rate over time
            plt.subplot(2, 1, 1)
            time_axis = np.arange(len(sampling_rates_array)) * 0.02  # 20ms intervals for 50Hz
            plt.plot(time_axis, sampling_rates_array, 'b-', linewidth=1, alpha=0.7, label='Instantaneous Rate')
            plt.axhline(y=mean_rate, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_rate:.1f} Hz')
            plt.axhline(y=50, color='g', linestyle='--', linewidth=2, label='Target: 50.0 Hz')
            plt.fill_between(time_axis, mean_rate - std_rate, mean_rate + std_rate, alpha=0.3, color='red', label=f'±1 SD: {std_rate:.2f} Hz')
            plt.xlabel('Time (s)')
            plt.ylabel('Sampling Rate (Hz)')
            plt.title('Real-Time Sampling Rate During Task')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.ylim(max(0, min_rate - 10), max_rate + 10)
            
            # Plot 2: Histogram of sampling rates
            plt.subplot(2, 1, 2)
            plt.hist(sampling_rates_array, bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(x=mean_rate, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_rate:.1f} Hz')
            plt.axvline(x=50, color='g', linestyle='--', linewidth=2, label='Target: 50.0 Hz')
            plt.xlabel('Sampling Rate (Hz)')
            plt.ylabel('Count')
            plt.title('Distribution of Sampling Rates')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        # Save sampling rate data to CSV
        sampling_data = pd.DataFrame({
            'sample_number': range(len(sampling_rates_array)),
            'time_approx': time_axis,
            'sampling_rate_hz': sampling_rates_array
        })
        sampling_data.to_csv(f"SamplingRate_Trial{self.trialNum}_v3.csv", index=False)
        print(f"Sampling rate data saved to SamplingRate_Trial{self.trialNum}_v3.csv")

    def filter_by_velocity(self, df, velocity_threshold=0.03, data_type="", phase_type=""):
        """Filter DataFrame based on phase-specific velocity criteria
        
        Args:
            df: DataFrame to filter
            velocity_threshold: velocity threshold (default 0.02)
            data_type: description for logging
            phase_type: 'flexion' or 'extension' for phase-specific filtering
        """
        if 'velocity' not in df.columns:
            print(f"Warning: No velocity column found in {data_type} data. Skipping velocity filtering.")
            return df
            
        initial_count = len(df)
        
        # Phase-specific filtering logic
        if phase_type.lower() == 'flexion':
            # For flexion: remove rows where velocity > -threshold (keep velocity <= -threshold, i.e., negative velocities)
            filtered_df = df[df['velocity'] <= -velocity_threshold]
            filter_description = f"velocity > -{velocity_threshold}"
        elif phase_type.lower() == 'extension':
            # For extension: remove rows where velocity < threshold (keep velocity >= threshold, i.e., positive velocities)  
            filtered_df = df[df['velocity'] >= velocity_threshold]
            filter_description = f"velocity < {velocity_threshold}"
        else:
            # Fallback to original symmetric filtering if phase not specified
            filtered_df = df[(df['velocity'] <= -velocity_threshold) | (df['velocity'] >= velocity_threshold)]
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
        """Calculate and store sinusoidal fit parameters for flexion and extension MVC data separately"""
        if (hasattr(self, "df_flexionMVC") and not self.df_flexionMVC.empty and 
            hasattr(self, "df_extensionMVC") and not self.df_extensionMVC.empty):
            
            # Filter flexion MVC data by velocity
            print("Applying velocity filtering to MVC data...")
            filtered_flexion_df = self.filter_by_velocity(self.df_flexionMVC, velocity_threshold=0.03, data_type="Flexion MVC", phase_type="flexion")
            filtered_extension_df = self.filter_by_velocity(self.df_extensionMVC, velocity_threshold=0.03, data_type="Extension MVC", phase_type="extension")
            
            if len(filtered_flexion_df) == 0 or len(filtered_extension_df) == 0:
                print("Warning: Not enough data points remain after velocity filtering for MVC calculation. Using original data.")
                filtered_flexion_df = self.df_flexionMVC
                filtered_extension_df = self.df_extensionMVC
            
            # Fit sinusoid to filtered flexion MVC data
            flexion_times_raw = filtered_flexion_df["time"].values
            flexion_torques = filtered_flexion_df["torque"].values
            
            # Normalize time to start at 0 for better phase consistency
            flexion_times = flexion_times_raw - flexion_times_raw[0]
            
            dt_flexion = np.mean(np.diff(flexion_times))
            N_flexion = len(flexion_times)
            yf_flexion = rfft(flexion_torques)
            xf_flexion = rfftfreq(N_flexion, dt_flexion)
            flexion_dominant_freq = xf_flexion[np.argmax(np.abs(yf_flexion[1:])) + 1]
            
            X_flexion = np.column_stack((
                np.sin(2 * np.pi * flexion_dominant_freq * flexion_times),
                np.cos(2 * np.pi * flexion_dominant_freq * flexion_times)
            ))
            flexion_reg = LinearRegression()
            flexion_reg.fit(X_flexion, flexion_torques)
            
            # Fit sinusoid to filtered extension MVC data
            extension_times_raw = filtered_extension_df["time"].values
            extension_torques = filtered_extension_df["torque"].values
            
            # Normalize time to start at 0 for better phase consistency
            extension_times = extension_times_raw - extension_times_raw[0]
            
            dt_extension = np.mean(np.diff(extension_times))
            N_extension = len(extension_times)
            yf_extension = rfft(extension_torques)
            xf_extension = rfftfreq(N_extension, dt_extension)
            extension_dominant_freq = xf_extension[np.argmax(np.abs(yf_extension[1:])) + 1]
            
            X_extension = np.column_stack((
                np.sin(2 * np.pi * extension_dominant_freq * extension_times),
                np.cos(2 * np.pi * extension_dominant_freq * extension_times)
            ))
            extension_reg = LinearRegression()
            extension_reg.fit(X_extension, extension_torques)
            
            # Store the fitted parameters for real-time use
            self.flexion_freq = flexion_dominant_freq
            self.flexion_reg = flexion_reg
            self.extension_freq = extension_dominant_freq
            self.extension_reg = extension_reg
            self.mvc_fitted = True
            
            print(f"MVC sinusoidal fits calculated (velocity-filtered):")
            print(f"Flexion - Frequency: {flexion_dominant_freq:.3f} Hz, Period: {1/flexion_dominant_freq:.2f} seconds")
            print(f"Extension - Frequency: {extension_dominant_freq:.3f} Hz, Period: {1/extension_dominant_freq:.2f} seconds")
            
            # Calculate and print actual sampling rates
            print(f"Actual sampling analysis:")
            print(f"Flexion MVC - Mean interval: {dt_flexion:.4f}s, Effective rate: {1/dt_flexion:.1f} Hz")
            print(f"Extension MVC - Mean interval: {dt_extension:.4f}s, Effective rate: {1/dt_extension:.1f} Hz")
            
            # Calculate interval statistics
            flexion_intervals = np.diff(flexion_times)
            extension_intervals = np.diff(extension_times)
            print(f"Flexion MVC - Interval std: {np.std(flexion_intervals)*1000:.2f}ms")
            print(f"Extension MVC - Interval std: {np.std(extension_intervals)*1000:.2f}ms")

    def calculate_flexion_baseline_fit(self):
        """Calculate and store sinusoidal fit parameters for flexion baseline data"""
        if hasattr(self, "df_baseline") and not self.df_baseline.empty:
            # Filter baseline data by velocity
            print("Applying velocity filtering to flexion baseline data...")
            filtered_df = self.filter_by_velocity(self.df_baseline, velocity_threshold=0.03, data_type="Flexion Baseline", phase_type="flexion")
            
            if len(filtered_df) == 0:
                print("Warning: Not enough data points remain after velocity filtering for flexion baseline. Using original data.")
                filtered_df = self.df_baseline
            
            baseline_times_raw = filtered_df["time"].values
            baseline_torques = filtered_df["torque"].values
            
            # Normalize time to start at 0 for better phase consistency
            baseline_times = baseline_times_raw - baseline_times_raw[0]
            
            dt_baseline = np.mean(np.diff(baseline_times))
            N_baseline = len(baseline_times)
            yf_baseline = rfft(baseline_torques)
            xf_baseline = rfftfreq(N_baseline, dt_baseline)
            baseline_dominant_freq = xf_baseline[np.argmax(np.abs(yf_baseline[1:])) + 1]
            
            X_baseline = np.column_stack((
                np.sin(2 * np.pi * baseline_dominant_freq * baseline_times),
                np.cos(2 * np.pi * baseline_dominant_freq * baseline_times)
            ))
            baseline_reg = LinearRegression()
            baseline_reg.fit(X_baseline, baseline_torques)
            
            # Store the fitted parameters
            self.flexion_baseline_freq = baseline_dominant_freq
            self.flexion_baseline_reg = baseline_reg
            self.flexion_baseline_fitted = True
            
            print(f"Flexion Baseline sinusoidal fit calculated (velocity-filtered):")
            print(f"Frequency: {baseline_dominant_freq:.3f} Hz, Period: {1/baseline_dominant_freq:.2f} seconds")
            print(f"Coefficients: {baseline_reg.coef_}, Intercept: {baseline_reg.intercept_:.3f}")

    def calculate_extension_baseline_fit(self):
        """Calculate and store sinusoidal fit parameters for extension baseline data"""
        if hasattr(self, "df_ext_baseline") and not self.df_ext_baseline.empty:
            # Filter baseline data by velocity
            print("Applying velocity filtering to extension baseline data...")
            filtered_df = self.filter_by_velocity(self.df_ext_baseline, velocity_threshold=0.03, data_type="Extension Baseline", phase_type="extension")
            
            if len(filtered_df) == 0:
                print("Warning: Not enough data points remain after velocity filtering for extension baseline. Using original data.")
                filtered_df = self.df_ext_baseline
            
            baseline_times_raw = filtered_df["time"].values
            baseline_torques = filtered_df["torque"].values
            
            # Normalize time to start at 0 for better phase consistency
            baseline_times = baseline_times_raw - baseline_times_raw[0]
            
            dt_baseline = np.mean(np.diff(baseline_times))
            N_baseline = len(baseline_times)
            yf_baseline = rfft(baseline_torques)
            xf_baseline = rfftfreq(N_baseline, dt_baseline)
            baseline_dominant_freq = xf_baseline[np.argmax(np.abs(yf_baseline[1:])) + 1]
            
            X_baseline = np.column_stack((
                np.sin(2 * np.pi * baseline_dominant_freq * baseline_times),
                np.cos(2 * np.pi * baseline_dominant_freq * baseline_times)
            ))
            baseline_reg = LinearRegression()
            baseline_reg.fit(X_baseline, baseline_torques)
            
            # Store the fitted parameters
            self.extension_baseline_freq = baseline_dominant_freq
            self.extension_baseline_reg = baseline_reg
            self.extension_baseline_fitted = True
            
            print(f"Extension Baseline sinusoidal fit calculated (velocity-filtered):")
            print(f"Frequency: {baseline_dominant_freq:.3f} Hz, Period: {1/baseline_dominant_freq:.2f} seconds")
            print(f"Coefficients: {baseline_reg.coef_}, Intercept: {baseline_reg.intercept_:.3f}")

    def plot_baseline_analysis(self):
        """Plot flexion and extension baseline data with their sinusoidal fits"""
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
            # Use filtered and normalized data for plotting
            filtered_flexion_df = self.filter_by_velocity(self.df_baseline, velocity_threshold=0.03, data_type="Flexion Baseline", phase_type="flexion")
            if len(filtered_flexion_df) == 0:
                filtered_flexion_df = self.df_baseline
            
            flexion_times_raw = filtered_flexion_df["time"].values
            flexion_torques = filtered_flexion_df["torque"].values
            
            # Normalize time to start at 0 (same as used in fitting)
            flexion_times = flexion_times_raw - flexion_times_raw[0]
            
            # Calculate fit using normalized time
            X_flexion = np.column_stack((
                np.sin(2 * np.pi * self.flexion_baseline_freq * flexion_times),
                np.cos(2 * np.pi * self.flexion_baseline_freq * flexion_times)
            ))
            flexion_fit = self.flexion_baseline_reg.predict(X_flexion)
            
            # Calculate fit quality metrics
            flexion_r2 = self.flexion_baseline_reg.score(X_flexion, flexion_torques)
            flexion_rmse = np.sqrt(np.mean((flexion_torques - flexion_fit)**2))
            
            plt.subplot(1, plot_count, current_plot)
            plt.plot(flexion_times, flexion_torques, 'b-', linewidth=1, alpha=0.7, label='Filtered Flexion Baseline Data')
            plt.plot(flexion_times, flexion_fit, 'r--', linewidth=2, label=f'Sinusoidal Fit (f={self.flexion_baseline_freq:.3f} Hz)')
            plt.xlabel('Normalized Time (s)')
            plt.ylabel('Torque')
            plt.title(f'Filtered & Normalized Flexion Baseline Data and Fit\nR²={flexion_r2:.3f}, RMSE={flexion_rmse:.4f}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            current_plot += 1
            
        # Plot extension baseline if available
        if has_extension:
            # Use filtered and normalized data for plotting
            filtered_extension_df = self.filter_by_velocity(self.df_ext_baseline, velocity_threshold=0.03, data_type="Extension Baseline", phase_type="extension")
            if len(filtered_extension_df) == 0:
                filtered_extension_df = self.df_ext_baseline
            
            extension_times_raw = filtered_extension_df["time"].values
            extension_torques = filtered_extension_df["torque"].values
            
            # Normalize time to start at 0 (same as used in fitting)
            extension_times = extension_times_raw - extension_times_raw[0]
            
            # Calculate fit using normalized time
            X_extension = np.column_stack((
                np.sin(2 * np.pi * self.extension_baseline_freq * extension_times),
                np.cos(2 * np.pi * self.extension_baseline_freq * extension_times)
            ))
            extension_fit = self.extension_baseline_reg.predict(X_extension)
            
            # Calculate fit quality metrics
            extension_r2 = self.extension_baseline_reg.score(X_extension, extension_torques)
            extension_rmse = np.sqrt(np.mean((extension_torques - extension_fit)**2))
            
            plt.subplot(1, plot_count, current_plot)
            plt.plot(extension_times, extension_torques, 'b-', linewidth=1, alpha=0.7, label='Filtered Extension Baseline Data')
            plt.plot(extension_times, extension_fit, 'r--', linewidth=2, label=f'Sinusoidal Fit (f={self.extension_baseline_freq:.3f} Hz)')
            plt.xlabel('Normalized Time (s)')
            plt.ylabel('Torque')
            plt.title(f'Filtered & Normalized Extension Baseline Data and Fit\nR²={extension_r2:.3f}, RMSE={extension_rmse:.4f}')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print analysis information
        print(f"\nBaseline Analysis Results:")
        if has_flexion:
            print(f"Flexion Baseline - Frequency: {self.flexion_baseline_freq:.3f} Hz, Period: {1/self.flexion_baseline_freq:.2f} seconds")
            print(f"Flexion Baseline - R²: {flexion_r2:.3f}, RMSE: {flexion_rmse:.4f}")
            print(f"Flexion Baseline - Coefficients: {self.flexion_baseline_reg.coef_}, Intercept: {self.flexion_baseline_reg.intercept_:.3f}")
        if has_extension:
            print(f"Extension Baseline - Frequency: {self.extension_baseline_freq:.3f} Hz, Period: {1/self.extension_baseline_freq:.2f} seconds")
            print(f"Extension Baseline - R²: {extension_r2:.3f}, RMSE: {extension_rmse:.4f}")
            print(f"Extension Baseline - Coefficients: {self.extension_baseline_reg.coef_}, Intercept: {self.extension_baseline_reg.intercept_:.3f}")

    def get_baseline_corrected_value(self, time_val, torque_value=None, is_flexion_phase=None):
        """Calculate baseline sinusoidal value for subtraction at given time
        
        Args:
            time_val: Time point for calculation
            torque_value: If provided, use torque sign to determine baseline (positive=flexion, negative=extension)
            is_flexion_phase: If torque_value not provided, use this phase indicator
        """
        # Determine which baseline to use based on torque sign if available
        if torque_value is not None:
            use_flexion_baseline = torque_value >= 0  # Positive torque = flexion, negative = extension
        elif is_flexion_phase is not None:
            use_flexion_baseline = is_flexion_phase
        else:
            return 0.0  # No way to determine which baseline to use
        
        if use_flexion_baseline and self.flexion_baseline_fitted:
            X_baseline = np.array([[
                np.sin(2 * np.pi * self.flexion_baseline_freq * time_val),
                np.cos(2 * np.pi * self.flexion_baseline_freq * time_val)
            ]])
            return self.flexion_baseline_reg.predict(X_baseline)[0]
        elif not use_flexion_baseline and self.extension_baseline_fitted:
            X_baseline = np.array([[
                np.sin(2 * np.pi * self.extension_baseline_freq * time_val),
                np.cos(2 * np.pi * self.extension_baseline_freq * time_val)
            ]])
            return self.extension_baseline_reg.predict(X_baseline)[0]
        else:
            return 0.0  # No baseline correction available

    def plot_combined_mvc_analysis(self):
        """Plot individual MVC data with separate sinusoidal fits and their combination"""
        if (hasattr(self, "df_flexionMVC") and not self.df_flexionMVC.empty and 
            hasattr(self, "df_extensionMVC") and not self.df_extensionMVC.empty and
            self.mvc_fitted):
            
            # Get filtered and normalized flexion MVC data (same as used in fitting)
            filtered_flexion_df = self.filter_by_velocity(self.df_flexionMVC, velocity_threshold=0.03, data_type="Flexion MVC", phase_type="flexion")
            if len(filtered_flexion_df) == 0:
                filtered_flexion_df = self.df_flexionMVC
            
            flexion_times_raw = filtered_flexion_df["time"].values
            flexion_torques = filtered_flexion_df["torque"].values
            
            # Normalize time to start at 0 (same as used in fitting)
            flexion_times = flexion_times_raw - flexion_times_raw[0]
            
            X_flexion = np.column_stack((
                np.sin(2 * np.pi * self.flexion_freq * flexion_times),
                np.cos(2 * np.pi * self.flexion_freq * flexion_times)
            ))
            flexion_fit = self.flexion_reg.predict(X_flexion)
            
            # Get filtered and normalized extension MVC data (same as used in fitting)
            filtered_extension_df = self.filter_by_velocity(self.df_extensionMVC, velocity_threshold=0.03, data_type="Extension MVC", phase_type="extension")
            if len(filtered_extension_df) == 0:
                filtered_extension_df = self.df_extensionMVC
            
            extension_times_raw = filtered_extension_df["time"].values
            extension_torques = filtered_extension_df["torque"].values
            
            # Normalize time to start at 0 (same as used in fitting)
            extension_times = extension_times_raw - extension_times_raw[0]
            
            X_extension = np.column_stack((
                np.sin(2 * np.pi * self.extension_freq * extension_times),
                np.cos(2 * np.pi * self.extension_freq * extension_times)
            ))
            extension_fit = self.extension_reg.predict(X_extension)
            
            # Create combined timeline for plotting - use the last flexion time as offset
            time_offset = flexion_times[-1] if len(flexion_times) > 0 else 0
            shifted_extension_times = extension_times + time_offset
            
            # Combine filtered data
            combined_times = np.concatenate([flexion_times, shifted_extension_times])
            combined_torques = np.concatenate([flexion_torques, extension_torques])
            
            # Create combined fitted sinusoids
            shifted_extension_fit = extension_fit  # Extension fit doesn't need time shifting, just positioning
            combined_fits = np.concatenate([flexion_fit, shifted_extension_fit])
            
            # Calculate timing statistics for normalized timestamps
            flexion_intervals = np.diff(flexion_times)
            extension_intervals = np.diff(extension_times)
            
            # Create the matplotlib plot
            plt.figure(figsize=(15, 12))
            
            # Plot 1: Individual MVC data with separate fits
            plt.subplot(4, 1, 1)
            plt.plot(flexion_times, flexion_torques, 'b-', linewidth=1, alpha=0.7, label='Filtered Flexion MVC Data')
            plt.plot(flexion_times, flexion_fit, 'r--', linewidth=2, label=f'Flexion Fit (f={self.flexion_freq:.3f} Hz)')
            plt.plot(shifted_extension_times, extension_torques, 'g-', linewidth=1, alpha=0.7, label='Filtered Extension MVC Data')
            plt.plot(shifted_extension_times, extension_fit, 'm--', linewidth=2, label=f'Extension Fit (f={self.extension_freq:.3f} Hz)')
            plt.axvline(x=time_offset, color='k', linestyle='--', alpha=0.5, label='Flexion/Extension Boundary')
            plt.xlabel('Normalized Time (s)')
            plt.ylabel('Torque')
            plt.title('Individual Filtered & Normalized MVC Data with Separate Sinusoidal Fits')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 2: Combined filtered data
            plt.subplot(4, 1, 2)
            plt.plot(combined_times, combined_torques, 'b-', linewidth=1, alpha=0.7, label='Combined Filtered MVC Data')
            plt.axvline(x=time_offset, color='k', linestyle='--', alpha=0.5, label='Flexion/Extension Boundary')
            plt.xlabel('Normalized Time (s)')
            plt.ylabel('Torque')
            plt.title('Combined Filtered Flexion + Extension MVC Data (Normalized Time)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 3: Combined data with combined sinusoidal fits
            plt.subplot(4, 1, 3)
            plt.plot(combined_times, combined_torques, 'b-', linewidth=1, alpha=0.7, label='Combined Filtered MVC Data')
            plt.plot(combined_times, combined_fits, 'r--', linewidth=3, label='Combined Sinusoidal Fits')
            plt.axvline(x=time_offset, color='k', linestyle='--', alpha=0.5, label='Flexion/Extension Boundary')
            plt.xlabel('Normalized Time (s)')
            plt.ylabel('Torque')
            plt.title('Combined Filtered MVC Data with Combined Sinusoidal Fits (Normalized Time)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 4: Timing intervals histogram for filtered data
            plt.subplot(4, 1, 4)
            plt.hist(flexion_intervals * 1000, bins=20, alpha=0.7, label=f'Filtered Flexion Intervals (mean: {np.mean(flexion_intervals)*1000:.1f}ms)', color='blue')
            plt.hist(extension_intervals * 1000, bins=20, alpha=0.7, label=f'Filtered Extension Intervals (mean: {np.mean(extension_intervals)*1000:.1f}ms)', color='green')
            plt.axvline(x=20, color='red', linestyle='--', label='Target 20ms (50Hz)')
            plt.xlabel('Sampling Interval (ms)')
            plt.ylabel('Count')
            plt.title('Filtered Data Sampling Interval Distribution')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Print analysis information
            print(f"MVC Analysis Results (Filtered & Normalized Data):")
            print(f"Flexion - Frequency: {self.flexion_freq:.3f} Hz, Period: {1/self.flexion_freq:.2f} seconds")
            print(f"Flexion - Coefficients: {self.flexion_reg.coef_}, Intercept: {self.flexion_reg.intercept_:.3f}")
            print(f"Extension - Frequency: {self.extension_freq:.3f} Hz, Period: {1/self.extension_freq:.2f} seconds")
            print(f"Extension - Coefficients: {self.extension_reg.coef_}, Intercept: {self.extension_reg.intercept_:.3f}")
            print(f"Filtered data timing statistics:")
            print(f"Flexion - Mean interval: {np.mean(flexion_intervals)*1000:.2f}ms, Std: {np.std(flexion_intervals)*1000:.2f}ms")
            print(f"Extension - Mean interval: {np.mean(extension_intervals)*1000:.2f}ms, Std: {np.std(extension_intervals)*1000:.2f}ms")
            print(f"Filtered data counts:")
            print(f"Flexion - Original: {len(self.df_flexionMVC)}, Filtered: {len(filtered_flexion_df)} ({len(filtered_flexion_df)/len(self.df_flexionMVC)*100:.1f}%)")
            print(f"Extension - Original: {len(self.df_extensionMVC)}, Filtered: {len(filtered_extension_df)} ({len(filtered_extension_df)/len(self.df_extensionMVC)*100:.1f}%)")

    def plot_torque_correction_comparison(self):
        """Plot comparison between uncorrected and corrected torque data"""
        if hasattr(self, "df_task") and not self.df_task.empty:
            times = self.df_task["time"].values
            corrected_torques = self.df_task["torque"].values
            uncorrected_torques = self.df_task["uncorrected_torque"].values
            
            # Calculate baseline corrections applied
            baseline_corrections = uncorrected_torques - corrected_torques
            
            # Create the matplotlib plot
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Uncorrected vs Corrected Torque
            plt.subplot(3, 1, 1)
            plt.plot(times, uncorrected_torques, 'b-', linewidth=1, alpha=0.7, label='Uncorrected Torque')
            plt.plot(times, corrected_torques, 'r-', linewidth=1, alpha=0.8, label='Corrected Torque')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Zero Line')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque')
            plt.title('Torque Comparison: Uncorrected vs Baseline-Corrected')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 2: Baseline Correction Applied
            plt.subplot(3, 1, 2)
            plt.plot(times, baseline_corrections, 'g-', linewidth=1, alpha=0.7, label='Baseline Correction Applied')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Zero Line')
            
            # Color regions based on flexion/extension
            positive_mask = uncorrected_torques >= 0
            negative_mask = uncorrected_torques < 0
            
            if np.any(positive_mask):
                plt.fill_between(times, 0, baseline_corrections, where=positive_mask[range(len(times))], 
                               alpha=0.3, color='blue', label='Flexion Regions (Positive Torque)')
            if np.any(negative_mask):
                plt.fill_between(times, 0, baseline_corrections, where=negative_mask[range(len(times))], 
                               alpha=0.3, color='orange', label='Extension Regions (Negative Torque)')
            
            plt.xlabel('Time (s)')
            plt.ylabel('Baseline Correction (Uncorrected - Corrected)')
            plt.title('Baseline Correction Applied Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 3: MVC Target vs Corrected Torque
            plt.subplot(3, 1, 3)
            plt.plot(times, corrected_torques, 'b-', linewidth=2, alpha=0.8, label='Corrected Torque')
            
            # Generate MVC target line if MVC data is available
            if (hasattr(self, "df_flexionMVC") and not self.df_flexionMVC.empty and 
                hasattr(self, "df_extensionMVC") and not self.df_extensionMVC.empty and
                self.mvc_fitted):
                
                # Calculate MVC target for each time point
                mvc_targets = []
                
                for t in times:
                    # Calculate which phase we're in based on the cycle period
                    flexion_times = self.df_flexionMVC["time"].values
                    time_offset = flexion_times[-1] if len(flexion_times) > 0 else 0
                    cycle_period = 2 * time_offset
                    
                    cycle_time = t % cycle_period
                    
                    if cycle_time <= time_offset:
                        # Flexion phase - FIX: Properly reshape the array
                        X_flexion = np.array([[
                            np.sin(2 * np.pi * self.flexion_freq * cycle_time),
                            np.cos(2 * np.pi * self.flexion_freq * cycle_time)
                        ]])
                        mvc_value = self.flexion_reg.predict(X_flexion)[0]
                        
                        # Apply baseline correction if available
                        if self.flexion_baseline_fitted:
                            baseline_correction = self.get_baseline_corrected_value(t, is_flexion_phase=True)
                            mvc_value -= baseline_correction
                    else:
                        # Extension phase - FIX: Properly reshape the array
                        extension_phase_time = cycle_time - time_offset
                        X_extension = np.array([[
                            np.sin(2 * np.pi * self.extension_freq * extension_phase_time),
                            np.cos(2 * np.pi * self.extension_freq * extension_phase_time)
                        ]])
                        mvc_value = self.extension_reg.predict(X_extension)[0]
                        
                        # Apply baseline correction if available
                        if self.extension_baseline_fitted:
                            baseline_correction = self.get_baseline_corrected_value(t, is_flexion_phase=False)
                            mvc_value -= baseline_correction
                    
                    mvc_targets.append(mvc_value)
                
                plt.plot(times, mvc_targets, 'r--', linewidth=2, alpha=0.9, label='MVC Target (Baseline Corrected)')
            
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Zero Line')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque')
            plt.title('Corrected Torque vs MVC Target')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Print correction statistics
            difference = np.abs(baseline_corrections)
            print(f"\nBaseline Correction Analysis:")
            print(f"Mean absolute correction: {np.mean(difference):.4f}")
            print(f"Standard deviation of corrections: {np.std(difference):.4f}")
            print(f"Maximum correction applied: {np.max(difference):.4f}")
            print(f"Percentage of data corrected: {(np.sum(difference > 0.001) / len(difference)) * 100:.1f}%")
            
            # Statistics by torque direction
            positive_mask = uncorrected_torques >= 0
            negative_mask = uncorrected_torques < 0
            
            flexion_corrections = baseline_corrections[positive_mask]
            extension_corrections = baseline_corrections[negative_mask]
            
            if len(flexion_corrections) > 0:
                print(f"Flexion corrections - Mean: {np.mean(flexion_corrections):.4f}, Std: {np.std(flexion_corrections):.4f}")
            if len(extension_corrections) > 0:
                print(f"Extension corrections - Mean: {np.mean(extension_corrections):.4f}, Std: {np.std(extension_corrections):.4f}")
                
            # Print MVC comparison statistics if available
            if 'mvc_targets' in locals():
                mvc_targets = np.array(mvc_targets)
                tracking_error = corrected_torques - mvc_targets
                print(f"\nMVC Tracking Analysis:")
                print(f"Mean tracking error: {np.mean(tracking_error):.4f}")
                print(f"RMS tracking error: {np.sqrt(np.mean(tracking_error**2)):.4f}")
                print(f"Max absolute tracking error: {np.max(np.abs(tracking_error)):.4f}")
        else:
            print("No task data available for torque correction comparison.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
