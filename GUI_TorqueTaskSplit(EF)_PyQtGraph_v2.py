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
import matplotlib.pyplot as plt

def sin_func(t, A, f, phi, offset):
    return A * np.sin(2 * np.pi * f * t + phi) + offset

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time DAQ Plotting v2 - Real Timestamps")
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
        self.df_task = pd.DataFrame(columns=["time", "position", "torque", "uncorrected_torque", "voltage"])
        self.df_baseline = pd.DataFrame(columns=["time", "torque"])
        self.df_flexionMVC = pd.DataFrame(columns=["time", "torque"])
        self.df_ext_baseline = pd.DataFrame(columns=["time", "torque"])
        self.df_extensionMVC = pd.DataFrame(columns=["time", "torque"])

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
        
        # Setup plot
        self.plot_widget.clear()
        self.plot_widget.setTitle("Setting Flexion Baseline")
        self.curve = self.plot_widget.plot(pen='b', name='Baseline Torque')
        self.dot = pg.ScatterPlotItem(size=12, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(30)  # Same 10ms interval as task

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
        
        # Setup plot
        self.plot_widget.clear()
        self.plot_widget.setTitle("Setting Extension Baseline")
        self.curve = self.plot_widget.plot(pen='b', name='Baseline Torque')
        self.dot = pg.ScatterPlotItem(size=12, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(30)  # Same 10ms interval as task

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
        
        # Setup plot
        self.plot_widget.clear()
        self.plot_widget.setTitle("Setting Flexion MVC")
        self.curve = self.plot_widget.plot(pen='b', name='MVC Torque')
        self.dot = pg.ScatterPlotItem(size=12, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(30)  # Same 8ms interval as task

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
        
        # Setup plot
        self.plot_widget.clear()
        self.plot_widget.setTitle("Setting Extension MVC")
        self.curve = self.plot_widget.plot(pen='b', name='MVC Torque')
        self.dot = pg.ScatterPlotItem(size=12, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(30)  # Same 8ms interval as task

    def start_task(self):
        self.acquiring = True
        self.df_task = pd.DataFrame(columns=["time", "position", "torque", "uncorrected_torque", "voltage"])
        self.start_time = None  # Will be set on first data point
        self.collection_mode = None  # Reset collection mode for task
        
        # Reset sampling rate tracking
        self.sampling_rates = []
        self.last_sample_time = None
        
        # Setup plot for task
        self.plot_widget.clear()
        self.plot_widget.setTitle("Real-Time Torque Plot")
        self.curve = self.plot_widget.plot(pen='b', name='Measured Torque')
        self.dot = pg.ScatterPlotItem(size=12, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        self.target_curve = self.plot_widget.plot(pen=pg.mkPen('r', style=QtCore.Qt.DashLine), name='Target')
        self.flexion_curve = self.plot_widget.plot(pen=pg.mkPen('r', width=4, style=QtCore.Qt.DotLine), name='Combined MVC')
        
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
        self.timer.start(1)  # update every 10 ms

    def update_plot(self):
        if not self.acquiring:
            return
        
        # Set start time on first data point
        if self.start_time is None:
            self.start_time = time.time()
            
        value = self.input_task.read(number_of_samples_per_channel=10)
        
        # Use real wall-clock time for actual sampling timestamps
        t = time.time() - self.start_time
        
        # Calculate instantaneous sampling rate
        current_sample_time = time.time()
        if self.last_sample_time is not None:
            interval = current_sample_time - self.last_sample_time
            instantaneous_rate = 1.0 / interval if interval > 0 else 0
            self.sampling_rates.append(instantaneous_rate)
        self.last_sample_time = current_sample_time
        
        position = np.mean(value[0])
        torque = np.mean(value[1])
        voltage = np.mean(value[2])
        
        # Handle different collection modes
        if self.collection_mode is not None:
            # This is baseline or MVC collection
            self.collection_data.append([t, torque])
            
            # Check if collection duration is complete
            if t >= self.collection_duration:
                self.finish_collection()
                return
                
            # Plot the collection data
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
        else:
            # This is the main task
            ao0 = 9  # Example: output value (change as needed)
            self.output_task.write(ao0)
            
            # Apply baseline correction to torque measurement based on torque sign
            # Positive torque = flexion (use flexion baseline)
            # Negative torque = extension (use extension baseline)
            baseline_correction = self.get_baseline_corrected_value(t, torque_value=torque)
            corrected_torque = torque - baseline_correction
            
            # --- Append to DataFrame with both corrected and uncorrected torque ---
            new_row = pd.DataFrame([[t, position, corrected_torque, torque, voltage]], columns=self.df_task.columns)
            if self.df_task.empty:
                self.df_task = new_row
            else:
                self.df_task = pd.concat([self.df_task, new_row], ignore_index=True)

            # Moving window for plotting - but we'll override this for centered view during task
            if len(self.df_task) < self.window_size:
                x_window = self.df_task["time"].values
                y_window = self.df_task["torque"].values
            else:
                x_window = self.df_task["time"].values[-self.window_size:]
                y_window = self.df_task["torque"].values[-self.window_size:]

            # Plot sinusoid of best fit for combined flexion + extension MVC if both are available
            if (hasattr(self, "df_flexionMVC") and not self.df_flexionMVC.empty and 
                hasattr(self, "df_extensionMVC") and not self.df_extensionMVC.empty and
                self.mvc_fitted):
                
                # Use pre-calculated sinusoidal fit parameters
                # Create centered view with look-ahead
                current_time = t
                window_start = current_time - (self.look_ahead_time * self.center_position)
                window_end = current_time + (self.look_ahead_time * (1 - self.center_position))
                
                # Generate time points for the window (reduced resolution for performance)
                time_points = np.linspace(window_start, window_end, 50)  # Reduced from 200 to 50
                
                # Pre-calculate sinusoidal components for better performance
                flexion_times = self.df_flexionMVC["time"].values
                time_offset = flexion_times[-1] if len(flexion_times) > 0 else 0
                cycle_period = 2 * time_offset
                
                # Vectorized calculation instead of loop
                cycle_times = time_points % cycle_period
                flexion_mask = cycle_times <= time_offset
                extension_mask = ~flexion_mask
                
                # Calculate flexion phase values
                flexion_phase_times = cycle_times[flexion_mask]
                extension_phase_times = cycle_times[extension_mask] - time_offset
                
                y_combined = np.zeros_like(time_points)
                
                if len(flexion_phase_times) > 0:
                    X_flexion = np.column_stack((
                        np.sin(2 * np.pi * self.flexion_freq * flexion_phase_times),
                        np.cos(2 * np.pi * self.flexion_freq * flexion_phase_times)
                    ))
                    flexion_mvc_values = self.flexion_reg.predict(X_flexion)
                    
                    # Subtract flexion baseline correction from flexion MVC values
                    if self.flexion_baseline_fitted:
                        baseline_corrections = np.array([
                            self.get_baseline_corrected_value(t, is_flexion_phase=True) 
                            for t in flexion_phase_times
                        ])
                        flexion_mvc_values -= baseline_corrections
                    
                    y_combined[flexion_mask] = flexion_mvc_values
                
                if len(extension_phase_times) > 0:
                    X_extension = np.column_stack((
                        np.sin(2 * np.pi * self.extension_freq * extension_phase_times),
                        np.cos(2 * np.pi * self.extension_freq * extension_phase_times)
                    ))
                    extension_mvc_values = self.extension_reg.predict(X_extension)
                    
                    # Subtract extension baseline correction from extension MVC values
                    if self.extension_baseline_fitted:
                        baseline_corrections = np.array([
                            self.get_baseline_corrected_value(t, is_flexion_phase=False) 
                            for t in extension_phase_times
                        ])
                        extension_mvc_values -= baseline_corrections
                    
                    y_combined[extension_mask] = extension_mvc_values
                
                # Plot the look-ahead combined sinusoid
                self.flexion_curve.setData(time_points, y_combined)
                
                # Update the main curve display for centered view
                if len(self.df_task) > 1:
                    # Filter data within the current window
                    task_times = self.df_task["time"].values
                    task_torques = self.df_task["torque"].values
                    
                    # Get data points within the current window
                    window_mask = (task_times >= window_start) & (task_times <= window_end)
                    windowed_times = task_times[window_mask]
                    windowed_torques = task_torques[window_mask]
                    
                    if len(windowed_times) > 0:
                        self.curve.setData(windowed_times, windowed_torques)
                        # Keep the dot at current position
                        self.dot.setData([current_time], [task_torques[-1]])
                        # Set x-range to show the look-ahead window
                        self.plot_widget.setXRange(window_start, window_end)
                    else:
                        # Fallback to normal window if no data in range
                        self.curve.setData(x_window, y_window)
                        self.dot.setData([x_window[-1]], [y_window[-1]])
                        self.plot_widget.setXRange(x_window[0], x_window[-1])
                else:
                    # Fallback for early data
                    self.curve.setData(x_window, y_window)
                    self.dot.setData([x_window[-1]], [y_window[-1]])
                    self.plot_widget.setXRange(x_window[0], x_window[-1])
            else:
                # No MVC data available, use normal plotting
                self.curve.setData(x_window, y_window)
                self.dot.setData([x_window[-1]], [y_window[-1]])
                self.plot_widget.setXRange(x_window[0], x_window[-1])

            # Stop after task_total_time seconds
            if t > self.task_total_time:
                self.timer.stop()
                # Set output to 0 before stopping
                self.output_task.write(0)
                self.stopDAQ()
                self.acquiring = False
                
                # Display sampling rate analysis
                self.display_sampling_rate_analysis()
                
                # Plot torque correction comparison
                self.plot_torque_correction_comparison()
                
                # --- Save all data to CSV ---
                self.df_task.to_csv(f"DAQ_Task_Trial{self.trialNum}_v2.csv", index=False)
                print("Task complete. Data saved.")
                self.trialNum += 1

    def finish_collection(self):
        """Finish baseline or MVC data collection"""
        self.timer.stop()
        self.stopDAQ()
        self.acquiring = False
        
        # Save data to appropriate DataFrame
        if self.collection_mode == 'flexion_baseline':
            self.df_baseline = pd.DataFrame(self.collection_data, columns=["time", "torque"])
            print("Flexion Baseline set.")
            # Save baseline data to CSV
            self.df_baseline.to_csv("FlexionBaseline_Data_v2.csv", index=False)
            print("Flexion baseline data saved to FlexionBaseline_Data_v2.csv")
            # Calculate baseline sinusoidal fit
            self.calculate_flexion_baseline_fit()
        elif self.collection_mode == 'extension_baseline':
            self.df_ext_baseline = pd.DataFrame(self.collection_data, columns=["time", "torque"])
            print("Extension Baseline set.")
            # Save extension baseline data to CSV
            self.df_ext_baseline.to_csv("ExtensionBaseline_Data_v2.csv", index=False)
            print("Extension baseline data saved to ExtensionBaseline_Data_v2.csv")
            # Calculate baseline sinusoidal fit
            self.calculate_extension_baseline_fit()
        elif self.collection_mode == 'flexion_mvc':
            self.df_flexionMVC = pd.DataFrame(self.collection_data, columns=["time", "torque"])
            print("Flexion MVC set.")
            # Save flexion MVC data to CSV
            self.df_flexionMVC.to_csv("FlexionMVC_Data_v2.csv", index=False)
            print("Flexion MVC data saved to FlexionMVC_Data_v2.csv")
        elif self.collection_mode == 'extension_mvc':
            self.df_extensionMVC = pd.DataFrame(self.collection_data, columns=["time", "torque"])
            print("Extension MVC set.")
            # Save extension MVC data to CSV
            self.df_extensionMVC.to_csv("ExtensionMVC_Data_v2.csv", index=False)
            print("Extension MVC data saved to ExtensionMVC_Data_v2.csv")
            # Check if both MVCs are now available and calculate sinusoidal fit
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
            sampling_data.to_csv(f"SamplingRate_Trial{self.trialNum}_v2.csv", index=False)
            print(f"Sampling rate data saved to SamplingRate_Trial{self.trialNum}_v2.csv")

    def calculate_mvc_fit(self):
        """Calculate and store sinusoidal fit parameters for flexion and extension MVC data separately"""
        if (hasattr(self, "df_flexionMVC") and not self.df_flexionMVC.empty and 
            hasattr(self, "df_extensionMVC") and not self.df_extensionMVC.empty):
            
            # Fit sinusoid to flexion MVC data
            flexion_times = self.df_flexionMVC["time"].values
            flexion_torques = self.df_flexionMVC["torque"].values
            
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
            
            # Fit sinusoid to extension MVC data
            extension_times = self.df_extensionMVC["time"].values
            extension_torques = self.df_extensionMVC["torque"].values
            
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
            
            print(f"MVC sinusoidal fits calculated:")
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
            baseline_times = self.df_baseline["time"].values
            baseline_torques = self.df_baseline["torque"].values
            
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
            
            print(f"Flexion Baseline sinusoidal fit calculated:")
            print(f"Frequency: {baseline_dominant_freq:.3f} Hz, Period: {1/baseline_dominant_freq:.2f} seconds")
            print(f"Coefficients: {baseline_reg.coef_}, Intercept: {baseline_reg.intercept_:.3f}")

    def calculate_extension_baseline_fit(self):
        """Calculate and store sinusoidal fit parameters for extension baseline data"""
        if hasattr(self, "df_ext_baseline") and not self.df_ext_baseline.empty:
            baseline_times = self.df_ext_baseline["time"].values
            baseline_torques = self.df_ext_baseline["torque"].values
            
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
            
            print(f"Extension Baseline sinusoidal fit calculated:")
            print(f"Frequency: {baseline_dominant_freq:.3f} Hz, Period: {1/baseline_dominant_freq:.2f} seconds")
            print(f"Coefficients: {baseline_reg.coef_}, Intercept: {baseline_reg.intercept_:.3f}")

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
            
            # Get flexion MVC data and fit
            flexion_times = self.df_flexionMVC["time"].values
            flexion_torques = self.df_flexionMVC["torque"].values
            
            X_flexion = np.column_stack((
                np.sin(2 * np.pi * self.flexion_freq * flexion_times),
                np.cos(2 * np.pi * self.flexion_freq * flexion_times)
            ))
            flexion_fit = self.flexion_reg.predict(X_flexion)
            
            # Get extension MVC data and fit
            extension_times = self.df_extensionMVC["time"].values
            extension_torques = self.df_extensionMVC["torque"].values
            
            X_extension = np.column_stack((
                np.sin(2 * np.pi * self.extension_freq * extension_times),
                np.cos(2 * np.pi * self.extension_freq * extension_times)
            ))
            extension_fit = self.extension_reg.predict(X_extension)
            
            # Create combined timeline and fits
            time_offset = flexion_times[-1] if len(flexion_times) > 0 else 0
            shifted_extension_times = extension_times + time_offset
            
            # Combine raw data
            combined_times = np.concatenate([flexion_times, shifted_extension_times])
            combined_torques = np.concatenate([flexion_torques, extension_torques])
            
            # Create combined fitted sinusoids
            shifted_extension_fit = extension_fit  # Extension fit doesn't need time shifting, just positioning
            combined_fits = np.concatenate([flexion_fit, shifted_extension_fit])
            
            # Calculate timing statistics for real timestamps
            flexion_intervals = np.diff(flexion_times)
            extension_intervals = np.diff(extension_times)
            
            # Create the matplotlib plot
            plt.figure(figsize=(15, 12))
            
            # Plot 1: Individual MVC data with separate fits
            plt.subplot(4, 1, 1)
            plt.plot(flexion_times, flexion_torques, 'b-', linewidth=1, alpha=0.7, label='Flexion MVC Data')
            plt.plot(flexion_times, flexion_fit, 'r--', linewidth=2, label=f'Flexion Fit (f={self.flexion_freq:.3f} Hz)')
            plt.plot(shifted_extension_times, extension_torques, 'g-', linewidth=1, alpha=0.7, label='Extension MVC Data')
            plt.plot(shifted_extension_times, extension_fit, 'm--', linewidth=2, label=f'Extension Fit (f={self.extension_freq:.3f} Hz)')
            plt.axvline(x=time_offset, color='k', linestyle='--', alpha=0.5, label='Flexion/Extension Boundary')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque')
            plt.title('Individual MVC Data with Separate Sinusoidal Fits (Real Timestamps)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 2: Combined raw data
            plt.subplot(4, 1, 2)
            plt.plot(combined_times, combined_torques, 'b-', linewidth=1, alpha=0.7, label='Combined MVC Data')
            plt.axvline(x=time_offset, color='k', linestyle='--', alpha=0.5, label='Flexion/Extension Boundary')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque')
            plt.title('Combined Flexion + Extension MVC Data (Real Timestamps)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 3: Combined data with combined sinusoidal fits
            plt.subplot(4, 1, 3)
            plt.plot(combined_times, combined_torques, 'b-', linewidth=1, alpha=0.7, label='Combined MVC Data')
            plt.plot(combined_times, combined_fits, 'r--', linewidth=3, label='Combined Sinusoidal Fits')
            plt.axvline(x=time_offset, color='k', linestyle='--', alpha=0.5, label='Flexion/Extension Boundary')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque')
            plt.title('Combined MVC Data with Combined Sinusoidal Fits (Real Timestamps)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 4: Timing intervals histogram
            plt.subplot(4, 1, 4)
            plt.hist(flexion_intervals * 1000, bins=20, alpha=0.7, label=f'Flexion Intervals (mean: {np.mean(flexion_intervals)*1000:.1f}ms)', color='blue')
            plt.hist(extension_intervals * 1000, bins=20, alpha=0.7, label=f'Extension Intervals (mean: {np.mean(extension_intervals)*1000:.1f}ms)', color='green')
            plt.axvline(x=20, color='red', linestyle='--', label='Target 20ms (50Hz)')
            plt.xlabel('Sampling Interval (ms)')
            plt.ylabel('Count')
            plt.title('Real Sampling Interval Distribution')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Print analysis information
            print(f"MVC Analysis Results (Real Timestamps):")
            print(f"Flexion - Frequency: {self.flexion_freq:.3f} Hz, Period: {1/self.flexion_freq:.2f} seconds")
            print(f"Flexion - Coefficients: {self.flexion_reg.coef_}, Intercept: {self.flexion_reg.intercept_:.3f}")
            print(f"Extension - Frequency: {self.extension_freq:.3f} Hz, Period: {1/self.extension_freq:.2f} seconds")
            print(f"Extension - Coefficients: {self.extension_reg.coef_}, Intercept: {self.extension_reg.intercept_:.3f}")
            print(f"Real timing statistics:")
            print(f"Flexion - Mean interval: {np.mean(flexion_intervals)*1000:.2f}ms, Std: {np.std(flexion_intervals)*1000:.2f}ms")
            print(f"Extension - Mean interval: {np.mean(extension_intervals)*1000:.2f}ms, Std: {np.std(extension_intervals)*1000:.2f}ms")

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
            
            # Plot 3: Difference Statistics
            plt.subplot(3, 1, 3)
            difference = np.abs(baseline_corrections)
            plt.hist(difference, bins=30, alpha=0.7, color='purple', edgecolor='black')
            plt.axvline(x=np.mean(difference), color='r', linestyle='--', linewidth=2, 
                       label=f'Mean Correction: {np.mean(difference):.3f}')
            plt.axvline(x=np.std(difference), color='orange', linestyle='--', linewidth=2, 
                       label=f'Std Correction: {np.std(difference):.3f}')
            plt.xlabel('Absolute Baseline Correction')
            plt.ylabel('Count')
            plt.title('Distribution of Baseline Corrections Applied')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Print correction statistics
            print(f"\nBaseline Correction Analysis:")
            print(f"Mean absolute correction: {np.mean(difference):.4f}")
            print(f"Standard deviation of corrections: {np.std(difference):.4f}")
            print(f"Maximum correction applied: {np.max(difference):.4f}")
            print(f"Percentage of data corrected: {(np.sum(difference > 0.001) / len(difference)) * 100:.1f}%")
            
            # Statistics by torque direction
            flexion_corrections = baseline_corrections[positive_mask]
            extension_corrections = baseline_corrections[negative_mask]
            
            if len(flexion_corrections) > 0:
                print(f"Flexion corrections - Mean: {np.mean(flexion_corrections):.4f}, Std: {np.std(flexion_corrections):.4f}")
            if len(extension_corrections) > 0:
                print(f"Extension corrections - Mean: {np.mean(extension_corrections):.4f}, Std: {np.std(extension_corrections):.4f}")
        else:
            print("No task data available for torque correction comparison.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
