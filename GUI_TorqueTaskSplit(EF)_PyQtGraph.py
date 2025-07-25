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

        self.window_size = 100
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.start_time = None
        self.acquiring = False
        self.trialNum = 1
        
        # Add frame counter for consistent timing
        self.frame_counter = 0
        self.target_sample_rate = 100  # 100 Hz (10ms intervals)
        
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
        self.frame_counter = 0
        
        # Setup plot
        self.plot_widget.clear()
        self.plot_widget.setTitle("Setting Flexion Baseline")
        self.curve = self.plot_widget.plot(pen='b', name='Baseline Torque')
        self.dot = pg.ScatterPlotItem(size=12, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(10)  # Same 10ms interval as task

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
        self.frame_counter = 0
        
        # Setup plot
        self.plot_widget.clear()
        self.plot_widget.setTitle("Setting Extension Baseline")
        self.curve = self.plot_widget.plot(pen='b', name='Baseline Torque')
        self.dot = pg.ScatterPlotItem(size=12, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(10)  # Same 10ms interval as task

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
        self.frame_counter = 0
        
        # Setup plot
        self.plot_widget.clear()
        self.plot_widget.setTitle("Setting Flexion MVC")
        self.curve = self.plot_widget.plot(pen='b', name='MVC Torque')
        self.dot = pg.ScatterPlotItem(size=12, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(10)  # Same 10ms interval as task

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
        self.frame_counter = 0
        
        # Setup plot
        self.plot_widget.clear()
        self.plot_widget.setTitle("Setting Extension MVC")
        self.curve = self.plot_widget.plot(pen='b', name='MVC Torque')
        self.dot = pg.ScatterPlotItem(size=12, brush=pg.mkBrush('b'))
        self.plot_widget.addItem(self.dot)
        
        # Start DAQ and timer
        self.startDAQ()
        self.timer.start(10)  # Same 10ms interval as task

    def start_task(self):
        self.acquiring = True
        self.df_task = pd.DataFrame(columns=["time", "position", "torque", "voltage"])
        self.start_time = None  # Will be set on first data point
        self.frame_counter = 0
        self.collection_mode = None  # Reset collection mode for task
        
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
        self.timer.start(10)  # update every 10 ms

    def update_plot(self):
        if not self.acquiring:
            return
        
        # Set start time on first data point
        if self.start_time is None:
            self.start_time = time.time()
            self.frame_counter = 0
            
        value = self.input_task.read(number_of_samples_per_channel=10)
        
        # Use frame counter for consistent timing instead of wall clock time
        t = self.frame_counter / self.target_sample_rate
        self.frame_counter += 1
        
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
            
            # --- Append to DataFrame ---
            new_row = pd.DataFrame([[t, position, torque, voltage]], columns=self.df_task.columns)
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
                    y_combined[flexion_mask] = self.flexion_reg.predict(X_flexion)
                
                if len(extension_phase_times) > 0:
                    X_extension = np.column_stack((
                        np.sin(2 * np.pi * self.extension_freq * extension_phase_times),
                        np.cos(2 * np.pi * self.extension_freq * extension_phase_times)
                    ))
                    y_combined[extension_mask] = self.extension_reg.predict(X_extension)
                
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
                # --- Save all data to CSV ---
                self.df_task.to_csv(f"DAQ_Task_Trial{self.trialNum}.csv", index=False)
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
            self.df_baseline.to_csv("FlexionBaseline_Data.csv", index=False)
            print("Flexion baseline data saved to FlexionBaseline_Data.csv")
        elif self.collection_mode == 'extension_baseline':
            self.df_ext_baseline = pd.DataFrame(self.collection_data, columns=["time", "torque"])
            print("Extension Baseline set.")
            # Save extension baseline data to CSV
            self.df_ext_baseline.to_csv("ExtensionBaseline_Data.csv", index=False)
            print("Extension baseline data saved to ExtensionBaseline_Data.csv")
        elif self.collection_mode == 'flexion_mvc':
            self.df_flexionMVC = pd.DataFrame(self.collection_data, columns=["time", "torque"])
            print("Flexion MVC set.")
            # Save flexion MVC data to CSV
            self.df_flexionMVC.to_csv("FlexionMVC_Data.csv", index=False)
            print("Flexion MVC data saved to FlexionMVC_Data.csv")
        elif self.collection_mode == 'extension_mvc':
            self.df_extensionMVC = pd.DataFrame(self.collection_data, columns=["time", "torque"])
            print("Extension MVC set.")
            # Save extension MVC data to CSV
            self.df_extensionMVC.to_csv("ExtensionMVC_Data.csv", index=False)
            print("Extension MVC data saved to ExtensionMVC_Data.csv")
            # Check if both MVCs are now available and calculate sinusoidal fit
            self.calculate_mvc_fit()
            self.plot_combined_mvc_analysis()
        
        # Reset collection variables
        self.collection_mode = None
        self.collection_data = []
        self.plot_widget.setTitle("Real-Time Torque Plot")  # Reset title

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
            
            # Create the matplotlib plot
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Individual MVC data with separate fits
            plt.subplot(3, 1, 1)
            plt.plot(flexion_times, flexion_torques, 'b-', linewidth=1, alpha=0.7, label='Flexion MVC Data')
            plt.plot(flexion_times, flexion_fit, 'r--', linewidth=2, label=f'Flexion Fit (f={self.flexion_freq:.3f} Hz)')
            plt.plot(shifted_extension_times, extension_torques, 'g-', linewidth=1, alpha=0.7, label='Extension MVC Data')
            plt.plot(shifted_extension_times, extension_fit, 'm--', linewidth=2, label=f'Extension Fit (f={self.extension_freq:.3f} Hz)')
            plt.axvline(x=time_offset, color='k', linestyle='--', alpha=0.5, label='Flexion/Extension Boundary')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque')
            plt.title('Individual MVC Data with Separate Sinusoidal Fits')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 2: Combined raw data
            plt.subplot(3, 1, 2)
            plt.plot(combined_times, combined_torques, 'b-', linewidth=1, alpha=0.7, label='Combined MVC Data')
            plt.axvline(x=time_offset, color='k', linestyle='--', alpha=0.5, label='Flexion/Extension Boundary')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque')
            plt.title('Combined Flexion + Extension MVC Data')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 3: Combined data with combined sinusoidal fits
            plt.subplot(3, 1, 3)
            plt.plot(combined_times, combined_torques, 'b-', linewidth=1, alpha=0.7, label='Combined MVC Data')
            plt.plot(combined_times, combined_fits, 'r--', linewidth=3, label='Combined Sinusoidal Fits')
            plt.axvline(x=time_offset, color='k', linestyle='--', alpha=0.5, label='Flexion/Extension Boundary')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque')
            plt.title('Combined MVC Data with Combined Sinusoidal Fits')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Print analysis information
            print(f"MVC Analysis Results:")
            print(f"Flexion - Frequency: {self.flexion_freq:.3f} Hz, Period: {1/self.flexion_freq:.2f} seconds")
            print(f"Flexion - Coefficients: {self.flexion_reg.coef_}, Intercept: {self.flexion_reg.intercept_:.3f}")
            print(f"Extension - Frequency: {self.extension_freq:.3f} Hz, Period: {1/self.extension_freq:.2f} seconds")
            print(f"Extension - Coefficients: {self.extension_reg.coef_}, Intercept: {self.extension_reg.intercept_:.3f}")

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
