import time
import nidaqmx
import numpy as np
import streamlit as st
import scipy.signal as signal 
import scipy.optimize 
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from scipy.fft import rfft, rfftfreq

if 'DaqStatus' not in st.session_state:
    st.session_state.DaqStatus = "**DAQ Not Set**"
if 'MVCStatus' not in st.session_state:
    st.session_state.MVCStatus = "**MVC Not Set**"
if 'flexionMVC' not in st.session_state:
    st.session_state.flexionMVC = []
if 'extensionMVC' not in st.session_state:
    st.session_state.extensionMVC = []
if 'zeroBaselineStatus' not in st.session_state:
    st.session_state.flexionZeroBaselineStatus = "**Zero Baseline Not Set**"
if 'zeroBaseline' not in st.session_state:
    st.session_state.zeroBaseline = []
if 'trial' not in st.session_state:
    st.session_state.trial = 0
if 'conductingTrial' not in st.session_state:
    st.session_state.conductingTrial = False
if 'xs' not in st.session_state:
    st.session_state.xs = []
if 'ys' not in st.session_state:
    st.session_state.ys = []
if 'pos' not in st.session_state:
    st.session_state.pos = []
if 'voltage' not in st.session_state:
    st.session_state.voltage = []
if 'baseline_flex_reg' not in st.session_state:
    st.session_state.baseline_flex_reg = None
if 'baselineFLEX' not in st.session_state:
    st.session_state.baselineFLEX = []
if 'baseline_flex_freq' not in st.session_state:
    st.session_state.baseline_flex_freq = None
if 'baseline_ext_reg' not in st.session_state:
    st.session_state.baseline_ext_reg = None
if 'baselineEXT' not in st.session_state:
    st.session_state.baselineEXT = []
if 'baseline_ext_freq' not in st.session_state:
    st.session_state.baseline_ext_freq = None

def startDAQ(daq_input):
    if 'InputTask' in st.session_state and st.session_state.InputTask is not None:
        print("still in task")
        try:
            st.session_state.InputTask.stop()
            st.session_state.InputTask.close()
        except Exception:
            pass

    if 'OutputTask' in st.session_state and st.session_state.OutputTask is not None:
        print("still out task")
        try:
            st.session_state.OutputTask.stop()
            st.session_state.OutputTask.close()
        except Exception:
            pass

    st.session_state.InputTask = nidaqmx.Task()

    st.session_state.InputTask.ai_channels.add_ai_voltage_chan(f"{daq_input}/ai3", terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
    #make this position
    st.session_state.InputTask.ai_channels.add_ai_voltage_chan(f"{daq_input}/ai2", terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
    #make this torque
    st.session_state.InputTask.ai_channels.add_ai_voltage_chan(f"{daq_input}/ai4", terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
    #make this channel you write voltage to 
    st.session_state.OutputTask = nidaqmx.Task()
    st.session_state.OutputTask.ao_channels.add_ao_voltage_chan(f"{daq_input}/ao0",min_val=-10.0, max_val=10.0)  # Analog output channel

    st.session_state.InputTask.start()
    st.session_state.OutputTask.start()

def CheckForDAQ(daq_input):
    system = nidaqmx.system.System.local()
    available_devices = [str(dev.name) for dev in system.devices]
    return daq_input in available_devices

def moving_average(arr,column):
    """Simple moving average for a 1D numpy array."""
    Fs = 1250
    Fcut = 75
    #og was 15
    
    # return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')
    B,A = signal.butter(2,(2*Fcut)/Fs,'low',analog=False)
    arr_filt = signal.filtfilt(B,A,arr[:,column],padlen=100)
    #padlen=100
    return arr_filt

def set_baseline(daq_input, durationCPM):
    if daq_input == "":
        st.error("Please enter a valid DAQ input.")
        return

    baselineReadings = []

    startDAQ(daq_input)

    Ts = 0.15483107604086402

    start_time = time.time()
    while (time.time() - start_time) < durationCPM:
        value = st.session_state.InputTask.read(number_of_samples_per_channel=10)
        value = [time.time() - start_time, np.mean(value[0]),np.mean(value[1])]
        baselineReadings.append(value)
        time.sleep(Ts)

    st.session_state.InputTask.stop()
    st.session_state.InputTask.close() 

    baselineReadings = np.array(baselineReadings)

    time_vals = baselineReadings[:, 0]
    torque_vals = baselineReadings[:, 2]

    dt = np.mean(np.diff(time_vals))
    N = len(time_vals)
    yf = rfft(torque_vals)
    xf = rfftfreq(N, dt)
    dominant_freq = xf[np.argmax(np.abs(yf[1:])) + 1]  

    X = np.column_stack((
        np.sin(2 * np.pi * dominant_freq * time_vals),
        np.cos(2 * np.pi * dominant_freq * time_vals)
    ))
    reg = LinearRegression()
    reg.fit(X, torque_vals)
    return reg, baselineReadings, dominant_freq

def helperMVC(daq_input, durationCPM=21, window_size=50):
    if daq_input == "":
        st.error("Please enter a valid DAQ input.")
        return

    y_range = [-10, 10]

    Ts = 0.01

    # Lists to store data
    time_vals = []
    torque_vals = []

    plot_placeholder = st.empty()
    startDAQ(daq_input)
    start_time = time.time()

    while (time.time() - start_time) < durationCPM:
        value = st.session_state.InputTask.read(number_of_samples_per_channel=20)
        current_time = time.time() - start_time
        torque = np.mean(value[1]) 
        time_vals.append(current_time)
        torque_vals.append(torque)

        # Moving window logic
        i = len(time_vals) - 1
        mid_idx = window_size // 2
        data_len = len(time_vals)

        if data_len < window_size:
            start_idx = 0
            end_idx = data_len
            circle_idx = i
        elif i < mid_idx:
            start_idx = 0
            end_idx = window_size
            circle_idx = i
        elif i > data_len - mid_idx - 1:
            start_idx = data_len - window_size
            end_idx = data_len
            circle_idx = i - start_idx
        else:
            start_idx = i - mid_idx
            end_idx = i + mid_idx
            circle_idx = mid_idx

        # Windowed data
        time_window = time_vals[start_idx:end_idx]
        torque_window = torque_vals[start_idx:end_idx]

        fig, ax = plt.subplots()
        ax.set_ylim(y_range)
        ax.axhline(0, color='k', linestyle='-', linewidth=1)

        if len(time_window) > 1:
            ax.plot(time_window, torque_window, color='b', linestyle='-', linewidth=2, label='Measured Torque')
        if len(time_window) > 0:
            ax.scatter(time_window[-1], torque_window[-1], color='b', s=40, zorder=5, label='Current Point')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Voltage)')
        ax.grid()
        ax.legend()
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(Ts)

    st.session_state.InputTask.stop()
    st.session_state.InputTask.close()
    return time_vals, torque_vals

def FindFlexionMVC(daq_input, durationCPM=2, window_size=50):
    time_vals, torque_vals = helperMVC(daq_input, durationCPM, window_size)

    flexionReadings = np.column_stack((time_vals, torque_vals))
    mask = flexionReadings[:, 1] >= min(np.array(st.session_state.baselineFLEX)[:, 2])
    #swtich with zeroBaseline
    flexionReadings = flexionReadings[mask]
    st.session_state.flexionMVC = flexionReadings

def FindExtensionMVC(daq_input, durationCPM=2, window_size=50):
    time_vals, torque_vals = helperMVC(daq_input, durationCPM, window_size)

    extensionReadings = np.column_stack((time_vals, torque_vals))
    extensionReadings[:, 1] *= -1
    #take out when doing real trials
    mask = extensionReadings[:, 1] <= max(np.array(st.session_state.baselineEXT)[:, 2])
    #swtich with zeroBaseline
    extensionReadings = extensionReadings[mask]
    st.session_state.extensionMVC = extensionReadings

def start_task(daq_input, durationCPM=2, num_cycles=1, window_size=10):

    trialLength = (durationCPM *2)* num_cycles

    extensionMVC_shifted = st.session_state.extensionMVC.copy()
    extensionMVC_shifted[:, 0] += st.session_state.flexionMVC[-1, 0]
    fused_MVC = np.vstack((st.session_state.flexionMVC, extensionMVC_shifted))

    repeated_MVC = []
    for i in range(100):
        time_offset = i * fused_MVC[-1, 0]
        segment = fused_MVC.copy()
        segment[:, 0] += time_offset
        repeated_MVC.append(segment)

    repeated_MVC = np.vstack(repeated_MVC)

    # if len(repeated_MVC[1]) > 100:
    #     repeated_MVC[:, 1] = moving_average(repeated_MVC, 1)

    time_vals = repeated_MVC[:, 0]
    torque_vals = repeated_MVC[:, 1]

    dt = np.mean(np.diff(time_vals))
    N = len(time_vals)
    yf = rfft(torque_vals)
    xf = rfftfreq(N, dt)
    dominant_freq = xf[np.argmax(np.abs(yf[1:])) + 1]  

    X = np.column_stack((
            np.sin(2 * np.pi * dominant_freq * time_vals),
            np.cos(2 * np.pi * dominant_freq * time_vals)
    ))
    reg = LinearRegression()
    reg.fit(X, torque_vals)

    X_sin = np.column_stack((
                np.sin(2 * np.pi * dominant_freq * time_vals),
                np.cos(2 * np.pi * dominant_freq * time_vals)
    ))
    MVC_fit = reg.predict(X_sin)


    y_top_range = math.ceil(st.session_state.flexionMVC[:,1].max() + (.1 * st.session_state.flexionMVC[:,1].max()))
    y_bottom_range = math.floor(st.session_state.extensionMVC[:,1].min() - (.1 * abs(st.session_state.extensionMVC[:,1].min())))

    if y_top_range < 10:
        y_top_range = 10
    if y_bottom_range > -10:
        y_bottom_range = -10
    
    y_range = [y_bottom_range, y_top_range]

    Ts = 0.001 
    #.01

    st.session_state.xs = []
    st.session_state.ys = []
    st.session_state.pos = []
    st.session_state.voltage = []

    plot_placeholder = st.empty()
    startDAQ(daq_input)
    start_time = time.time()
    # current_time = 0

    while st.session_state.conductingTrial:
        st.session_state.OutputTask.write(9)
        value = st.session_state.InputTask.read(number_of_samples_per_channel=20)
        value = [np.mean(value[0]),np.mean(value[1]),np.mean(value[2])]
        elapsed_time = time.time() - start_time

        if (elapsed_time) >= trialLength:
            break

        st.session_state.xs.append(elapsed_time)
        st.session_state.pos.append(value[0])
        st.session_state.ys.append(value[1])
        st.session_state.voltage.append(value[2])

        i = len(st.session_state.xs) - 1
        mid_idx = window_size // 2
        data_len = len(st.session_state.xs)

        if data_len < window_size:
            start_idx = 0
            end_idx = data_len
            circle_idx = i
        elif i < mid_idx:
            start_idx = 0
            end_idx = window_size
            circle_idx = i
        elif i > data_len - mid_idx - 1:
            start_idx = data_len - window_size
            end_idx = data_len
            circle_idx = i - start_idx
        else:
            start_idx = i - mid_idx
            end_idx = i + mid_idx
            circle_idx = mid_idx

        # Windowed data
        xs_window = st.session_state.xs[start_idx:end_idx]
        ys_window = st.session_state.ys[start_idx:end_idx]

        fig, ax = plt.subplots()
        ax.set_ylim(y_range)
        ax.axhline(0, color='k', linestyle='-', linewidth=1)

        # MVC_time = repeated_MVC[:, 0]
        # MVC_torque = repeated_MVC[:, 1]
        # ax.plot(MVC_time[start_idx:end_idx+20], MVC_torque[start_idx:end_idx+20], color='red', linestyle='-', linewidth=5, label='Target MVC')
        mvc_end = end_idx + 20

        ax.plot(time_vals[start_idx:mvc_end], MVC_fit[start_idx:mvc_end], color='red', linestyle='-', label='Baseline Sinusoid Fit', linewidth=2)

        if len(xs_window) > 1:
            ax.plot(xs_window, ys_window, color='b', linestyle='-', linewidth=2, label='Measured Path')
        if len(xs_window) > 0:
            ax.scatter(xs_window[-1], ys_window[-1], color='b', s=40, zorder=5, label='Current Point')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Voltage)')
        ax.grid()
        ax.legend()
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(Ts)

    print('we ended')
    st.session_state.OutputTask.write(0)
    elapsed = st.session_state.xs[-1]
    st.success(f"Task completed successfully! Elapsed time: {elapsed:.2f} seconds")

    st.session_state.InputTask.stop()
    st.session_state.InputTask.close()

    st.session_state.OutputTask.stop()
    st.session_state.OutputTask.close()

    arr = np.column_stack((st.session_state.xs, st.session_state.pos, st.session_state.ys, st.session_state.voltage))
    np.savetxt(f'Trial{st.session_state.trial}.csv', arr, delimiter=',', fmt='%s')
    st.session_state.arr = arr

    st.session_state.trial += 1

def create_components():
    st.title("Real-Time Plotting")

    with st.expander("Settings", expanded=True):
        st.subheader("Check For DAQ")
        daq_input = st.text_input("**DAQ Input**", value="")
        if st.button("Check Validity"):
            if CheckForDAQ(daq_input):
                st.session_state.DaqStatus = "**DAQ Set**"
            else:
                st.session_state.DaqStatus = "**DAQ Does Not Exist**"
        st.markdown(st.session_state.DaqStatus, help="Connected DAQ Status")

        st.subheader("Set Up")
        col1, col2, col3 = st.columns(3)
        with col1:
            durationCPM = st.number_input("**Length Of Flexion**", value=10.0,step=1.0)
            if st.button("Set Flexion Baseline"):
                st.session_state.baseline_flex_reg, st.session_state.baselineFLEX, st.session_state.baseline_flex_freq = set_baseline(daq_input, durationCPM)

                print('flex', st.session_state.baselineFLEX)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  
            if st.button("Set Extension Baseline"):
                st.session_state.baseline_ext_reg, st.session_state.baselineEXT, st.session_state.baseline_ext_freq = set_baseline(daq_input, durationCPM)
            if st.button("Set Flexion MVC"):
                FindFlexionMVC(daq_input, durationCPM)
        with col3:
            num_cycles = st.number_input("**Number of Cycles**", value=1, step=1)
            if st.button("Set Extension MVC"):
                FindExtensionMVC(daq_input,durationCPM)

        st.subheader("Run Task")
        if st.button("Start"):
            st.session_state.conductingTrial = True
            start_task(daq_input,durationCPM,num_cycles)

    with st.expander("Stopping", expanded=True):
        if not isinstance(st.session_state.flexionMVC, list):

            extensionMVC_shifted = st.session_state.extensionMVC.copy()
            extensionMVC_shifted[:, 0] += st.session_state.flexionMVC[-1, 0]
            fused_MVC = np.vstack((st.session_state.flexionMVC, extensionMVC_shifted))

            repeated_MVC = []
            for i in range(10):
                time_offset = i * fused_MVC[-1, 0]
                segment = fused_MVC.copy()
                segment[:, 0] += time_offset
                repeated_MVC.append(segment)

            repeated_MVC = np.vstack(repeated_MVC)

            fig, ax = plt.subplots()

            ax.plot(repeated_MVC[:,0][:100], repeated_MVC[:,1][:100], label='flex mvc', linewidth=2)
            time_vals = repeated_MVC[:, 0]
            torque_vals = repeated_MVC[:, 1]

            dt = np.mean(np.diff(time_vals))
            N = len(time_vals)
            yf = rfft(torque_vals)
            xf = rfftfreq(N, dt)
            dominant_freq = xf[np.argmax(np.abs(yf[1:])) + 1]  

            X = np.column_stack((
                np.sin(2 * np.pi * dominant_freq * time_vals),
                np.cos(2 * np.pi * dominant_freq * time_vals)
            ))
            reg = LinearRegression()
            reg.fit(X, torque_vals)

            X_sin = np.column_stack((
                    np.sin(2 * np.pi * dominant_freq * time_vals),
                    np.cos(2 * np.pi * dominant_freq * time_vals)
                ))
            baseline_fit = reg.predict(X_sin)
            ax.plot(time_vals[:100], baseline_fit[:100], color='red', linestyle='-', label='Baseline Sinusoid Fit', linewidth=2)
            # ax.plot(st.session_state.extensionMVC[:,0], st.session_state.extensionMVC[:,1], label='ext mvc', linestyle=':')
            # if hasattr(st.session_state, "baselineFLEX") and st.session_state.baselineFLEX is not None:
            #     baseline = st.session_state.baselineFLEX
            #     ax.plot(baseline[:,0], baseline[:,2], color='orange', label='Baseline Readings')
            #     # Sinusoidal best fit
            #     baseline_times = baseline[:,0]
            #     dominant_freq = st.session_state.baseline_flex_freq  # Store this in session state when you call set_baseline
            #     X_sin = np.column_stack((
            #         np.sin(2 * np.pi * dominant_freq * baseline_times),
            #         np.cos(2 * np.pi * dominant_freq * baseline_times)
            #     ))
            #     baseline_fit = st.session_state.baseline_flex_reg.predict(X_sin)
            #     ax.plot(baseline_times, baseline_fit, color='red', linestyle='-', label='Baseline Sinusoid Fit', linewidth=2)

            ax.set_xlabel('Time')
            ax.set_ylabel('Torque')
            ax.legend()
            ax.grid()
            st.pyplot(fig)

create_components()

# need to test irl tmrw with DAQ as well as add in baseline functionality + figure out CPM
