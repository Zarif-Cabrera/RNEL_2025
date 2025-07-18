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
if 'AllTorques' not in st.session_state:
    st.session_state.AllTorques = []
if 'MVCStatus' not in st.session_state:
    st.session_state.MVCStatus = "**MVC Not Set**"
if 'flexionMVC' not in st.session_state:
    st.session_state.flexionMVC = []
if 'extensionMVC' not in st.session_state:
    st.session_state.extensionMVC = []
if 'MaxTorque' not in st.session_state:
    st.session_state.MaxTorque = 0
if 'flexionZeroBaselineStatus' not in st.session_state:
    st.session_state.flexionZeroBaselineStatus = "**Flexion Zero Baseline Not Set**"
if 'extensionZeroBaselineStatus' not in st.session_state:
    st.session_state.extensionZeroBaselineStatus = "**Extension Zero Baseline Not Set**"
if 'flexionZeroBaseline' not in st.session_state:
    st.session_state.flexionZeroBaseline = []
if 'extensionZeroBaseline' not in st.session_state:
    st.session_state.extensionZeroBaseline = []
if 'running' not in st.session_state:
    st.session_state.running = False
if 'trial' not in st.session_state:
    st.session_state.trial = 0
if 'creatingPath' not in st.session_state:
    st.session_state.creatingPath = True
if 'conductingTrial' not in st.session_state:
    st.session_state.conductingTrial = False
if 'xs' not in st.session_state:
    st.session_state.xs = []
if 'ys' not in st.session_state:
    st.session_state.ys = []
if 'CurrentTime' not in st.session_state:
    st.session_state.CurrentTime = []

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

def start_flexion_baseline(daq_input):
    if daq_input == "":
        st.error("Please enter a valid DAQ input.")
        return

    Tstop = 5
    Ts = .01
    N = int(Tstop/Ts)
    baselineReadings = []

    startDAQ(daq_input)

    for i in range(N):
        value = st.session_state.InputTask.read(number_of_samples_per_channel=10)
        value = [np.mean(value[0]),np.mean(value[1])]
        baselineReadings.append(value)
        time.sleep(Ts)

    st.session_state.InputTask.stop()
    st.session_state.InputTask.close() 

    baselineReadings = np.array(baselineReadings)

    st.session_state.zeroBaseline = [np.mean(baselineReadings[:,0]),np.mean(baselineReadings[:,1])] 
    st.session_state.zeroBaseline = [0,0]
    st.session_state.zeroBaselineStatus = "**Zero Baseline Set**"

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

def FindFlexionMVC(daq_input, durationCPM=21, window_size=50):
    time_vals, torque_vals = helperMVC(daq_input, durationCPM, window_size)

    flexionReadings = np.column_stack((time_vals, torque_vals))
    mask = flexionReadings[:, 1] >= 0  
    #swtich with zeroBaseline
    flexionReadings = flexionReadings[mask]
    st.session_state.flexionMVC = flexionReadings

def FindExtensionMVC(daq_input, durationCPM=21, window_size=50):
    time_vals, torque_vals = helperMVC(daq_input, durationCPM, window_size)

    extensionReadings = np.column_stack((time_vals, torque_vals))
    extensionReadings[:, 1] *= -1
    #take out when doing real trials
    mask = extensionReadings[:, 1] <= 0  
    #swtich with zeroBaseline
    extensionReadings = extensionReadings[mask]
    st.session_state.extensionMVC = extensionReadings

def start_task(daq_input, durationCPM=10, num_cycles=1, window_size=50):

    trialLength = (durationCPM *2)* num_cycles

    extensionMVC_shifted = st.session_state.extensionMVC.copy()
    extensionMVC_shifted[:, 0] += st.session_state.flexionMVC[-1, 0]
    fused_MVC = np.vstack((st.session_state.flexionMVC, extensionMVC_shifted))

    repeated_MVC = []
    for i in range(num_cycles):
        time_offset = i * fused_MVC[-1, 0]
        segment = fused_MVC.copy()
        segment[:, 0] += time_offset
        repeated_MVC.append(segment)

    repeated_MVC = np.vstack(repeated_MVC)
    print(len(repeated_MVC[1]),'repeated')
    if len(repeated_MVC[1]) > 100:
        repeated_MVC[:, 1] = moving_average(repeated_MVC, 1)


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
    st.session_state.Time = []

    plot_placeholder = st.empty()
    startDAQ(daq_input)
    start_time = time.time()
    # current_time = 0

    while st.session_state.conductingTrial:
        st.session_state.OutputTask.write(9)
        value = st.session_state.InputTask.read(number_of_samples_per_channel=20)
        value = [np.mean(value[0]),np.mean(value[1])]
        elapsed_time = time.time() - start_time

        if (elapsed_time) >= trialLength:
            break

        st.session_state.xs.append(elapsed_time)
        st.session_state.ys.append(value[1])
        st.session_state.Time.append(elapsed_time)

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
        time_window = st.session_state.Time[start_idx:end_idx]


        fig, ax = plt.subplots()
        ax.set_ylim(y_range)
        ax.axhline(0, color='k', linestyle='-', linewidth=1)

        MVC_time = repeated_MVC[:, 0]
        MVC_torque = repeated_MVC[:, 1]
        ax.plot(MVC_time[start_idx:end_idx+20], MVC_torque[start_idx:end_idx+20], color='red', linestyle=':', linewidth=2, label='Target MVC')


        if len(xs_window) > 1:
            ax.plot(time_window, ys_window, color='b', linestyle='-', linewidth=2, label='Measured Path')
        if len(xs_window) > 0:
            ax.scatter(time_window[-1], ys_window[-1], color='b', s=40, zorder=5, label='Current Point')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Voltage)')
        ax.grid()
        ax.legend()
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(Ts)

    print('we ended')
    st.session_state.OutputTask.write(0)
    elapsed = st.session_state.Time[-1]
    st.success(f"Task completed successfully! Elapsed time: {elapsed:.2f} seconds")

    st.session_state.InputTask.stop()
    st.session_state.InputTask.close()

    st.session_state.OutputTask.stop()
    st.session_state.OutputTask.close()

    arr = np.column_stack((st.session_state.xs, st.session_state.ys))
    np.savetxt(f'Trial{st.session_state.trial}.csv', arr, delimiter=',', fmt='%s')
    st.session_state.arr = arr

    st.session_state.trial += 1

def FindMVC():
    reg = LinearRegression()
    MVC_Path = np.array(st.session_state.flexionMVC)
    time = MVC_Path[:,0]
    # pos = MVC_Path[:,1]
    tor = MVC_Path[:,1]

    # print(time,'time')

    dt = np.mean(np.diff(time))
    N = len(time)
    # print(N,"N")
    yf = rfft(tor)
    xf = rfftfreq(N, dt)
    dominant_freq = xf[np.argmax(np.abs(yf[1:])) + 1]  
    # dominant_freq = dominant_freq + .006
    print(dominant_freq,"dom freq")

    X = np.column_stack((np.sin(2 * np.pi * dominant_freq * time), np.cos(2 * np.pi * dominant_freq * time)))
    reg.fit(X, tor)
    y_pred = reg.predict(X)
    return reg, dominant_freq, time, tor, y_pred

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
        col1, col2 = st.columns(2)
        with col1:
            durationCPM = st.number_input("**Length Of Cycle**", value=10.0,step=1.0)
            if st.button("Set Flexion MVC"):
                FindFlexionMVC(daq_input,durationCPM)
        with col2:
            num_cycles = st.number_input("**Number of Cycles**", value=1, step=1)
            if st.button("Set Extension MVC"):
                FindExtensionMVC(daq_input,durationCPM)

        st.subheader("Run Task")
        if st.button("Start"):
            st.session_state.conductingTrial = True
            start_task(daq_input,durationCPM,num_cycles)

    with st.expander("Stopping", expanded=True):
        if not isinstance(st.session_state.flexionMVC, list):

            reg, dominant_freq, time, tor, y_pred = FindMVC()

            y_pred = reg.predict(np.column_stack((np.sin(2 * np.pi * dominant_freq * st.session_state.flexionMVC[:,0]), np.cos(2 * np.pi * dominant_freq * st.session_state.flexionMVC[:,0]))))

            print(len(st.session_state.flexionMVC))
            fig, ax = plt.subplots()
            ax.plot(st.session_state.flexionMVC[:,0], y_pred, label='Linear Regression Prediction', linestyle='--')

            ax.plot(st.session_state.flexionMVC[:,0], st.session_state.flexionMVC[:,1], label='unsmooth value', linewidth=2)
            ax.plot(st.session_state.extensionMVC[:,0], st.session_state.extensionMVC[:,1], label='smoothed value', linestyle=':')
            ax.set_xlabel('Time')
            ax.set_ylabel('Torque')
            ax.legend()
            ax.grid()
            st.pyplot(fig)

create_components()

# need to test irl tmrw with DAQ as well as add in baseline functionality + figure out CPM
