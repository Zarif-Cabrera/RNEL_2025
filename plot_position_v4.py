import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_position_over_time(csv_filename="DAQ_Task_Trial1_v4.csv"):
    """
    Plot position over time from DAQ task data
    
    Parameters:
    csv_filename (str): Name of the CSV file to read
    """
    
    # Check if file exists
    if not os.path.exists(csv_filename):
        print(f"Error: File '{csv_filename}' not found!")
        return
    
    try:
        # Read the CSV file
        print(f"Reading data from {csv_filename}...")
        df = pd.read_csv(csv_filename)
        
        # Check if required columns exist
        required_columns = ['time', 'position']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Display basic statistics
        print(f"Data loaded successfully!")
        print(f"  Total data points: {len(df)}")
        print(f"  Time range: {df['time'].min():.3f} to {df['time'].max():.3f} seconds")
        print(f"  Position range: {df['position'].min():.4f} to {df['position'].max():.4f}")
        print(f"  Duration: {df['time'].max() - df['time'].min():.3f} seconds")
        
        # Find and print times when position is at minimum
        min_position = df['position'].min()
        min_positions_mask = df['position'] == min_position
        min_times = df[min_positions_mask]['time'].values
        
        print(f"\nMinimum Position Analysis:")
        print(f"  Minimum position value: {min_position:.4f}")
        print(f"  Number of times at minimum: {len(min_times)}")
        print(f"  Times when position is minimum:")
        for i, time_val in enumerate(min_times):
            print(f"    {i+1:2d}. Time: {time_val:.3f}s")
        
        # Find near-minimum positions (within 0.1% of minimum)
        position_range = df['position'].max() - df['position'].min()
        tolerance = position_range * 0.001  # 0.1% tolerance
        near_min_mask = df['position'] <= (min_position + tolerance)
        near_min_times = df[near_min_mask]['time'].values
        
        if len(near_min_times) > len(min_times):
            print(f"\nNear-Minimum Position Analysis (within {tolerance:.4f} of minimum):")
            print(f"  Number of near-minimum positions: {len(near_min_times)}")
            print(f"  Near-minimum times:")
            for i, time_val in enumerate(near_min_times):
                if i < 20:  # Limit to first 20 to avoid too much output
                    print(f"    {i+1:2d}. Time: {time_val:.3f}s")
                elif i == 20:
                    print(f"    ... and {len(near_min_times)-20} more")
                    break
        
        # Find local minima (turning points) using scipy if available
        try:
            from scipy.signal import find_peaks
            # Find local minima by finding peaks in the inverted signal
            inverted_position = -df['position'].values
            peaks, properties = find_peaks(inverted_position, height=None, distance=50)  # distance prevents too close peaks
            
            if len(peaks) > 0:
                local_min_times = df.iloc[peaks]['time'].values
                local_min_positions = df.iloc[peaks]['position'].values
                
                print(f"\nLocal Minima Analysis (turning points):")
                print(f"  Number of local minima found: {len(local_min_times)}")
                print(f"  Local minima times and positions:")
                for i, (time_val, pos_val) in enumerate(zip(local_min_times, local_min_positions)):
                    print(f"    {i+1:2d}. Time: {time_val:6.3f}s, Position: {pos_val:.4f}")
        except ImportError:
            print("\nNote: scipy not available - skipping local minima detection")
            local_min_times = None
            local_min_positions = None
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Add interactive coordinate display
        def on_click(event):
            if event.inaxes is not None and event.button == 1:  # Left mouse button
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    print(f"Clicked coordinates: Time = {x:.3f}s, Position = {y:.4f}")
                    
                    # Find nearest data point
                    time_diff = np.abs(df['time'].values - x)
                    nearest_idx = np.argmin(time_diff)
                    nearest_time = df['time'].iloc[nearest_idx]
                    nearest_position = df['position'].iloc[nearest_idx]
                    nearest_torque = df['torque'].iloc[nearest_idx] if 'torque' in df.columns else 'N/A'
                    nearest_velocity = df['velocity'].iloc[nearest_idx] if 'velocity' in df.columns else 'N/A'
                    
                    print(f"Nearest data point: Time = {nearest_time:.3f}s, Position = {nearest_position:.4f}")
                    if 'torque' in df.columns:
                        print(f"                    Torque = {nearest_torque:.4f}, Velocity = {nearest_velocity:.4f}")
        
        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        # Main plot: Position over time
        plt.subplot(2, 1, 1)
        plt.plot(df['time'].values, df['position'].values, 'b-', linewidth=1.5, alpha=0.8, label='Position')
        
        # Mark minimum positions on the plot
        min_position = df['position'].min()
        min_positions_mask = df['position'] == min_position
        min_times = df[min_positions_mask]['time'].values
        min_positions_values = df[min_positions_mask]['position'].values
        plt.scatter(min_times, min_positions_values, color='red', s=50, zorder=5, label=f'Global Min ({len(min_times)} points)')
        
        # Add local minima to plot if available
        try:
            from scipy.signal import find_peaks
            # Find local minima by finding peaks in the inverted signal
            inverted_position = -df['position'].values
            peaks, properties = find_peaks(inverted_position, height=None, distance=50)
            
            if len(peaks) > 0:
                local_min_times = df.iloc[peaks]['time'].values
                local_min_positions = df.iloc[peaks]['position'].values
                plt.scatter(local_min_times, local_min_positions, color='orange', s=30, zorder=4, 
                           marker='v', label=f'Local Min ({len(local_min_times)} points)')
        except ImportError:
            pass
        
        plt.xlabel('Time (s)')
        plt.ylabel('Position')
        plt.title(f'Position Over Time - {csv_filename}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistics text box
        stats_text = f"Mean: {df['position'].mean():.4f}\n"
        stats_text += f"Std: {df['position'].std():.4f}\n"
        stats_text += f"Range: {df['position'].max() - df['position'].min():.4f}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Secondary plot: Position histogram
        plt.subplot(2, 1, 2)
        plt.hist(df['position'].values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Position')
        plt.ylabel('Frequency')
        plt.title('Position Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = csv_filename.replace('.csv', '_position_plot.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
        
        # Add instruction for interactive use
        print(f"\nInteractive Plot Instructions:")
        print(f"  - Left-click on the plot to see coordinates")
        print(f"  - Nearest data point information will be displayed")
        print(f"  - Close the plot window to continue")
        
        # Show the plot
        plt.show()
        
        # Optional: Analyze cycles if data spans multiple cycles
        total_time = df['time'].max() - df['time'].min()
        if total_time > 15:  # If more than 15 seconds, likely multiple cycles
            print("\nCycle Analysis:")
            # Estimate cycle period (assuming 10-second cycles based on your setup)
            estimated_cycle_period = 10.0
            num_cycles = total_time / estimated_cycle_period
            print(f"  Estimated cycles: {num_cycles:.1f} (assuming {estimated_cycle_period}s period)")
            
            # Plot position for first few cycles
            plt.figure(figsize=(12, 6))
            
            # Show first 30 seconds (3 cycles)
            mask = df['time'] <= 30
            if mask.any():
                plt.plot(df[mask]['time'].values, df[mask]['position'].values, 'b-', linewidth=2)
                plt.xlabel('Time (s)')
                plt.ylabel('Position')
                plt.title('Position - First 30 Seconds (3 Cycles)')
                plt.grid(True, alpha=0.3)
                
                # Add vertical lines at cycle boundaries
                for i in range(1, 4):
                    cycle_time = i * estimated_cycle_period
                    if cycle_time <= 30:
                        plt.axvline(x=cycle_time, color='r', linestyle='--', alpha=0.7, label=f'Cycle {i}' if i == 1 else "")
                
                if 'Cycle 1' in plt.gca().get_legend_handles_labels()[1]:
                    plt.legend()
                
                cycle_plot_filename = csv_filename.replace('.csv', '_position_cycles.png')
                plt.savefig(cycle_plot_filename, dpi=300, bbox_inches='tight')
                print(f"Cycle plot saved as: {cycle_plot_filename}")
                plt.show()
        
    except Exception as e:
        print(f"Error reading or plotting data: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_all_trials():
    """Plot position for all available trial files"""
    import glob
    
    # Find all DAQ_Task_Trial*_v4.csv files
    pattern = "DAQ_Task_Trial*_v4.csv"
    trial_files = glob.glob(pattern)
    
    if not trial_files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(trial_files)} trial files:")
    for i, filename in enumerate(trial_files):
        print(f"  {i+1}. {filename}")
    
    # Plot each trial
    for filename in trial_files:
        print(f"\n{'='*50}")
        print(f"Processing: {filename}")
        print('='*50)
        plot_position_over_time(filename)

if __name__ == "__main__":
    # You can modify this to plot specific trials or all trials
    
    # Option 1: Plot a specific trial
    plot_position_over_time("DAQ_Task_Trial1_v4.csv")
    
    # Option 2: Uncomment the line below to plot all available trials
    # plot_all_trials()
    
    print("\nPlotting complete!")
