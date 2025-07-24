import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def calculate_sinusoidal_fit(times, torques, label=""):
    """Calculate sinusoidal fit for given time and torque data"""
    dt = np.mean(np.diff(times))
    N = len(times)
    yf = rfft(torques)
    xf = rfftfreq(N, dt)
    dominant_freq = xf[np.argmax(np.abs(yf[1:])) + 1]
    
    X = np.column_stack((
        np.sin(2 * np.pi * dominant_freq * times),
        np.cos(2 * np.pi * dominant_freq * times)
    ))
    reg = LinearRegression()
    reg.fit(X, torques)
    
    # Calculate fit
    fit_values = reg.predict(X)
    
    # Calculate fit quality metrics
    r2 = reg.score(X, torques)
    rmse = np.sqrt(np.mean((torques - fit_values)**2))
    
    print(f"{label} - Frequency: {dominant_freq:.3f} Hz, R²: {r2:.3f}, RMSE: {rmse:.4f}")
    print(f"{label} - Coefficients: {reg.coef_}, Intercept: {reg.intercept_:.3f}")
    print(f"{label} - Data points: {len(times)}")
    
    return dominant_freq, reg, fit_values, r2, rmse

def filter_by_velocity(df, velocity_threshold=0.02, phase_type=""):
    """Filter DataFrame based on phase-specific velocity criteria
    
    Args:
        df: DataFrame to filter
        velocity_threshold: velocity threshold (default 0.02)
        phase_type: 'flexion' or 'extension' for phase-specific filtering
    """
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
    
    print(f"Velocity filtering results:")
    print(f"Initial data points: {initial_count}")
    print(f"Final data points: {final_count}")
    print(f"Removed data points: {removed_count} ({removed_percentage:.1f}%)")
    print(f"Filter criterion: removed where {filter_description}")
    print(f"Phase type: {phase_type if phase_type else 'symmetric'}")
    print(f"Velocity threshold: ±{velocity_threshold}")
    
    return filtered_df

def compare_baseline_fits(csv_file_path, data_type="Flexion Baseline"):
    """Compare sinusoidal fits before and after velocity filtering"""
    try:
        # Load the data
        df = pd.read_csv(csv_file_path)
        print(f"\nAnalyzing {data_type} data from: {csv_file_path}")
        print(f"Columns in data: {df.columns.tolist()}")
        
        # Check if velocity column exists
        if 'velocity' not in df.columns:
            print(f"Warning: 'velocity' column not found in {csv_file_path}")
            print("Available columns:", df.columns.tolist())
            return
        
        # Original data
        original_times = df['time'].values
        original_torques = df['torque'].values
        original_velocities = df['velocity'].values
        
        # Determine phase type for filtering
        phase_type = "flexion" if "Flexion" in data_type else "extension" if "Extension" in data_type else ""
        
        # Filter data by velocity
        filtered_df = filter_by_velocity(df, velocity_threshold=0.02, phase_type=phase_type)
        
        if len(filtered_df) == 0:
            print(f"Warning: No data points remain after velocity filtering for {data_type}")
            return
        
        filtered_times = filtered_df['time'].values
        filtered_torques = filtered_df['torque'].values
        filtered_velocities = filtered_df['velocity'].values
        
        # Calculate sinusoidal fits
        print(f"\n{data_type} - Original Data Fit:")
        orig_freq, orig_reg, orig_fit, orig_r2, orig_rmse = calculate_sinusoidal_fit(
            original_times, original_torques, f"{data_type} Original"
        )
        
        print(f"\n{data_type} - Filtered Data Fit:")
        filt_freq, filt_reg, filt_fit, filt_r2, filt_rmse = calculate_sinusoidal_fit(
            filtered_times, filtered_torques, f"{data_type} Filtered"
        )
        
        # Create comparison plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'{data_type} Data: Original vs Velocity-Filtered Comparison', fontsize=16)
        
        # Plot 1: Original data and fit
        axes[0, 0].plot(original_times, original_torques, 'b-', linewidth=1, alpha=0.7, label='Original Data')
        axes[0, 0].plot(original_times, orig_fit, 'r--', linewidth=2, label=f'Fit (f={orig_freq:.3f} Hz)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Torque')
        axes[0, 0].set_title(f'Original Data\nR²={orig_r2:.3f}, RMSE={orig_rmse:.4f}, N={len(original_times)}')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Filtered data and fit
        axes[0, 1].plot(filtered_times, filtered_torques, 'g-', linewidth=1, alpha=0.7, label='Filtered Data')
        axes[0, 1].plot(filtered_times, filt_fit, 'm--', linewidth=2, label=f'Fit (f={filt_freq:.3f} Hz)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Torque')
        axes[0, 1].set_title(f'Velocity-Filtered Data\nR²={filt_r2:.3f}, RMSE={filt_rmse:.4f}, N={len(filtered_times)}')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Velocity over time (original)
        axes[1, 0].plot(original_times, original_velocities, 'b-', linewidth=1, alpha=0.7, label='Original Velocity')
        
        # Phase-specific threshold lines and filtering regions
        if phase_type == "flexion":
            axes[1, 0].axhline(y=-0.02, color='r', linestyle='--', alpha=0.7, label='-0.02 Threshold')
            axes[1, 0].fill_between(original_times, -0.02, np.max(original_velocities), alpha=0.3, color='red', label='Filtered Region (vel > -0.02)')
            filter_title = 'Flexion: Keep velocity ≤ -0.02'
        elif phase_type == "extension":
            axes[1, 0].axhline(y=0.02, color='r', linestyle='--', alpha=0.7, label='0.02 Threshold')
            axes[1, 0].fill_between(original_times, np.min(original_velocities), 0.02, alpha=0.3, color='red', label='Filtered Region (vel < 0.02)')
            filter_title = 'Extension: Keep velocity ≥ 0.02'
        else:
            axes[1, 0].axhline(y=0.02, color='r', linestyle='--', alpha=0.7, label='±0.02 Threshold')
            axes[1, 0].axhline(y=-0.02, color='r', linestyle='--', alpha=0.7)
            axes[1, 0].fill_between(original_times, -0.02, 0.02, alpha=0.3, color='red', label='Filtered Region')
            filter_title = 'Keep |velocity| ≥ 0.02'
            
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Velocity')
        axes[1, 0].set_title(f'Original Velocity Data\n{filter_title}')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Velocity over time (filtered)
        axes[1, 1].plot(filtered_times, filtered_velocities, 'g-', linewidth=1, alpha=0.7, label='Filtered Velocity')
        
        # Same threshold lines for filtered data
        if phase_type == "flexion":
            axes[1, 1].axhline(y=-0.02, color='r', linestyle='--', alpha=0.7, label='-0.02 Threshold')
            filtered_title = 'Filtered Data: velocity ≤ -0.02'
        elif phase_type == "extension":
            axes[1, 1].axhline(y=0.02, color='r', linestyle='--', alpha=0.7, label='0.02 Threshold')
            filtered_title = 'Filtered Data: velocity ≥ 0.02'
        else:
            axes[1, 1].axhline(y=0.02, color='r', linestyle='--', alpha=0.7, label='±0.02 Threshold')
            axes[1, 1].axhline(y=-0.02, color='r', linestyle='--', alpha=0.7)
            filtered_title = 'Filtered Data: |velocity| ≥ 0.02'
            
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Velocity')
        axes[1, 1].set_title(filtered_title)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # Plot 5: Velocity histogram (original)
        axes[2, 0].hist(original_velocities, bins=50, alpha=0.7, color='blue', edgecolor='black')
        
        # Phase-specific threshold lines for histograms
        if phase_type == "flexion":
            axes[2, 0].axvline(x=-0.02, color='r', linestyle='--', linewidth=2, label='-0.02 Threshold')
            hist_title_orig = 'Original Velocity Distribution\n(Flexion: will keep ≤ -0.02)'
            hist_title_filt = 'Filtered Velocity Distribution\n(Flexion: kept ≤ -0.02)'
        elif phase_type == "extension":
            axes[2, 0].axvline(x=0.02, color='r', linestyle='--', linewidth=2, label='0.02 Threshold')
            hist_title_orig = 'Original Velocity Distribution\n(Extension: will keep ≥ 0.02)'
            hist_title_filt = 'Filtered Velocity Distribution\n(Extension: kept ≥ 0.02)'
        else:
            axes[2, 0].axvline(x=0.02, color='r', linestyle='--', linewidth=2, label='±0.02 Threshold')
            axes[2, 0].axvline(x=-0.02, color='r', linestyle='--', linewidth=2)
            hist_title_orig = 'Original Velocity Distribution'
            hist_title_filt = 'Filtered Velocity Distribution'
            
        axes[2, 0].axvline(x=0, color='k', linestyle='-', alpha=0.5, label='Zero')
        axes[2, 0].set_xlabel('Velocity')
        axes[2, 0].set_ylabel('Count')
        axes[2, 0].set_title(hist_title_orig)
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].legend()
        
        # Plot 6: Velocity histogram (filtered)
        axes[2, 1].hist(filtered_velocities, bins=50, alpha=0.7, color='green', edgecolor='black')
        
        # Same threshold lines for filtered histogram
        if phase_type == "flexion":
            axes[2, 1].axvline(x=-0.02, color='r', linestyle='--', linewidth=2, label='-0.02 Threshold')
        elif phase_type == "extension":
            axes[2, 1].axvline(x=0.02, color='r', linestyle='--', linewidth=2, label='0.02 Threshold')
        else:
            axes[2, 1].axvline(x=0.02, color='r', linestyle='--', linewidth=2, label='±0.02 Threshold')
            axes[2, 1].axvline(x=-0.02, color='r', linestyle='--', linewidth=2)
            
        axes[2, 1].axvline(x=0, color='k', linestyle='-', alpha=0.5, label='Zero')
        axes[2, 1].set_xlabel('Velocity')
        axes[2, 1].set_ylabel('Count')
        axes[2, 1].set_title(hist_title_filt)
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print comparison summary
        print(f"\n{data_type} Comparison Summary:")
        print(f"Original Fit - Frequency: {orig_freq:.3f} Hz, R²: {orig_r2:.3f}, RMSE: {orig_rmse:.4f}")
        print(f"Filtered Fit - Frequency: {filt_freq:.3f} Hz, R²: {filt_r2:.3f}, RMSE: {filt_rmse:.4f}")
        print(f"Frequency change: {((filt_freq - orig_freq) / orig_freq * 100):+.1f}%")
        print(f"R² change: {((filt_r2 - orig_r2) / orig_r2 * 100):+.1f}%")
        print(f"RMSE change: {((filt_rmse - orig_rmse) / orig_rmse * 100):+.1f}%")
        
        return {
            'original': {'freq': orig_freq, 'reg': orig_reg, 'r2': orig_r2, 'rmse': orig_rmse, 'data_points': len(original_times)},
            'filtered': {'freq': filt_freq, 'reg': filt_reg, 'r2': filt_r2, 'rmse': filt_rmse, 'data_points': len(filtered_times)}
        }
        
    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found.")
        return None
    except Exception as e:
        print(f"Error processing {csv_file_path}: {e}")
        return None

def compare_mvc_fits(flexion_csv, extension_csv):
    """Compare MVC fits before and after velocity filtering"""
    print("\n" + "="*60)
    print("MVC DATA COMPARISON: ORIGINAL vs VELOCITY-FILTERED")
    print("="*60)
    
    flexion_results = compare_baseline_fits(flexion_csv, "Flexion MVC")
    extension_results = compare_baseline_fits(extension_csv, "Extension MVC")
    
    if flexion_results and extension_results:
        print(f"\nCOMBINED MVC ANALYSIS:")
        print(f"Flexion MVC - Original: {flexion_results['original']['data_points']} points, Filtered: {flexion_results['filtered']['data_points']} points")
        print(f"Extension MVC - Original: {extension_results['original']['data_points']} points, Filtered: {extension_results['filtered']['data_points']} points")
        
        total_orig = flexion_results['original']['data_points'] + extension_results['original']['data_points']
        total_filt = flexion_results['filtered']['data_points'] + extension_results['filtered']['data_points']
        print(f"Total data reduction: {total_orig} → {total_filt} ({((total_orig - total_filt) / total_orig * 100):.1f}% removed)")

if __name__ == "__main__":
    print("BASELINE AND MVC VELOCITY FILTERING COMPARISON")
    print("="*60)
    
    # Define file paths (adjust these to match your actual file names)
    baseline_files = [
        "FlexionBaseline_Data_v3.csv",
        "ExtensionBaseline_Data_v3.csv"
    ]
    
    mvc_files = [
        "FlexionMVC_Data_v3.csv",
        "ExtensionMVC_Data_v3.csv"
    ]
    
    # Compare baseline data
    print("\nBASELINE DATA COMPARISON: ORIGINAL vs VELOCITY-FILTERED")
    print("="*60)
    
    for i, baseline_file in enumerate(baseline_files):
        data_type = "Flexion Baseline" if "Flexion" in baseline_file else "Extension Baseline"
        compare_baseline_fits(baseline_file, data_type)
        if i < len(baseline_files) - 1:
            print("\n" + "-"*60)
    
    # Compare MVC data
    if len(mvc_files) >= 2:
        compare_mvc_fits(mvc_files[0], mvc_files[1])
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey findings to look for:")
    print("1. How much data is removed by velocity filtering")
    print("2. Changes in fit quality (R² and RMSE)")
    print("3. Changes in dominant frequency")
    print("4. Whether filtering removes noise or important signal")
