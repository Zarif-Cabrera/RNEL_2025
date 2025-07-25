#!/usr/bin/env python3
"""
Example usage of the MVC Curve Fitting Analysis tool

This script demonstrates how to use the MVCCurveFitter class to analyze
recorded MVC data and find the best curve fit.
"""

from MVC_Curve_Fitting_Analysis import MVCCurveFitter
import os

def analyze_mvc_file(csv_file_path):
    """
    Analyze a single MVC CSV file
    
    Args:
        csv_file_path: Path to the CSV file containing MVC data
    """
    print(f"Analyzing MVC file: {csv_file_path}")
    print("=" * 60)
    
    # Create the curve fitter
    fitter = MVCCurveFitter(csv_file_path, velocity_threshold=0.03)
    
    # Load the data
    if not fitter.load_data():
        print("Failed to load data. Exiting.")
        return
    
    # Filter by velocity (for flexion data)
    fitter.filter_by_velocity(phase_type="flexion")
    
    # Fit polynomial models (test degrees 1-10)
    print("\nFitting polynomial models...")
    poly_results = fitter.fit_polynomial_models(max_degree=10)
    
    # Fit custom function models
    print("\nFitting custom function models...")
    custom_results = fitter.fit_custom_functions()
    
    # Plot the results
    print("\nGenerating plots...")
    fitter.plot_results(custom_results, save_plot=True)
    
    # Export the fitted model
    print("\nExporting fitted model...")
    predict_function = fitter.export_fitted_model()
    
    # Demonstrate prediction
    if predict_function:
        print("\nExample predictions:")
        import numpy as np
        
        # Predict at some time points
        test_times = np.array([0, 1, 2, 3, 4, 5])  # normalized time
        predictions = predict_function(test_times)
        
        for t, pred in zip(test_times, predictions):
            print(f"  Time {t:.1f}s: Predicted torque = {pred:.4f}")
    
    print(f"\nAnalysis complete for {csv_file_path}")
    print("=" * 60)

def analyze_all_mvc_files():
    """
    Analyze all MVC CSV files in the current directory
    """
    # Look for MVC CSV files
    mvc_files = []
    for file in os.listdir('.'):
        if file.endswith('.csv') and 'MVC' in file.upper():
            mvc_files.append(file)
    
    if not mvc_files:
        print("No MVC CSV files found in current directory.")
        print("Looking for files with 'MVC' in the name...")
        return
    
    print(f"Found {len(mvc_files)} MVC CSV files:")
    for i, file in enumerate(mvc_files, 1):
        print(f"  {i}. {file}")
    
    # Analyze each file
    for file in mvc_files:
        try:
            analyze_mvc_file(file)
            print("\n" + "="*80 + "\n")
        except Exception as e:
            print(f"Error analyzing {file}: {e}")
            continue

def main():
    """Main function"""
    print("MVC Curve Fitting Analysis - Example Usage")
    print("=" * 80)
    
    # Check if specific files exist (based on your code's output file names)
    flexion_mvc_file = "FlexionMVC_Data_v3.csv"
    extension_mvc_file = "ExtensionMVC_Data_v3.csv"
    
    files_to_analyze = []
    
    if os.path.exists(flexion_mvc_file):
        files_to_analyze.append(flexion_mvc_file)
    
    if os.path.exists(extension_mvc_file):
        files_to_analyze.append(extension_mvc_file)
    
    if files_to_analyze:
        print(f"Found specific MVC files from your application:")
        for file in files_to_analyze:
            print(f"  - {file}")
        
        # Analyze each file
        for file in files_to_analyze:
            analyze_mvc_file(file)
            print("\n" + "="*80 + "\n")
    else:
        print("No specific MVC files found. Searching for any MVC CSV files...")
        analyze_all_mvc_files()

if __name__ == "__main__":
    main()
