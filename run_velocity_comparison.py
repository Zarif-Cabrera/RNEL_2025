#!/usr/bin/env python3
"""
Test script to run the velocity filtering comparison analysis.
This script will look for the baseline and MVC CSV files and generate comparison plots.
"""

import os
import sys

# Add the current directory to the path to import our comparison module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baseline_velocity_filter_comparison import compare_baseline_fits, compare_mvc_fits

def main():
    """Main function to run the velocity filtering comparison"""
    print("VELOCITY FILTERING COMPARISON ANALYSIS")
    print("="*60)
    print("This script compares sinusoidal fits before and after velocity filtering.")
    print("It will look for CSV files in the current directory.")
    print()
    
    # Define expected file names
    baseline_files = [
        "FlexionBaseline_Data_v3.csv",
        "ExtensionBaseline_Data_v3.csv"
    ]
    
    mvc_files = [
        "FlexionMVC_Data_v3.csv", 
        "ExtensionMVC_Data_v3.csv"
    ]
    
    # Check which files exist
    existing_baseline_files = []
    existing_mvc_files = []
    
    for file in baseline_files:
        if os.path.exists(file):
            existing_baseline_files.append(file)
            print(f"✓ Found baseline file: {file}")
        else:
            print(f"✗ Missing baseline file: {file}")
    
    for file in mvc_files:
        if os.path.exists(file):
            existing_mvc_files.append(file)
            print(f"✓ Found MVC file: {file}")
        else:
            print(f"✗ Missing MVC file: {file}")
    
    print()
    
    # Process baseline files
    if existing_baseline_files:
        print("BASELINE DATA ANALYSIS")
        print("-" * 40)
        for i, baseline_file in enumerate(existing_baseline_files):
            data_type = "Flexion Baseline" if "Flexion" in baseline_file else "Extension Baseline"
            print(f"\nAnalyzing {data_type}...")
            compare_baseline_fits(baseline_file, data_type)
            
            if i < len(existing_baseline_files) - 1:
                print("\n" + "-" * 60)
    else:
        print("No baseline CSV files found. Please run the GUI first to collect baseline data.")
    
    # Process MVC files
    if len(existing_mvc_files) >= 2:
        print("\n" + "=" * 60)
        print("MVC DATA ANALYSIS")
        print("-" * 40)
        compare_mvc_fits(existing_mvc_files[0], existing_mvc_files[1])
    elif len(existing_mvc_files) == 1:
        print(f"\nOnly one MVC file found: {existing_mvc_files[0]}")
        print("Need both flexion and extension MVC files for comparison.")
        data_type = "Flexion MVC" if "Flexion" in existing_mvc_files[0] else "Extension MVC"
        compare_baseline_fits(existing_mvc_files[0], data_type)
    else:
        print("\nNo MVC CSV files found. Please run the GUI first to collect MVC data.")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    if not existing_baseline_files and not existing_mvc_files:
        print("\nTo generate data files:")
        print("1. Run the main GUI: python GUI_TorqueTaskSplit(EF)_PyQtGraph_v3.py")
        print("2. Click 'Set Flexion Baseline' and 'Set Extension Baseline'")
        print("3. Click 'Set Flexion MVC' and 'Set Extension MVC'")  
        print("4. The CSV files will be automatically saved")
        print("5. Run this script again to see the comparison")

if __name__ == "__main__":
    main()
