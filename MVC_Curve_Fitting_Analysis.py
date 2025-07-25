import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit
import argparse
import os

class MVCCurveFitter:
    def __init__(self, csv_file_path, velocity_threshold=0.03):
        """
        Initialize the MVC Curve Fitter
        
        Args:
            csv_file_path: Path to the CSV file containing MVC data
            velocity_threshold: Velocity threshold for filtering (default 0.03)
        """
        self.csv_file_path = csv_file_path
        self.velocity_threshold = velocity_threshold
        self.df_raw = None
        self.df_filtered = None
        self.best_model = None
        self.best_degree = None
        self.best_r2 = None
        self.best_rmse = None
        
    def load_data(self):
        """Load MVC data from CSV file"""
        try:
            self.df_raw = pd.read_csv(self.csv_file_path)
            print(f"Loaded data from {self.csv_file_path}")
            print(f"Raw data shape: {self.df_raw.shape}")
            print(f"Columns: {list(self.df_raw.columns)}")
            
            # Check if required columns exist
            required_columns = ['time', 'torque', 'velocity']
            missing_columns = [col for col in required_columns if col not in self.df_raw.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def filter_by_velocity(self, phase_type="flexion"):
        """
        Filter DataFrame based on phase-specific velocity criteria
        
        Args:
            phase_type: 'flexion' or 'extension' for phase-specific filtering
        """
        if 'velocity' not in self.df_raw.columns:
            print("Warning: No velocity column found. Skipping velocity filtering.")
            self.df_filtered = self.df_raw.copy()
            return
            
        initial_count = len(self.df_raw)
        
        # Phase-specific filtering logic (same as in your main code)
        if phase_type.lower() == 'flexion':
            # For flexion: keep velocity <= -threshold (negative velocities)
            self.df_filtered = self.df_raw[self.df_raw['velocity'] <= -self.velocity_threshold]
            filter_description = f"velocity > -{self.velocity_threshold}"
        elif phase_type.lower() == 'extension':
            # For extension: keep velocity >= threshold (positive velocities)  
            self.df_filtered = self.df_raw[self.df_raw['velocity'] >= self.velocity_threshold]
            filter_description = f"velocity < {self.velocity_threshold}"
        else:
            # Fallback to symmetric filtering
            self.df_filtered = self.df_raw[
                (self.df_raw['velocity'] <= -self.velocity_threshold) | 
                (self.df_raw['velocity'] >= self.velocity_threshold)
            ]
            filter_description = f"-{self.velocity_threshold} < velocity < {self.velocity_threshold}"
        
        final_count = len(self.df_filtered)
        removed_count = initial_count - final_count
        removed_percentage = (removed_count / initial_count) * 100 if initial_count > 0 else 0
        
        print(f"\nVelocity filtering results ({phase_type}):")
        print(f"  Initial data points: {initial_count}")
        print(f"  Final data points: {final_count}")
        print(f"  Removed data points: {removed_count} ({removed_percentage:.1f}%)")
        print(f"  Filter criterion: removed where {filter_description}")
        
        # Normalize time to start at 0 (same as in your main code)
        if len(self.df_filtered) > 0:
            time_raw = self.df_filtered["time"].values
            self.df_filtered = self.df_filtered.copy()  # Avoid SettingWithCopyWarning
            self.df_filtered["time_normalized"] = time_raw - time_raw[0]
    
    def fit_polynomial_models(self, max_degree=10):
        """
        Fit polynomial models of different degrees and find the best one
        
        Args:
            max_degree: Maximum polynomial degree to test
        """
        if self.df_filtered is None or len(self.df_filtered) == 0:
            print("Error: No filtered data available for fitting")
            return
            
        X = self.df_filtered["time_normalized"].values.reshape(-1, 1)
        y = self.df_filtered["torque"].values
        
        print(f"\nFitting polynomial models (degrees 1-{max_degree})...")
        
        best_r2 = -np.inf
        results = []
        
        for degree in range(1, max_degree + 1):
            # Create polynomial pipeline
            poly_model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
            
            # Fit the model
            poly_model.fit(X, y)
            
            # Make predictions
            y_pred = poly_model.predict(X)
            
            # Calculate metrics
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            results.append({
                'degree': degree,
                'r2': r2,
                'rmse': rmse,
                'model': poly_model
            })
            
            print(f"  Degree {degree:2d}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
            
            # Track best model
            if r2 > best_r2:
                best_r2 = r2
                self.best_model = poly_model
                self.best_degree = degree
                self.best_r2 = r2
                self.best_rmse = rmse
        
        print(f"\nBest model: Polynomial degree {self.best_degree}")
        print(f"Best R²: {self.best_r2:.4f}")
        print(f"Best RMSE: {self.best_rmse:.4f}")
        
        return results
    
    def fit_custom_functions(self):
        """
        Fit custom function models (exponential, logarithmic, power, etc.)
        """
        if self.df_filtered is None or len(self.df_filtered) == 0:
            print("Error: No filtered data available for fitting")
            return
            
        X = self.df_filtered["time_normalized"].values
        y = self.df_filtered["torque"].values
        
        print(f"\nFitting custom function models...")
        
        custom_results = []
        
        # Define custom functions
        def exponential_func(x, a, b, c):
            return a * np.exp(b * x) + c
            
        def logarithmic_func(x, a, b, c):
            # Avoid log of zero or negative numbers
            return a * np.log(np.maximum(b * x + 1, 1e-10)) + c
            
        def power_func(x, a, b, c):
            return a * np.power(x + 1, b) + c
            
        def gaussian_func(x, a, b, c, d):
            return a * np.exp(-(x - b)**2 / (2 * c**2)) + d
        
        # Try fitting each function
        functions = [
            ('Exponential', exponential_func, [1, 0.1, 0]),
            ('Logarithmic', logarithmic_func, [1, 1, 0]),
            ('Power', power_func, [1, 0.5, 0]),
            ('Gaussian', gaussian_func, [1, np.mean(X), np.std(X), 0])
        ]
        
        for name, func, initial_guess in functions:
            try:
                # Fit the function
                popt, pcov = curve_fit(func, X, y, p0=initial_guess, maxfev=5000)
                
                # Make predictions
                y_pred = func(X, *popt)
                
                # Calculate metrics
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                custom_results.append({
                    'name': name,
                    'function': func,
                    'parameters': popt,
                    'r2': r2,
                    'rmse': rmse
                })
                
                print(f"  {name:12s}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
                
                # Check if this is better than polynomial
                if r2 > self.best_r2:
                    print(f"    ^ Better than best polynomial (R² = {self.best_r2:.4f})")
                    
            except Exception as e:
                print(f"  {name:12s}: Failed to fit ({str(e)[:50]}...)")
        
        return custom_results
    
    def plot_results(self, custom_results=None, save_plot=True):
        """
        Plot the original data, filtered data, and fitted curves
        
        Args:
            custom_results: Results from custom function fitting
            save_plot: Whether to save the plot to file
        """
        if self.df_filtered is None or self.best_model is None:
            print("Error: No data or model available for plotting")
            return
            
        # Create the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MVC Curve Fitting Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Raw vs Filtered Data
        ax1 = axes[0, 0]
        ax1.plot(self.df_raw["time"].values, self.df_raw["torque"].values, 'b-', alpha=0.3, linewidth=1, label='Raw Data')
        if len(self.df_filtered) > 0:
            # Use original time for plotting
            original_times = self.df_filtered["time"].values
            ax1.plot(original_times, self.df_filtered["torque"].values, 'r-', linewidth=2, label='Velocity Filtered Data')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Torque')
        ax1.set_title('Raw vs Velocity-Filtered Data')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Best Polynomial Fit
        ax2 = axes[0, 1]
        X_normalized = self.df_filtered["time_normalized"].values.reshape(-1, 1)
        y_actual = self.df_filtered["torque"].values
        y_pred = self.best_model.predict(X_normalized)
        
        ax2.plot(self.df_filtered["time_normalized"].values, y_actual, 'b-', linewidth=2, alpha=0.7, label='Filtered Data')
        ax2.plot(self.df_filtered["time_normalized"].values, y_pred, 'r--', linewidth=3, label=f'Polynomial Fit (degree {self.best_degree})')
        ax2.set_xlabel('Normalized Time (s)')
        ax2.set_ylabel('Torque')
        ax2.set_title(f'Best Polynomial Fit\nR² = {self.best_r2:.4f}, RMSE = {self.best_rmse:.4f}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Residuals
        ax3 = axes[1, 0]
        residuals = y_actual - y_pred
        ax3.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Predicted Torque')
        ax3.set_ylabel('Residuals (Actual - Predicted)')
        ax3.set_title('Residual Plot')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model Comparison
        ax4 = axes[1, 1]
        if custom_results:
            # Plot best custom function if available
            best_custom = max(custom_results, key=lambda x: x['r2'])
            X_plot = self.df_filtered["time_normalized"].values
            y_custom_pred = best_custom['function'](X_plot, *best_custom['parameters'])
            
            ax4.plot(X_plot, y_actual, 'b-', linewidth=2, alpha=0.7, label='Filtered Data')
            ax4.plot(X_plot, y_pred, 'r--', linewidth=2, label=f'Polynomial (R²={self.best_r2:.3f})')
            ax4.plot(X_plot, y_custom_pred, 'g:', linewidth=2, 
                    label=f'{best_custom["name"]} (R²={best_custom["r2"]:.3f})')
            ax4.set_title('Model Comparison')
        else:
            # Just show polynomial fit details
            ax4.plot(X_normalized.flatten(), y_actual, 'b-', linewidth=2, alpha=0.7, label='Filtered Data')
            ax4.plot(X_normalized.flatten(), y_pred, 'r--', linewidth=3, label=f'Polynomial Fit')
            ax4.set_title('Polynomial Model Details')
            
        ax4.set_xlabel('Normalized Time (s)')
        ax4.set_ylabel('Torque')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        if save_plot:
            # Generate filename based on input CSV
            base_name = os.path.splitext(os.path.basename(self.csv_file_path))[0]
            plot_filename = f"{base_name}_curve_fitting_analysis.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved as: {plot_filename}")
        
        plt.show()
    
    def export_fitted_model(self):
        """
        Export the fitted model parameters and create a function for prediction
        """
        if self.best_model is None:
            print("Error: No fitted model available for export")
            return None
            
        # Extract polynomial coefficients
        poly_features = self.best_model.named_steps['poly']
        linear_model = self.best_model.named_steps['linear']
        
        coefficients = linear_model.coef_
        intercept = linear_model.intercept_
        
        print(f"\nFitted Polynomial Model (degree {self.best_degree}):")
        print(f"Intercept: {intercept:.6f}")
        print("Coefficients:")
        for i, coef in enumerate(coefficients):
            if i == 0:
                continue  # Skip the constant term (already in intercept)
            print(f"  x^{i}: {coef:.6f}")
        
        # Create prediction function
        def predict_torque(time_normalized):
            """
            Predict torque values for given normalized time points
            
            Args:
                time_normalized: Array of normalized time values (starting from 0)
                
            Returns:
                Array of predicted torque values
            """
            time_reshaped = np.array(time_normalized).reshape(-1, 1)
            return self.best_model.predict(time_reshaped)
        
        # Save model parameters to file
        base_name = os.path.splitext(os.path.basename(self.csv_file_path))[0]
        model_filename = f"{base_name}_fitted_model_params.txt"
        
        with open(model_filename, 'w') as f:
            f.write(f"MVC Curve Fitting Results\n")
            f.write(f"========================\n\n")
            f.write(f"Input file: {self.csv_file_path}\n")
            f.write(f"Velocity threshold: {self.velocity_threshold}\n")
            f.write(f"Raw data points: {len(self.df_raw)}\n")
            f.write(f"Filtered data points: {len(self.df_filtered)}\n\n")
            f.write(f"Best Model: Polynomial degree {self.best_degree}\n")
            f.write(f"R² score: {self.best_r2:.6f}\n")
            f.write(f"RMSE: {self.best_rmse:.6f}\n\n")
            f.write(f"Model Parameters:\n")
            f.write(f"Intercept: {intercept:.6f}\n")
            for i, coef in enumerate(coefficients):
                if i == 0:
                    continue
                f.write(f"x^{i} coefficient: {coef:.6f}\n")
        
        print(f"Model parameters saved to: {model_filename}")
        
        return predict_torque

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Analyze MVC data and fit curves')
    parser.add_argument('csv_file', help='Path to the MVC CSV file')
    parser.add_argument('--velocity_threshold', type=float, default=0.03, 
                       help='Velocity threshold for filtering (default: 0.03)')
    parser.add_argument('--phase_type', choices=['flexion', 'extension'], default='flexion',
                       help='Phase type for velocity filtering (default: flexion)')
    parser.add_argument('--max_degree', type=int, default=10,
                       help='Maximum polynomial degree to test (default: 10)')
    parser.add_argument('--no_custom', action='store_true',
                       help='Skip custom function fitting')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip plotting')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' not found")
        return
    
    # Create fitter instance
    fitter = MVCCurveFitter(args.csv_file, args.velocity_threshold)
    
    # Load and process data
    if not fitter.load_data():
        return
    
    # Filter data
    fitter.filter_by_velocity(args.phase_type)
    
    # Fit polynomial models
    poly_results = fitter.fit_polynomial_models(args.max_degree)
    
    # Fit custom functions
    custom_results = None
    if not args.no_custom:
        custom_results = fitter.fit_custom_functions()
    
    # Plot results
    if not args.no_plot:
        fitter.plot_results(custom_results)
    
    # Export model
    predict_function = fitter.export_fitted_model()
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()
