"""Kinematic Bicycle Model Parameter Fitting"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from tkinter import filedialog
import tkinter as tk
import os

# Configuration
MAX_STEER_ANGLE_DEG = 70  
MAX_STEER_ANGLE_RAD = np.deg2rad(MAX_STEER_ANGLE_DEG)

class BicycleModelFitter:
    """Kinematic bicycle model parameter fitter"""
    
    def __init__(self):
        self.data = None
        self.l_f = None  # Front axle to CoM distance
        self.l_r = None  # Rear axle to CoM distance
        
    def load_data(self, csv_path=None):
        """Load trajectory data"""
        if csv_path is None:
            root = tk.Tk()
            root.withdraw()
            csv_path = filedialog.askopenfilename(
                title="Select trajectory.csv",
                filetypes=[("CSV files", "*.csv")],
                initialdir="./collected_data"
            )
        
        if not csv_path:
            return False
            
        self.data = pd.read_csv(csv_path)
        print(f"Loaded {len(self.data)} data points")
        self._preprocess()
        return True
        
    def _preprocess(self):
        """Preprocess data"""
        self.data['v'] = self.data['speed'] / 3.6  # km/h -> m/s
        self.data['delta_f'] = self.data['steer'] * MAX_STEER_ANGLE_RAD
        self.data['psi_dot'] = np.deg2rad(self.data['angular_vz'])
        
        # Filter valid data points
        self.data['valid'] = (
            (self.data['v'] > 5.0) &  
            (np.abs(self.data['delta_f']) > 0.1) &  
            (np.abs(self.data['psi_dot']) > 0.1)
        )
        
        valid_count = self.data['valid'].sum()
        print(f"Valid data points: {valid_count}/{len(self.data)}")
        
    def fit_parameters(self):
        """Fit l_f and l_r parameters"""
        valid_data = self.data[self.data['valid']].copy()
        
        if len(valid_data) < 10:
            print("Error: Not enough valid data points")
            return False
            
        def model_residuals(params):
            l_f, l_r = params
            beta = np.arctan((l_r / (l_f + l_r)) * np.tan(valid_data['delta_f'].values))
            psi_dot_pred = (valid_data['v'].values / l_r) * np.sin(beta)
            return psi_dot_pred - valid_data['psi_dot'].values
            
        initial_guess = [1.4, 1.4]
        lower_bounds = [0.8, 0.8]
        upper_bounds = [2.0, 2.0]
        
        print("\nFitting parameters...")
        result = least_squares(
            model_residuals, 
            initial_guess,
            bounds=(lower_bounds, upper_bounds),
            verbose=1
        )
        
        self.l_f = result.x[0]
        self.l_r = result.x[1]
        
        residuals = result.fun
        rmse = np.sqrt(np.mean(residuals**2))
        
        print("\n" + "="*50)
        print("FITTING RESULTS")
        print("="*50)
        print(f"l_f (front axle to CoM): {self.l_f:.3f} m")
        print(f"l_r (rear axle to CoM):  {self.l_r:.3f} m")
        print(f"L (wheelbase):           {self.l_f + self.l_r:.3f} m")
        print(f"RMSE:                    {rmse:.4f} rad/s")
        print(f"Success:                 {result.success}")
        print("="*50)
        
        return True
        
    def validate_and_plot(self):
        """Validate results and plot"""
        if self.l_f is None or self.l_r is None:
            print("Please fit parameters first")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        test_data = self.data.copy()
        
        # Model prediction
        beta = np.arctan((self.l_r / (self.l_f + self.l_r)) * np.tan(test_data['delta_f']))
        psi_dot_pred = (test_data['v'] / self.l_r) * np.sin(beta)
        
        # Plot 1: Time series comparison
        ax = axes[0, 0]
        time = test_data.index * 0.05
        ax.plot(time, test_data['psi_dot'], 'b-', label='Measured', alpha=0.6)
        ax.plot(time, psi_dot_pred, 'r--', label='Predicted', alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Yaw Rate (rad/s)')
        ax.set_title('Yaw Rate Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        ax = axes[0, 1]
        valid_mask = test_data['valid']
        ax.scatter(test_data.loc[valid_mask, 'psi_dot'], 
                  psi_dot_pred[valid_mask], alpha=0.5, s=1)
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel('Measured Yaw Rate (rad/s)')
        ax.set_ylabel('Predicted Yaw Rate (rad/s)')
        ax.set_title('Measured vs Predicted')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        ax = axes[1, 0]
        error = psi_dot_pred - test_data['psi_dot']
        valid_error = error[valid_mask]
        ax.hist(valid_error, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Prediction Error (rad/s)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error Distribution (RMSE: {np.sqrt(np.mean(valid_error**2)):.4f})')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Parameters
        ax = axes[1, 1]
        params = ['l_f', 'l_r', 'L']
        values = [self.l_f, self.l_r, self.l_f + self.l_r]
        colors = ['blue', 'red', 'green']
        bars = ax.bar(params, values, color=colors, alpha=0.7)
        ax.set_ylabel('Distance (m)')
        ax.set_title('Identified Parameters')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
    def save_results(self, filename='bicycle_params.txt'):
        """Save results to file"""
        if self.l_f is None or self.l_r is None:
            print("No results to save")
            return
            
        with open(filename, 'w') as f:
            f.write("="*50 + "\n")
            f.write("KINEMATIC BICYCLE MODEL PARAMETERS\n")
            f.write("="*50 + "\n\n")
            f.write(f"l_f (front axle to CoM): {self.l_f:.4f} m\n")
            f.write(f"l_r (rear axle to CoM):  {self.l_r:.4f} m\n")
            f.write(f"L (wheelbase):           {self.l_f + self.l_r:.4f} m\n")
            f.write(f"\nMax steering angle:      {MAX_STEER_ANGLE_DEG}°\n")
            f.write(f"Data points used:        {self.data['valid'].sum()}\n")
            
        print(f"Results saved to {filename}")

def main():
    print("\n" + "="*60)
    print(" KINEMATIC BICYCLE MODEL PARAMETER FITTING")
    print("="*60 + "\n")
    
    fitter = BicycleModelFitter()
    
    print("Step 1: Load data")
    if not fitter.load_data():
        print("Failed to load data")
        return
        
    print("\nStep 2: Fit parameters")
    if not fitter.fit_parameters():
        print("Failed to fit parameters")
        return
        
    print("\nStep 3: Validate and visualize")
    fitter.validate_and_plot()
    
    print("\nStep 4: Save results")
    fitter.save_results()
    
    print("\nDone!")

if __name__ == "__main__":
    main()