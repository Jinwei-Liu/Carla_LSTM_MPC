"""CARLA Data Visualization Tool"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import json
import h5py
from datetime import datetime
import seaborn as sns
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

class DataVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CARLA Data Visualization & Analysis Tool")
        self.root.geometry("1400x900")
        
        # Data storage
        self.data = None
        self.current_episode = None
        self.episodes = []
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup user interface"""
        # Top control bar
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Load data button
        ttk.Button(control_frame, text="Load Data", 
                  command=self.load_data).pack(side=tk.LEFT, padx=5)
        
        # Episode selection
        ttk.Label(control_frame, text="Episode:").pack(side=tk.LEFT, padx=5)
        self.episode_var = tk.StringVar()
        self.episode_combo = ttk.Combobox(control_frame, textvariable=self.episode_var,
                                          state="readonly", width=20)
        self.episode_combo.pack(side=tk.LEFT, padx=5)
        self.episode_combo.bind('<<ComboboxSelected>>', self.on_episode_change)
        
        # Analysis type selection
        ttk.Label(control_frame, text="Analysis Type:").pack(side=tk.LEFT, padx=20)
        
        analysis_buttons = [
            ("Trajectory", self.show_trajectory),
            ("Speed Analysis", self.show_speed_analysis),
            ("Control Analysis", self.show_control_analysis),
            ("Acceleration", self.show_acceleration_analysis),
            ("Statistics", self.show_statistics),
            ("3D Trajectory", self.show_3d_trajectory),
            ("Driving Behavior", self.show_driving_behavior),
            ("Replay", self.show_replay)
        ]
        
        for text, command in analysis_buttons:
            ttk.Button(control_frame, text=text, 
                      command=command).pack(side=tk.LEFT, padx=2)
        
        # Main display area
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Please load data...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_data(self):
        """Load data"""
        # Select data folder
        folder_path = filedialog.askdirectory(
            title="Select Data Folder",
            initialdir="./collected_data"
        )
        
        if not folder_path:
            return
        
        # Find all episodes
        self.episodes = []
        for item in os.listdir(folder_path):
            if item.startswith('episode_'):
                episode_path = os.path.join(folder_path, item)
                csv_path = os.path.join(episode_path, 'trajectory.csv')
                if os.path.exists(csv_path):
                    self.episodes.append({
                        'name': item,
                        'path': episode_path,
                        'csv': csv_path
                    })
        
        if not self.episodes:
            self.status_var.set("No data files found")
            return
        
        # Update dropdown
        episode_names = [ep['name'] for ep in self.episodes]
        self.episode_combo['values'] = episode_names
        self.episode_combo.set(episode_names[0])
        
        # Load first episode
        self.load_episode(self.episodes[0])
        
        self.status_var.set(f"Loaded {len(self.episodes)} episodes")
    
    def load_episode(self, episode_info):
        """Load single episode data"""
        try:
            self.data = pd.read_csv(episode_info['csv'])
            self.current_episode = episode_info
            
            # Parse JSON fields
            if 'nearby_vehicles' in self.data.columns:
                self.data['nearby_vehicles_parsed'] = self.data['nearby_vehicles'].apply(
                    lambda x: json.loads(x) if pd.notna(x) else []
                )
            
            # Calculate additional metrics
            self.calculate_metrics()
            
            self.status_var.set(f"Loaded: {episode_info['name']} - {len(self.data)} frames")
            
        except Exception as e:
            self.status_var.set(f"Loading failed: {e}")
    
    def calculate_metrics(self):
        """Calculate additional analysis metrics"""
        if self.data is None:
            return
        
        # Calculate time difference
        self.data['time_diff'] = self.data['timestamp'].diff()
        
        # Calculate jerk
        self.data['jerk_x'] = self.data['ax'].diff() / self.data['time_diff']
        self.data['jerk_y'] = self.data['ay'].diff() / self.data['time_diff']
        
        # Calculate total acceleration
        self.data['total_acceleration'] = np.sqrt(
            self.data['ax']**2 + self.data['ay']**2
        )
        
        # Calculate steering rate
        self.data['steer_rate'] = self.data['steer'].diff() / self.data['time_diff']
        
        # Calculate traveled distance
        self.data['distance'] = np.sqrt(
            self.data['x'].diff()**2 + self.data['y'].diff()**2
        ).cumsum()
    
    def on_episode_change(self, event=None):
        """Episode selection change"""
        selected = self.episode_var.get()
        for ep in self.episodes:
            if ep['name'] == selected:
                self.load_episode(ep)
                break
    
    def clear_display(self):
        """Clear display area"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
    def show_trajectory(self):
        """Show trajectory analysis"""
        if self.data is None:
            return
        
        self.clear_display()
        
        # Create figure
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # 2D trajectory
        ax1 = fig.add_subplot(gs[:, 0])
        scatter = ax1.scatter(self.data['x'], self.data['y'], 
                            c=self.data['speed'], cmap='viridis', 
                            s=2, alpha=0.6)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Vehicle Trajectory (Color = Speed)')
        ax1.axis('equal')
        plt.colorbar(scatter, ax=ax1, label='Speed (km/h)')
        
        # Add start and end markers
        ax1.plot(self.data['x'].iloc[0], self.data['y'].iloc[0], 
                'go', markersize=10, label='Start')
        ax1.plot(self.data['x'].iloc[-1], self.data['y'].iloc[-1], 
                'ro', markersize=10, label='End')
        ax1.legend()
        
        # Speed vs time
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.data.index * 0.05, self.data['speed'], 'b-', linewidth=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed (km/h)')
        ax2.set_title('Speed Profile')
        ax2.grid(True, alpha=0.3)
        
        # Steering vs time
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(self.data.index * 0.05, self.data['steer'], 'r-', linewidth=1)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Steering Angle')
        ax3.set_title('Steering Profile')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_speed_analysis(self):
        """Detailed speed analysis"""
        if self.data is None:
            return
        
        self.clear_display()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # Speed distribution histogram
        axes[0, 0].hist(self.data['speed'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Speed (km/h)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Speed Distribution')
        axes[0, 0].axvline(self.data['speed'].mean(), color='r', 
                           linestyle='--', label=f'Mean: {self.data["speed"].mean():.1f} km/h')
        axes[0, 0].legend()
        
        # Speed vs throttle/brake
        time = self.data.index * 0.05
        ax1 = axes[0, 1]
        ax1.plot(time, self.data['speed'], 'b-', label='Speed', linewidth=1)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Speed (km/h)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(time, self.data['throttle'], 'g-', label='Throttle', alpha=0.6)
        ax2.plot(time, self.data['brake'], 'r-', label='Brake', alpha=0.6)
        ax2.set_ylabel('Control Input', color='k')
        ax2.legend(loc='upper right')
        ax1.set_title('Speed vs Control Inputs')
        
        # Acceleration analysis
        axes[1, 0].plot(time, self.data['ax'], label='Longitudinal', alpha=0.7)
        axes[1, 0].plot(time, self.data['ay'], label='Lateral', alpha=0.7)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Acceleration (m/s²)')
        axes[1, 0].set_title('Acceleration Profile')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Speed-acceleration phase diagram
        axes[1, 1].scatter(self.data['speed'], self.data['ax'], 
                          alpha=0.5, s=1, c=time, cmap='viridis')
        axes[1, 1].set_xlabel('Speed (km/h)')
        axes[1, 1].set_ylabel('Longitudinal Acceleration (m/s²)')
        axes[1, 1].set_title('Speed-Acceleration Phase Diagram')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_control_analysis(self):
        """Control input analysis"""
        if self.data is None:
            return
        
        self.clear_display()
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        time = self.data.index * 0.05
        
        # Throttle analysis
        axes[0, 0].plot(time, self.data['throttle'], 'g-', linewidth=1)
        axes[0, 0].fill_between(time, 0, self.data['throttle'], alpha=0.3, color='g')
        axes[0, 0].set_ylabel('Throttle')
        axes[0, 0].set_title('Throttle Input')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Brake analysis
        axes[0, 1].plot(time, self.data['brake'], 'r-', linewidth=1)
        axes[0, 1].fill_between(time, 0, self.data['brake'], alpha=0.3, color='r')
        axes[0, 1].set_ylabel('Brake')
        axes[0, 1].set_title('Brake Input')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Steering analysis
        axes[1, 0].plot(time, self.data['steer'], 'b-', linewidth=1)
        axes[1, 0].fill_between(time, 0, self.data['steer'], alpha=0.3, color='b')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Steering Angle')
        axes[1, 0].set_title('Steering Input')
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Steering rate
        if 'steer_rate' in self.data.columns:
            axes[1, 1].plot(time, self.data['steer_rate'], 'orange', linewidth=1)
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Steering Rate')
            axes[1, 1].set_title('Steering Change Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Combined control inputs
        axes[2, 0].plot(time, self.data['throttle'], 'g-', label='Throttle', alpha=0.7)
        axes[2, 0].plot(time, self.data['brake'], 'r-', label='Brake', alpha=0.7)
        axes[2, 0].plot(time, np.abs(self.data['steer']), 'b-', label='|Steering|', alpha=0.7)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Control Value')
        axes[2, 0].set_title('All Control Inputs')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Control smoothness analysis
        throttle_smoothness = np.std(np.diff(self.data['throttle']))
        brake_smoothness = np.std(np.diff(self.data['brake']))
        steer_smoothness = np.std(np.diff(self.data['steer']))
        
        axes[2, 1].bar(['Throttle', 'Brake', 'Steering'], 
                      [throttle_smoothness, brake_smoothness, steer_smoothness],
                      color=['g', 'r', 'b'], alpha=0.7)
        axes[2, 1].set_ylabel('Control Change STD')
        axes[2, 1].set_title('Control Smoothness (Lower is Smoother)')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_acceleration_analysis(self):
        """Detailed acceleration analysis"""
        if self.data is None:
            return
        
        self.clear_display()
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        time = self.data.index * 0.05
        
        # Longitudinal acceleration
        axes[0, 0].plot(time, self.data['ax'], 'b-', linewidth=1)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Longitudinal Acc. (m/s²)')
        axes[0, 0].set_title('Longitudinal Acceleration')
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Lateral acceleration
        axes[0, 1].plot(time, self.data['ay'], 'r-', linewidth=1)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Lateral Acc. (m/s²)')
        axes[0, 1].set_title('Lateral Acceleration')
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total acceleration
        if 'total_acceleration' in self.data.columns:
            axes[0, 2].plot(time, self.data['total_acceleration'], 'g-', linewidth=1)
            axes[0, 2].set_xlabel('Time (s)')
            axes[0, 2].set_ylabel('Total Acc. (m/s²)')
            axes[0, 2].set_title('Total Acceleration')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Acceleration distribution
        axes[1, 0].hist(self.data['ax'], bins=50, alpha=0.7, color='b', edgecolor='black')
        axes[1, 0].set_xlabel('Longitudinal Acc. (m/s²)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Longitudinal Acc. Distribution')
        
        axes[1, 1].hist(self.data['ay'], bins=50, alpha=0.7, color='r', edgecolor='black')
        axes[1, 1].set_xlabel('Lateral Acc. (m/s²)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Lateral Acc. Distribution')
        
        # Comfort metrics
        comfort_data = {
            'Metric': ['Max Long. Acc.', 'Max Lat. Acc.', 'Avg. Jerk'],
            'Value': [
                self.data['ax'].abs().max(),
                self.data['ay'].abs().max(),
                self.data[['jerk_x', 'jerk_y']].abs().mean().mean() if 'jerk_x' in self.data.columns else 0
            ]
        }
        
        axes[1, 2].bar(comfort_data['Metric'], comfort_data['Value'], 
                      color=['b', 'r', 'orange'], alpha=0.7)
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].set_title('Comfort Metrics')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_statistics(self):
        """Show statistics report"""
        if self.data is None:
            return
        
        self.clear_display()
        
        # Create statistics frame
        stats_frame = ttk.Frame(self.main_frame)
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget
        text = tk.Text(stats_frame, wrap=tk.WORD, font=('Courier', 10))
        text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text.yview)
        
        # Generate statistics report
        report = self.generate_statistics_report()
        text.insert(tk.END, report)
        text.config(state=tk.DISABLED)
    
    def generate_statistics_report(self):
        """Generate statistics report"""
        report = []
        report.append("=" * 60)
        report.append("DRIVING DATA STATISTICS REPORT")
        report.append("=" * 60)
        report.append(f"\nEpisode: {self.current_episode['name']}")
        report.append(f"Data Frames: {len(self.data)}")
        report.append(f"Total Duration: {len(self.data) * 0.05:.2f} seconds")
        
        if 'distance' in self.data.columns:
            report.append(f"Total Distance Traveled: {self.data['distance'].iloc[-1]:.2f} meters")
        
        report.append("\n" + "-" * 40)
        report.append("SPEED STATISTICS")
        report.append("-" * 40)
        report.append(f"Average Speed: {self.data['speed'].mean():.2f} km/h")
        report.append(f"Maximum Speed: {self.data['speed'].max():.2f} km/h")
        report.append(f"Minimum Speed: {self.data['speed'].min():.2f} km/h")
        report.append(f"Speed STD: {self.data['speed'].std():.2f} km/h")
        
        report.append("\n" + "-" * 40)
        report.append("ACCELERATION STATISTICS")
        report.append("-" * 40)
        report.append(f"Longitudinal Acc. Range: [{self.data['ax'].min():.2f}, {self.data['ax'].max():.2f}] m/s²")
        report.append(f"Lateral Acc. Range: [{self.data['ay'].min():.2f}, {self.data['ay'].max():.2f}] m/s²")
        report.append(f"Mean Longitudinal Acc.: {self.data['ax'].mean():.2f} m/s²")
        report.append(f"Mean Lateral Acc.: {self.data['ay'].mean():.2f} m/s²")
        
        report.append("\n" + "-" * 40)
        report.append("CONTROL INPUT STATISTICS")
        report.append("-" * 40)
        report.append(f"Throttle Usage: {(self.data['throttle'] > 0).mean() * 100:.1f}%")
        report.append(f"Brake Usage: {(self.data['brake'] > 0).mean() * 100:.1f}%")
        report.append(f"Average Throttle: {self.data['throttle'].mean():.3f}")
        report.append(f"Average Brake: {self.data['brake'].mean():.3f}")
        report.append(f"Steering Range: [{self.data['steer'].min():.3f}, {self.data['steer'].max():.3f}]")
        
        # Driving style analysis
        report.append("\n" + "-" * 40)
        report.append("DRIVING STYLE ANALYSIS")
        report.append("-" * 40)
        
        # Aggressiveness score
        aggressive_score = (
            self.data['throttle'].mean() * 0.3 +
            (self.data['ax'].abs().mean() / 10) * 0.3 +
            (self.data['speed'].max() / 150) * 0.4
        )
        
        if aggressive_score < 0.3:
            style = "Conservative"
        elif aggressive_score < 0.6:
            style = "Normal"
        else:
            style = "Aggressive"
        
        report.append(f"Driving Style: {style} (Score: {aggressive_score:.2f})")
        
        # Smoothness
        smoothness = 1 - np.mean([
            np.std(np.diff(self.data['throttle'])),
            np.std(np.diff(self.data['brake'])),
            np.std(np.diff(self.data['steer']))
        ])
        report.append(f"Control Smoothness: {smoothness:.2f}")
        
        # Safety assessment
        report.append("\n" + "-" * 40)
        report.append("SAFETY ASSESSMENT")
        report.append("-" * 40)
        
        # Hard braking count
        hard_brake_count = (self.data['brake'] > 0.7).sum()
        report.append(f"Hard Braking Events: {hard_brake_count}")
        
        # Sharp turn count
        hard_turn_count = (self.data['steer'].abs() > 0.5).sum()
        report.append(f"Sharp Turn Events: {hard_turn_count}")
        
        # Overspeed (assuming 100km/h limit)
        overspeed_ratio = (self.data['speed'] > 100).mean()
        report.append(f"Overspeed Time Ratio: {overspeed_ratio * 100:.1f}%")
        
        return "\n".join(report)
    
    def show_3d_trajectory(self):
        """Show 3D trajectory"""
        if self.data is None:
            return
        
        self.clear_display()
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 3D trajectory
        scatter = ax.scatter(self.data['x'], self.data['y'], self.data['z'],
                           c=self.data['speed'], cmap='viridis', s=2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Vehicle Trajectory')
        
        plt.colorbar(scatter, ax=ax, label='Speed (km/h)')
        
        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_driving_behavior(self):
        """Driving behavior analysis"""
        if self.data is None:
            return
        
        self.clear_display()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # Speed-throttle relationship
        axes[0, 0].scatter(self.data['throttle'], self.data['speed'], 
                          alpha=0.5, s=1, c='green')
        axes[0, 0].set_xlabel('Throttle')
        axes[0, 0].set_ylabel('Speed (km/h)')
        axes[0, 0].set_title('Speed-Throttle Relationship')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Speed-brake relationship
        brake_data = self.data[self.data['brake'] > 0]
        if not brake_data.empty:
            axes[0, 1].scatter(brake_data['brake'], brake_data['speed'],
                             alpha=0.5, s=1, c='red')
        axes[0, 1].set_xlabel('Brake')
        axes[0, 1].set_ylabel('Speed (km/h)')
        axes[0, 1].set_title('Speed-Brake Relationship')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Steering-lateral acceleration relationship
        axes[1, 0].scatter(self.data['steer'], self.data['ay'],
                          alpha=0.5, s=1, c='blue')
        axes[1, 0].set_xlabel('Steering Angle')
        axes[1, 0].set_ylabel('Lateral Acc. (m/s²)')
        axes[1, 0].set_title('Steering-Lateral Acc. Relationship')
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Driving mode distribution
        driving_modes = []
        for _, row in self.data.iterrows():
            if row['throttle'] > 0.5:
                driving_modes.append('Accelerating')
            elif row['brake'] > 0.1:
                driving_modes.append('Braking')
            elif abs(row['steer']) > 0.1:
                driving_modes.append('Turning')
            else:
                driving_modes.append('Cruising')
        
        mode_counts = pd.Series(driving_modes).value_counts()
        axes[1, 1].pie(mode_counts.values, labels=mode_counts.index,
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Driving Mode Distribution')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_replay(self):
        """Real-time replay"""
        if self.data is None:
            return
        
        self.clear_display()
        
        # Create replay control panel
        control_panel = ttk.Frame(self.main_frame)
        control_panel.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Label(control_panel, text="Replay Control:").pack(side=tk.LEFT, padx=5)
        
        # Play button
        self.play_button = ttk.Button(control_panel, text="▶ Play",
                                      command=self.toggle_replay)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # Speed control
        ttk.Label(control_panel, text="Speed:").pack(side=tk.LEFT, padx=5)
        self.replay_speed = tk.Scale(control_panel, from_=0.1, to=5.0,
                                     resolution=0.1, orient=tk.HORIZONTAL,
                                     length=200)
        self.replay_speed.set(1.0)
        self.replay_speed.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        ttk.Label(control_panel, text="Progress:").pack(side=tk.LEFT, padx=5)
        self.progress = tk.Scale(control_panel, from_=0, to=len(self.data)-1,
                                resolution=1, orient=tk.HORIZONTAL,
                                length=400, command=self.on_progress_change)
        self.progress.pack(side=tk.LEFT, padx=5)
        
        # Create replay plot
        self.setup_replay_plot()
        
        self.is_playing = False
        self.current_frame = 0
    
    def setup_replay_plot(self):
        """Setup replay plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Trajectory plot
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Vehicle Position')
        ax1.set_aspect('equal')
        
        # Full trajectory (gray background)
        ax1.plot(self.data['x'], self.data['y'], 'gray', alpha=0.3, linewidth=1)
        
        # Current position marker
        self.position_point, = ax1.plot([], [], 'ro', markersize=8)
        
        # Trail line
        self.trail_line, = ax1.plot([], [], 'b-', linewidth=2)
        
        # Dashboard
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title('Real-time Data')
        
        # Speed display
        self.speed_text = ax2.text(5, 8, '', fontsize=20, ha='center')
        self.throttle_text = ax2.text(5, 6, '', fontsize=14, ha='center')
        self.brake_text = ax2.text(5, 4, '', fontsize=14, ha='center')
        self.steer_text = ax2.text(5, 2, '', fontsize=14, ha='center')
        
        self.replay_fig = fig
        self.replay_ax1 = ax1
        self.replay_ax2 = ax2
        
        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.replay_canvas = canvas
    
    def toggle_replay(self):
        """Toggle play state"""
        if self.is_playing:
            self.is_playing = False
            self.play_button.config(text="▶ Play")
        else:
            self.is_playing = True
            self.play_button.config(text="⏸ Pause")
            self.animate_replay()
    
    def on_progress_change(self, value):
        """Progress bar change"""
        self.current_frame = int(float(value))
        self.update_replay_frame()
    
    def update_replay_frame(self):
        """Update replay frame"""
        if self.current_frame >= len(self.data):
            self.current_frame = 0
            self.is_playing = False
            self.play_button.config(text="▶ Play")
            return
        
        # Get current frame data
        row = self.data.iloc[self.current_frame]
        
        # Update position marker
        self.position_point.set_data([row['x']], [row['y']])
        
        # Update trail
        trail_end = min(self.current_frame + 1, len(self.data))
        self.trail_line.set_data(self.data['x'][:trail_end], 
                                self.data['y'][:trail_end])
        
        # Update dashboard
        self.speed_text.set_text(f"Speed: {row['speed']:.1f} km/h")
        self.throttle_text.set_text(f"Throttle: {row['throttle']:.2%}")
        self.brake_text.set_text(f"Brake: {row['brake']:.2%}")
        self.steer_text.set_text(f"Steering: {row['steer']:.2f}")
        
        # Update progress bar
        self.progress.set(self.current_frame)
        
        # Refresh canvas
        self.replay_canvas.draw()
    
    def animate_replay(self):
        """Animate replay"""
        if self.is_playing:
            self.update_replay_frame()
            self.current_frame += 1
            
            # Calculate delay
            delay = int(50 / self.replay_speed.get())
            
            # Continue playing
            self.root.after(delay, self.animate_replay)
    
    def run(self):
        """Run visualization tool"""
        self.root.mainloop()


if __name__ == "__main__":
    visualizer = DataVisualizer()
    visualizer.run()