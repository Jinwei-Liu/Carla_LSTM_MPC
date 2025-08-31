"""左右后视镜数据查看器 - 验证保存的多相机数据（删除后视镜）"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MirrorDataViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CARLA Mirror Data Viewer - Left/Right Mirrors")
        self.root.geometry("1000x600")  # 调整窗口大小适应3个相机
        
        self.data = None
        self.current_frame = 0
        self.total_frames = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 控制面板
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 加载数据按钮
        ttk.Button(control_frame, text="Load Camera Data", 
                  command=self.load_data).pack(side=tk.LEFT, padx=5)
        
        # 帧控制
        ttk.Label(control_frame, text="Frame:").pack(side=tk.LEFT, padx=5)
        
        self.frame_var = tk.IntVar()
        self.frame_scale = tk.Scale(control_frame, from_=0, to=100,
                                   variable=self.frame_var, orient=tk.HORIZONTAL,
                                   length=300, command=self.on_frame_change)
        self.frame_scale.pack(side=tk.LEFT, padx=5)
        
        self.frame_label = ttk.Label(control_frame, text="0 / 0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        
        # 播放控制
        self.play_button = ttk.Button(control_frame, text="Play", 
                                     command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # 显示区域
        self.setup_display()
        
        self.is_playing = False
        
    def setup_display(self):
        """设置显示区域 - 1x3布局（删除后视镜）"""
        # 创建1x3的子图布局，适应主相机和左右后视镜
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle('CARLA Multi-Camera View (No Rear Mirror)')
        
        # 设置子图标题
        self.axes[0].set_title('Main Camera (Driver View)')
        self.axes[1].set_title('Left Mirror')
        self.axes[2].set_title('Right Mirror')
        # 删除了后视镜子图
        
        # 关闭坐标轴
        for ax in self.axes:
            ax.axis('off')
        
        # 嵌入matplotlib到tkinter
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.canvas = canvas
        
    def load_data(self):
        """加载相机数据"""
        # 选择cameras.h5文件
        file_path = filedialog.askopenfilename(
            title="Select cameras.h5 file",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            with h5py.File(file_path, 'r') as f:
                # 加载各相机数据（删除rear_mirror_camera）
                self.data = {}
                camera_names = ['main_camera', 'left_mirror_camera', 'right_mirror_camera']
                
                for camera_name in camera_names:
                    if camera_name in f:
                        self.data[camera_name] = f[camera_name][:]
                        print(f"Loaded {camera_name}: {self.data[camera_name].shape}")
                    else:
                        self.data[camera_name] = None
                        print(f"Warning: {camera_name} not found in file")
                
                # 获取帧数
                if self.data['main_camera'] is not None:
                    self.total_frames = self.data['main_camera'].shape[0]
                else:
                    # 找第一个可用的相机数据
                    for camera_data in self.data.values():
                        if camera_data is not None:
                            self.total_frames = camera_data.shape[0]
                            break
                
                # 更新界面
                if self.total_frames > 0:
                    self.frame_scale.config(to=self.total_frames-1)
                    self.frame_label.config(text=f"0 / {self.total_frames-1}")
                    self.current_frame = 0
                    self.display_frame()
                    print(f"Successfully loaded data with {self.total_frames} frames")
                    print("Note: Rear mirror camera has been removed from display")
                else:
                    print("Error: No camera data found")
                    
        except Exception as e:
            print(f"Error loading data: {e}")
            
    def on_frame_change(self, value):
        """帧数变化回调"""
        self.current_frame = int(value)
        self.frame_label.config(text=f"{self.current_frame} / {self.total_frames-1}")
        if self.data is not None:
            self.display_frame()
            
    def display_frame(self):
        """显示当前帧"""
        if self.data is None or self.current_frame >= self.total_frames:
            return
            
        # 相机数据和对应的子图（删除后视镜）
        camera_axes_map = {
            'main_camera': self.axes[0],
            'left_mirror_camera': self.axes[1],
            'right_mirror_camera': self.axes[2]
        }
        
        # 显示每个相机的当前帧
        for camera_name, ax in camera_axes_map.items():
            ax.clear()
            ax.axis('off')
            
            if self.data[camera_name] is not None:
                frame_data = self.data[camera_name][self.current_frame]
                
                # 确保数据格式正确 (H, W, 3)
                if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
                    # 数据值范围应该在0-255，如果在0-1则需要转换
                    if frame_data.max() <= 1.0:
                        frame_data = (frame_data * 255).astype(np.uint8)
                    else:
                        frame_data = frame_data.astype(np.uint8)
                        
                    ax.imshow(frame_data)
                else:
                    # 数据格式不正确，显示错误信息
                    ax.text(0.5, 0.5, f'Invalid data\nShape: {frame_data.shape}',
                           ha='center', va='center', transform=ax.transAxes)
                    
                # 设置标题（包含数据信息）
                title = camera_name.replace('_', ' ').title()
                if self.data[camera_name] is not None:
                    title += f'\n{self.data[camera_name].shape[1]}x{self.data[camera_name].shape[2]}'
                ax.set_title(title, fontsize=10)
            else:
                # 没有数据，显示占位符
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=16, color='red')
                ax.set_title(camera_name.replace('_', ' ').title(), fontsize=10)
        
        # 刷新显示
        self.canvas.draw()
        
    def toggle_play(self):
        """切换播放状态"""
        if not self.data:
            return
            
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.config(text="Pause")
            self.play_animation()
        else:
            self.play_button.config(text="Play")
            
    def play_animation(self):
        """播放动画"""
        if not self.is_playing:
            return
            
        # 下一帧
        self.current_frame = (self.current_frame + 1) % self.total_frames
        self.frame_var.set(self.current_frame)
        self.frame_label.config(text=f"{self.current_frame} / {self.total_frames-1}")
        self.display_frame()
        
        # 继续播放
        if self.is_playing:
            self.root.after(100, self.play_animation)  # 10 FPS
            
    def run(self):
        """运行查看器"""
        print("Mirror Data Viewer - Left/Right mirrors only")
        print("Rear mirror has been removed from the system")
        self.root.mainloop()

if __name__ == "__main__":
    viewer = MirrorDataViewer()
    viewer.run()