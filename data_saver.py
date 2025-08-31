"""数据保存器 - 支持多相机数据，可选择保存H5文件"""
import os
import json
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import config

class DataSaver:
    def __init__(self, save_path=None):
        self.save_path = save_path or config.SAVE_PATH
        os.makedirs(self.save_path, exist_ok=True)
        
        # 创建数据缓冲区
        self.trajectory_buffer = []
        self.sensor_data_buffer = []
        self.camera_data_buffer = []  # 相机数据缓冲区
        self.episode_count = 0
        
        # H5文件保存选项
        self.save_sensor_h5 = config.SAVE_SENSOR_H5
        self.save_camera_h5 = config.SAVE_CAMERA_H5
        
        # 创建元数据文件（删除rear_mirror_camera）
        self.metadata = {
            'session_id': config.SESSION_ID,
            'map': config.MAP_NAME,
            'vehicle': config.VEHICLE_MODEL,
            'weather': config.WEATHER_PRESET,
            'frequency': config.COLLECT_FREQUENCY,
            'start_time': datetime.now().isoformat(),
            'save_options': {
                'save_sensor_h5': self.save_sensor_h5,
                'save_camera_h5': self.save_camera_h5
            },
            'camera_config': {
                'main_camera': config.SENSORS['main_camera'],
                'left_mirror_camera': config.SENSORS['left_mirror_camera'],
                'right_mirror_camera': config.SENSORS['right_mirror_camera']
                # 删除了 rear_mirror_camera
            }
        }
        
        with open(os.path.join(self.save_path, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # 打印保存选项信息
        print("\n数据保存配置：")
        print(f"  - 保存 sensors.h5: {'是' if self.save_sensor_h5 else '否'}")
        print(f"  - 保存 cameras.h5: {'是' if self.save_camera_h5 else '否'}")
        print(f"  - 保存 trajectory.csv: 是（始终保存）")
        if not self.save_sensor_h5 and not self.save_camera_h5:
            print("  注意：H5文件都不会保存，只保存轨迹CSV文件")
        print("")
    
    def add_frame(self, frame_data):
        """添加一帧数据"""
        self.trajectory_buffer.append(frame_data['trajectory'])
        
        # 只在需要保存时才缓存传感器数据
        if self.save_sensor_h5:
            self.sensor_data_buffer.append(frame_data['sensors'])
        
        # 只在需要保存时才缓存相机数据
        if self.save_camera_h5 and 'cameras' in frame_data:
            self.camera_data_buffer.append(frame_data['cameras'])
    
    def save_episode(self):
        """保存一个episode的数据"""
        self.episode_count += 1
        episode_dir = os.path.join(self.save_path, f'episode_{self.episode_count:04d}')
        os.makedirs(episode_dir, exist_ok=True)
        
        saved_files = []
        
        # 始终保存轨迹数据为CSV
        if self.trajectory_buffer:
            df = pd.DataFrame(self.trajectory_buffer)
            csv_path = os.path.join(episode_dir, 'trajectory.csv')
            df.to_csv(csv_path, index=False)
            saved_files.append('trajectory.csv')
        
        # 根据配置选择性保存传感器数据
        if self.save_sensor_h5 and self.sensor_data_buffer:
            self._save_sensor_data(episode_dir)
            saved_files.append('sensors.h5')
        
        # 根据配置选择性保存相机数据
        if self.save_camera_h5 and self.camera_data_buffer:
            self._save_camera_data(episode_dir)
            saved_files.append('cameras.h5')
        
        # 清空缓冲区
        self.trajectory_buffer = []
        self.sensor_data_buffer = []
        self.camera_data_buffer = []
        
        saved_files_str = ', '.join(saved_files) if saved_files else '仅轨迹数据'
        print(f"Episode {self.episode_count} 已保存 - 文件: {saved_files_str}")
    
    def _save_sensor_data(self, episode_dir):
        """保存传感器数据"""
        h5_path = os.path.join(episode_dir, 'sensors.h5')
        
        with h5py.File(h5_path, 'w') as f:
            # 处理每种传感器数据
            for i, frame in enumerate(self.sensor_data_buffer):
                frame_group = f.create_group(f'frame_{i:06d}')
                
                # RGB相机 (如果传感器管理器有的话)
                if 'rgb_camera' in frame and frame['rgb_camera'] is not None:
                    frame_group.create_dataset('rgb_camera', 
                                              data=frame['rgb_camera'],
                                              compression='gzip')
                
                # LIDAR
                if 'lidar' in frame and frame['lidar'] is not None:
                    frame_group.create_dataset('lidar', 
                                              data=frame['lidar'],
                                              compression='gzip')
                
                # GNSS
                if 'gnss' in frame and frame['gnss'] is not None:
                    frame_group.create_dataset('gnss', data=frame['gnss'])
                
                # IMU
                if 'imu' in frame and frame['imu'] is not None:
                    imu_group = frame_group.create_group('imu')
                    imu_group.create_dataset('accelerometer', 
                                            data=frame['imu']['accelerometer'])
                    imu_group.create_dataset('gyroscope', 
                                            data=frame['imu']['gyroscope'])
                    imu_group.attrs['compass'] = frame['imu']['compass']
    
    def _save_camera_data(self, episode_dir):
        """保存多相机数据（删除后视镜相机）"""
        h5_path = os.path.join(episode_dir, 'cameras.h5')
        
        with h5py.File(h5_path, 'w') as f:
            # 为每个相机创建数据集（删除rear_mirror_camera）
            camera_names = ['main_camera', 'left_mirror_camera', 'right_mirror_camera']
            
            for camera_name in camera_names:
                # 收集该相机的所有帧数据
                camera_frames = []
                for frame in self.camera_data_buffer:
                    if camera_name in frame and frame[camera_name] is not None:
                        camera_frames.append(frame[camera_name])
                    else:
                        # 如果某帧没有数据，填充零数组
                        if camera_frames:  # 如果之前有数据，使用相同尺寸
                            camera_frames.append(np.zeros_like(camera_frames[0]))
                
                # 保存相机数据
                if camera_frames:
                    camera_data = np.array(camera_frames)
                    f.create_dataset(
                        camera_name,
                        data=camera_data,
                        compression='gzip',
                        compression_opts=9
                    )
                    print(f"  保存 {camera_name}: {camera_data.shape}")
            
            # 保存时间戳和元数据
            timestamps = [frame.get('timestamp', i) for i, frame in enumerate(self.camera_data_buffer)]
            f.create_dataset('timestamps', data=timestamps)
            
            # 保存相机配置信息（删除rear_mirror_camera）
            config_group = f.create_group('config')
            for camera_name in camera_names:
                if camera_name in config.SENSORS:
                    cam_config = config.SENSORS[camera_name]
                    cam_group = config_group.create_group(camera_name)
                    cam_group.attrs['width'] = cam_config['width']
                    cam_group.attrs['height'] = cam_config['height']
                    cam_group.attrs['fov'] = cam_config['fov']
                    cam_group.create_dataset('position', data=cam_config['position'])
                    cam_group.create_dataset('rotation', data=cam_config['rotation'])
    
    def get_save_info(self):
        """获取保存信息统计"""
        info = {
            'save_path': self.save_path,
            'episodes_saved': self.episode_count,
            'current_buffer_size': len(self.trajectory_buffer),
            'save_options': {
                'trajectory_csv': True,
                'sensors_h5': self.save_sensor_h5,
                'cameras_h5': self.save_camera_h5
            }
        }
        return info