"""数据收集核心类 - 简化版（仅支持方向盘控制）"""
import carla
import time
import json
import numpy as np
from datetime import datetime
import config
import utils
from sensor_manager import SensorManager
from vehicle_control import WheelControl
from data_saver import DataSaver

class DataCollector:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.sensor_manager = None
        self.controller = None
        self.data_saver = None
        self.traffic_actors = []
        
    def setup(self):
        """初始化CARLA环境"""
        try:
            # 连接到CARLA
            print("连接到CARLA服务器...")
            self.client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
            self.client.set_timeout(config.TIMEOUT)
            
            # 获取世界
            self.world = self.client.get_world()
            
            # 如果需要加载特定地图
            if config.MAP_NAME:
                current_map = self.world.get_map().name.split('/')[-1]
                if not current_map.startswith(config.MAP_NAME):
                    print(f"加载地图 {config.MAP_NAME}...")
                    self.world = self.client.load_world(config.MAP_NAME)
                    time.sleep(5)
            
            # 设置天气
            self._set_weather()
            
            # 生成主车辆
            print("生成主控车辆...")
            self._spawn_ego_vehicle()
            
            # 设置传感器
            print("设置传感器...")
            self.sensor_manager = SensorManager(self.world, self.vehicle)
            self.sensor_manager.setup_sensors()
            
            # 设置方向盘控制器
            print("初始化方向盘控制器...")
            self.controller = WheelControl(self.vehicle, self.world)
            
            # 设置数据保存器
            print("初始化数据保存器...")
            self.data_saver = DataSaver()
            
            # 生成背景交通
            print(f"生成 {config.TRAFFIC_VEHICLES} 辆背景车辆...")
            self.traffic_actors = utils.setup_traffic(
                self.world, 
                config.TRAFFIC_VEHICLES, 
                config.TRAFFIC_PEDESTRIANS
            )
            
            print(f"\n环境设置完成！")
            print(f"- 主车辆已生成")
            print(f"- {len(self.traffic_actors)} 辆背景车辆")
            print(f"- 数据保存路径: {self.data_saver.save_path}")
            
        except Exception as e:
            print(f"设置失败: {e}")
            raise
    
    def _set_weather(self):
        """设置天气"""
        weather_presets = {
            'ClearNoon': carla.WeatherParameters.ClearNoon,
            'CloudyNoon': carla.WeatherParameters.CloudyNoon,
            'WetNoon': carla.WeatherParameters.WetNoon,
            'HardRainNoon': carla.WeatherParameters.HardRainNoon
        }
        
        weather = weather_presets.get(config.WEATHER_PRESET, carla.WeatherParameters.ClearNoon)
        self.world.set_weather(weather)
        print(f"天气设置: {config.WEATHER_PRESET}")
    
    def _spawn_ego_vehicle(self):
        """生成主控车辆"""
        bp_library = self.world.get_blueprint_library()
        vehicle_bp = bp_library.filter(config.VEHICLE_MODEL)[0]
        
        # 设置颜色
        if vehicle_bp.has_attribute('color'):
            color = vehicle_bp.get_attribute('color').recommended_values[0]
            vehicle_bp.set_attribute('color', color)
        
        # 获取生成点
        spawn_points = self.world.get_map().get_spawn_points()
        if config.SPAWN_POINT_INDEX is not None and config.SPAWN_POINT_INDEX < len(spawn_points):
            spawn_point = spawn_points[config.SPAWN_POINT_INDEX]
        else:
            spawn_point = np.random.choice(spawn_points)
        
        # 生成车辆
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        time.sleep(0.5)
        
        # 初始化车辆状态
        physics_control = self.vehicle.get_physics_control()
        physics_control.use_gear_autobox = True
        self.vehicle.apply_physics_control(physics_control)
        
        # 初始化控制状态
        initial_control = carla.VehicleControl()
        initial_control.hand_brake = False
        initial_control.manual_gear_shift = False
        initial_control.gear = 0
        self.vehicle.apply_control(initial_control)
        
        print(f"Vehicle spawned: {config.VEHICLE_MODEL}")
        time.sleep(0.5)
    
    def collect_frame_data(self):
        """收集一帧数据"""
        try:
            # 获取车辆状态
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            acceleration = self.vehicle.get_acceleration()
            angular_velocity = self.vehicle.get_angular_velocity()
            
            # 获取控制输入
            control_data = self.controller.get_control_data()
            
            # 获取周围车辆信息
            nearby_vehicles = utils.find_nearby_vehicles(self.world, self.vehicle)
            
            # 获取车道信息
            waypoint = self.world.get_map().get_waypoint(transform.location)
            
            # 计算速度
            speed = utils.get_speed(self.vehicle)
            
            # 组装轨迹数据
            trajectory_data = {
                'timestamp': time.time(),
                'frame': self.world.get_snapshot().frame,
                
                # 位置和姿态
                'x': transform.location.x,
                'y': transform.location.y,
                'z': transform.location.z,
                'roll': transform.rotation.roll,
                'pitch': transform.rotation.pitch,
                'yaw': transform.rotation.yaw,
                
                # 速度和加速度
                'vx': velocity.x,
                'vy': velocity.y,
                'vz': velocity.z,
                'ax': acceleration.x,
                'ay': acceleration.y,
                'az': acceleration.z,
                'angular_vx': angular_velocity.x,
                'angular_vy': angular_velocity.y,
                'angular_vz': angular_velocity.z,
                
                # 控制输入
                'throttle': control_data['throttle'],
                'brake': control_data['brake'],
                'steer': control_data['steer'],
                'hand_brake': control_data['hand_brake'],
                'reverse': control_data['reverse'],
                
                # 车道信息
                'lane_id': waypoint.lane_id,
                'road_id': waypoint.road_id,
                'lane_width': waypoint.lane_width,
                'is_junction': waypoint.is_junction,
                
                # 速度（标量）
                'speed': speed,
                
                # 周围车辆
                'nearby_vehicles_count': len(nearby_vehicles),
                'nearby_vehicles': json.dumps(nearby_vehicles)
            }
            
            # 获取传感器数据
            sensor_data = self.sensor_manager.get_data()
            
            # 获取相机数据（包含后视镜）
            camera_data = self.controller.get_camera_data()
            
            return {
                'trajectory': trajectory_data,
                'sensors': sensor_data,
                'cameras': camera_data
            }
            
        except Exception as e:
            print(f"收集数据错误: {e}")
            return None
    
    def run(self):
        """主循环"""
        try:
            episode_start = time.time()
            frame_count = 0
            last_frame_time = time.time()
            last_status_time = time.time()
            
            print("\n" + "="*50)
            print("数据收集已开始！")
            print("使用方向盘控制车辆")
            print("="*50 + "\n")
            
            while True:
                # 更新控制器（处理方向盘输入）
                should_quit = self.controller.update()
                if should_quit:
                    break
                
                # 按指定频率收集数据
                current_time = time.time()
                if current_time - last_frame_time >= 1.0 / config.COLLECT_FREQUENCY:
                    frame_data = self.collect_frame_data()
                    if frame_data:
                        self.data_saver.add_frame(frame_data)
                        frame_count += 1
                    
                    last_frame_time = current_time
                
                # 定期显示状态（每2秒）
                if current_time - last_status_time >= 2.0:
                    elapsed = current_time - episode_start
                    speed = utils.get_speed(self.vehicle)
                    print(f"[状态] 时间: {elapsed:.1f}s | 帧数: {frame_count} | 速度: {speed:.1f} km/h")
                    last_status_time = current_time
                
                # 检查是否完成一个episode
                if current_time - episode_start > config.EPISODE_LENGTH:
                    print(f"\n[保存] Episode {self.data_saver.episode_count + 1} 完成，正在保存...")
                    self.data_saver.save_episode()
                    episode_start = current_time
                    frame_count = 0
                    print("[继续] 开始新的Episode\n")
        
        except KeyboardInterrupt:
            print("\n\n[中断] 用户中断")
        
        finally:
            # 保存最后的数据
            if self.data_saver and len(self.data_saver.trajectory_buffer) > 0:
                print("[保存] 正在保存最后的数据...")
                self.data_saver.save_episode()
            
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n[清理] 正在清理资源...")
        
        # 销毁控制器
        if self.controller:
            self.controller.destroy()
        
        # 销毁传感器
        if self.sensor_manager:
            self.sensor_manager.destroy()
        
        # 销毁车辆
        if self.vehicle:
            self.vehicle.destroy()
        
        # 销毁交通车辆
        for actor in self.traffic_actors:
            if actor:
                try:
                    actor.destroy()
                except:
                    pass
        
        print("[完成] 清理完成")