"""数据收集核心类 - 优化的重置逻辑"""
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
import math
import random

class StraightRoadDataCollector:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.sensor_manager = None
        self.controller = None
        self.data_saver = None
        
        # 环境特定属性
        self.ai_vehicles = []
        self.start_transform = None
        self.end_location = None
        self.episode_completed = False
        
    def setup(self):
        """初始化CARLA环境"""
        try:
            # 连接到CARLA
            print("连接到CARLA服务器...")
            self.client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
            self.client.set_timeout(config.TIMEOUT)
            
            # 获取世界并加载指定地图
            self.world = self.client.get_world()
            current_map = self.world.get_map().name.split('/')[-1]
            if not current_map.startswith(config.MAP_NAME):
                print(f"加载地图 {config.MAP_NAME}...")
                self.world = self.client.load_world(config.MAP_NAME)
                time.sleep(5)
            
            # 设置天气
            self._set_weather()
            
            # 设置起始和终点位置
            self._setup_route()
            
            # 生成主车辆
            print("生成主控车辆...")
            self._spawn_ego_vehicle()
            
            # 根据config生成AI车辆
            print("生成AI车辆...")
            self._spawn_ai_vehicles()
            
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
            
            print(f"\n环境设置完成！")
            print(f"- 起点: ({self.start_transform.location.x:.1f}, {self.start_transform.location.y:.1f}, {self.start_transform.location.z:.1f})")
            print(f"- 终点: ({self.end_location.x:.1f}, {self.end_location.y:.1f}, {self.end_location.z:.1f})")
            print(f"- AI车辆: {len(self.ai_vehicles)} 辆")
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
    
    def _setup_route(self):
        """设置起始和终点"""
        straight_config = config.STRAIGHT_ROAD_CONFIG
        
        # 获取地图所有生成点
        spawn_points = self.world.get_map().get_spawn_points()
        print(f"地图总共有 {len(spawn_points)} 个生成点")
        
        # 选择起始生成点
        start_index = straight_config.get('start_spawn_index', 0)
        if start_index >= len(spawn_points):
            start_index = 0
            print(f"警告: 起始生成点索引超出范围，使用索引 0")
        
        self.start_transform = spawn_points[start_index]
        
        # 计算终点位置
        end_coords = straight_config.get('end_location_coords')
        
        if end_coords and all(k in end_coords for k in ['x', 'y', 'z']):
            # 如果配置了坐标，直接创建 carla.Location 对象
            self.end_location = carla.Location(x=end_coords['x'], y=end_coords['y'], z=end_coords['z'])
            print("路线设置: 使用配置文件中的指定终点。")
            route_distance = self.start_transform.location.distance(self.end_location)
        
        print(f"路线设置:")
        print(f"  起点: ({self.start_transform.location.x:.1f}, {self.start_transform.location.y:.1f}, {self.start_transform.location.z:.1f})")
        print(f"  终点: ({self.end_location.x:.1f}, {self.end_location.y:.1f}, {self.end_location.z:.1f})")
        print(f"  距离: {route_distance}m")
    
    # def _calculate_end_point(self, start_transform, distance):
    #     """计算终点位置"""
    #     start_waypoint = self.world.get_map().get_waypoint(start_transform.location)
    #     current_waypoint = start_waypoint
    #     accumulated_distance = 0.0
    #     step_size = 5.0
        
    #     while accumulated_distance < distance:
    #         next_waypoints = current_waypoint.next(step_size)
    #         if not next_waypoints:
    #             break
    #         current_waypoint = next_waypoints[0]
    #         accumulated_distance += step_size
        
    #     return current_waypoint.transform.location
    
    def _spawn_ego_vehicle(self):
        """生成主控车辆"""
        bp_library = self.world.get_blueprint_library()
        vehicle_bp = bp_library.filter(config.VEHICLE_MODEL)[0]
        
        if vehicle_bp.has_attribute('color'):
            color = vehicle_bp.get_attribute('color').recommended_values[0]
            vehicle_bp.set_attribute('color', color)
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.start_transform)
        time.sleep(0.5)
        
        physics_control = self.vehicle.get_physics_control()
        physics_control.use_gear_autobox = True
        self.vehicle.apply_physics_control(physics_control)
        
        initial_control = carla.VehicleControl()
        initial_control.hand_brake = False
        initial_control.manual_gear_shift = False
        initial_control.gear = 0
        self.vehicle.apply_control(initial_control)
        
        print(f"主车生成完成: {config.VEHICLE_MODEL}")
        time.sleep(0.5)
    
    def _spawn_ai_vehicles(self):
        """根据config生成AI车辆"""
        bp_library = self.world.get_blueprint_library()
        world_map = self.world.get_map()
        
        # 获取Traffic Manager
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        
        # 获取主车起始路点
        ego_waypoint = world_map.get_waypoint(self.start_transform.location)
        
        # # 1. 首先生成前方静止障碍车辆（固定）
        # try:
        #     # 在前方10-20米随机位置生成静止车辆
        #     distance_ahead = random.uniform(10, 20)
            
        #     # 计算位置
        #     current_waypoint = ego_waypoint
        #     accumulated_distance = 0.0
            
        #     while accumulated_distance < distance_ahead:
        #         next_waypoints = current_waypoint.next(2.0)
        #         if not next_waypoints:
        #             break
        #         current_waypoint = next_waypoints[0]
        #         accumulated_distance += 2.0
            
        #     # 生成静止车辆（使用大众T2面包车作为障碍）
        #     static_vehicle_bp = bp_library.filter('vehicle.volkswagen.t2')[0]
        #     if static_vehicle_bp.has_attribute('color'):
        #         static_vehicle_bp.set_attribute('color', '255,0,0')  # 红色
            
        #     spawn_transform = current_waypoint.transform
        #     spawn_transform.location.z += 0.5
            
        #     static_vehicle = self.world.spawn_actor(static_vehicle_bp, spawn_transform)
            
        #     # 设置为静止（手刹拉起）
        #     static_control = carla.VehicleControl()
        #     static_control.hand_brake = True
        #     static_control.throttle = 0.0
        #     static_control.brake = 1.0
        #     static_vehicle.apply_control(static_control)
            
        #     self.ai_vehicles.append({
        #         'actor': static_vehicle,
        #         'config': {'model': 'static_obstacle', 'distance_ahead': distance_ahead},
        #         'spawn_transform': spawn_transform
        #     })
            
        #     print(f"✓ 前方静止障碍车辆生成完成，距离: {distance_ahead:.1f}米")
            
        # except Exception as e:
        #     print(f"生成前方静止车辆失败: {e}")
        
        # 2. 根据config生成随机交通车辆
        random_traffic_config = config.STRAIGHT_ROAD_CONFIG.get('random_traffic', {})
        num_random_vehicles = random_traffic_config.get('num_vehicles', config.TRAFFIC_VEHICLES)
        spawn_radius = random_traffic_config.get('spawn_radius', 200)
        
        # 获取所有生成点
        spawn_points = world_map.get_spawn_points()
        
        # 筛选在指定半径内的生成点
        suitable_spawn_points = []
        for sp in spawn_points:
            distance = sp.location.distance(self.start_transform.location)
            # 排除太近的位置（避免与主车碰撞）
            if 20 < distance < spawn_radius:
                suitable_spawn_points.append(sp)
        
        # 随机打乱
        random.shuffle(suitable_spawn_points)
        
        # 车辆模型列表
        vehicle_models = [
            'vehicle.audi.a2',
            'vehicle.audi.etron', 
            'vehicle.bmw.grandtourer',
            'vehicle.chevrolet.impala',
            'vehicle.dodge.charger_2020',
            'vehicle.ford.mustang',
            'vehicle.tesla.model3',
            'vehicle.mercedes.coupe_2020',
            'vehicle.mini.cooper_s',
            'vehicle.nissan.patrol_2021',
            'vehicle.seat.leon'
        ]
        
        # 生成随机交通车辆（减1因为已有一辆静止车）
        vehicles_to_spawn = min(num_random_vehicles - 1, len(suitable_spawn_points))
        vehicles_spawned = 0
        
        for i in range(vehicles_to_spawn):
            if i >= len(suitable_spawn_points):
                break
                
            try:
                spawn_point = suitable_spawn_points[i]
                
                # 随机选择车辆模型
                vehicle_model = random.choice(vehicle_models)
                vehicle_bp = bp_library.filter(vehicle_model)[0]
                
                # 随机颜色
                if vehicle_bp.has_attribute('color'):
                    colors = vehicle_bp.get_attribute('color').recommended_values
                    vehicle_bp.set_attribute('color', random.choice(colors))
                
                # 生成车辆
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                
                # 设置自动驾驶
                vehicle.set_autopilot(True, traffic_manager.get_port())
                
                # 随机设置驾驶行为
                speed_diff = random.uniform(-30, 20)  # 速度差异
                traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_diff)
                
                # 设置车距
                distance_to_leading = random.uniform(1.5, 3.0)
                traffic_manager.distance_to_leading_vehicle(vehicle, distance_to_leading)
                
                self.ai_vehicles.append({
                    'actor': vehicle,
                    'config': {
                        'model': vehicle_model,
                        'type': 'random_traffic',
                        'speed_diff': speed_diff
                    },
                    'spawn_transform': spawn_point
                })
                
                vehicles_spawned += 1
                
            except Exception as e:
                continue
        
        print(f"✓ 成功生成 {len(self.ai_vehicles)} 辆AI车辆")
        print(f"  - 前方静止障碍车: 1辆")
        print(f"  - 随机交通车辆: {vehicles_spawned}辆")
    
    def _check_episode_completion(self):
        """检查是否完成episode"""
        if self.vehicle is None or self.end_location is None:
            return False
        
        current_location = self.vehicle.get_location()
        distance_to_end = current_location.distance(self.end_location)
        completion_distance = config.STRAIGHT_ROAD_CONFIG['completion_distance']
        
        return distance_to_end <= completion_distance
    
    def _reset_episode(self):
        """重置episode - 只重置主车和静止障碍车"""
        print("\n[重置] 正在重置Episode...")
        
        # 1. 重置主车位置
        self.vehicle.set_transform(self.start_transform)
        
        # 停止车辆运动
        physics_control = self.vehicle.get_physics_control()
        self.vehicle.apply_physics_control(physics_control)
        
        reset_control = carla.VehicleControl()
        reset_control.hand_brake = True
        reset_control.throttle = 0.0
        reset_control.brake = 1.0
        reset_control.steer = 0.0
        self.vehicle.apply_control(reset_control)
        
        # 2. 只重置静止障碍车，保留其他自动驾驶车辆
        static_vehicle = None
        other_vehicles = []
        
        # 分离静止车和自动驾驶车辆
        for ai_vehicle_info in self.ai_vehicles:
            if ai_vehicle_info['config'].get('model') == 'static_obstacle':
                # 销毁旧的静止障碍车
                try:
                    ai_vehicle_info['actor'].destroy()
                    print("[重置] 移除旧的静止障碍车")
                except:
                    pass
            else:
                # 保留自动驾驶车辆
                other_vehicles.append(ai_vehicle_info)
        
        # 更新AI车辆列表（保留自动驾驶车辆）
        self.ai_vehicles = other_vehicles
        
        # # 3. 重新生成前方静止障碍车
        # try:
        #     bp_library = self.world.get_blueprint_library()
        #     world_map = self.world.get_map()
        #     ego_waypoint = world_map.get_waypoint(self.start_transform.location)
            
        #     # 在前方30-60米随机位置生成静止车辆
        #     distance_ahead = random.uniform(30, 60)
            
        #     # 计算位置
        #     current_waypoint = ego_waypoint
        #     accumulated_distance = 0.0
            
        #     while accumulated_distance < distance_ahead:
        #         next_waypoints = current_waypoint.next(2.0)
        #         if not next_waypoints:
        #             break
        #         current_waypoint = next_waypoints[0]
        #         accumulated_distance += 2.0
            
        #     # 生成静止车辆（使用大众T2面包车作为障碍）
        #     static_vehicle_bp = bp_library.filter('vehicle.volkswagen.t2')[0]
        #     if static_vehicle_bp.has_attribute('color'):
        #         static_vehicle_bp.set_attribute('color', '255,0,0')  # 红色
            
        #     spawn_transform = current_waypoint.transform
        #     spawn_transform.location.z += 0.5
            
        #     static_vehicle = self.world.spawn_actor(static_vehicle_bp, spawn_transform)
            
        #     # 设置为静止（手刹拉起）
        #     static_control = carla.VehicleControl()
        #     static_control.hand_brake = True
        #     static_control.throttle = 0.0
        #     static_control.brake = 1.0
        #     static_vehicle.apply_control(static_control)
            
        #     self.ai_vehicles.append({
        #         'actor': static_vehicle,
        #         'config': {'model': 'static_obstacle', 'distance_ahead': distance_ahead},
        #         'spawn_transform': spawn_transform
        #     })
            
        #     print(f"[重置] 新静止障碍车生成完成，距离: {distance_ahead:.1f}米")
        #     print(f"[重置] 保留 {len(other_vehicles)} 辆自动驾驶车辆继续运行")
            
        # except Exception as e:
        #     print(f"[重置] 生成静止障碍车失败: {e}")
        
        # 短暂等待
        time.sleep(1.0)
        
        # 4. 释放主车手刹
        final_control = carla.VehicleControl()
        final_control.hand_brake = False
        final_control.throttle = 0.0
        final_control.brake = 0.0
        final_control.steer = 0.0
        self.vehicle.apply_control(final_control)
        
        print("[重置] Episode重置完成\n")
    
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
            
            # 获取AI车辆信息
            ai_vehicles_info = self._get_ai_vehicles_info()
            
            # 获取车道信息
            waypoint = self.world.get_map().get_waypoint(transform.location)
            
            # 计算速度和距离信息
            speed = utils.get_speed(self.vehicle)
            distance_to_end = transform.location.distance(self.end_location)
            progress = 1.0 - (distance_to_end / self.start_transform.location.distance(self.end_location))
            
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
                
                # 速度和进度信息
                'speed': speed,
                'distance_to_end': distance_to_end,
                'progress': progress,
                
                # AI车辆信息
                'ai_vehicles_count': len(ai_vehicles_info),
                'ai_vehicles': json.dumps(ai_vehicles_info)
            }
            
            # 获取传感器数据
            sensor_data = self.sensor_manager.get_data()
            
            # 获取相机数据
            camera_data = self.controller.get_camera_data()
            
            return {
                'trajectory': trajectory_data,
                'sensors': sensor_data,
                'cameras': camera_data
            }
            
        except Exception as e:
            print(f"收集数据错误: {e}")
            return None
    
    def _get_ai_vehicles_info(self):
        """获取AI车辆信息"""
        ego_location = self.vehicle.get_location()
        ai_info = []
        
        for i, ai_vehicle_info in enumerate(self.ai_vehicles):
            try:
                ai_vehicle = ai_vehicle_info['actor']
                ai_location = ai_vehicle.get_location()
                ai_velocity = ai_vehicle.get_velocity()
                
                relative_pos = np.array([
                    ai_location.x - ego_location.x,
                    ai_location.y - ego_location.y,
                    ai_location.z - ego_location.z
                ])
                
                ai_info.append({
                    'id': f'ai_{i}',
                    'type': ai_vehicle_info['config'].get('type', 'unknown'),
                    'model': ai_vehicle_info['config'].get('model', 'unknown'),
                    'distance': ego_location.distance(ai_location),
                    'relative_position': relative_pos.tolist(),
                    'velocity': [ai_velocity.x, ai_velocity.y, ai_velocity.z],
                    'speed': utils.get_speed(ai_vehicle),
                    'is_static': ai_vehicle_info['config'].get('model') == 'static_obstacle'
                })
            except:
                continue
        
        return sorted(ai_info, key=lambda x: x['distance'])
    
    def run(self):
        """主循环"""
        try:
            episode_start = time.time()
            frame_count = 0
            last_frame_time = time.time()
            last_status_time = time.time()
            
            print("\n" + "="*50)
            print("数据收集已开始！")
            print("使用方向盘控制车辆，到达终点自动重置")
            print(f"完成判定距离: {config.STRAIGHT_ROAD_CONFIG['completion_distance']:.0f}m")
            print("="*50 + "\n")
            
            self.episode_completed = False
            
            while True:
                # 更新控制器
                should_quit = self.controller.update()
                if should_quit:
                    break
                
                # 检查是否完成episode
                if self._check_episode_completion():
                    print(f"\n[完成] Episode完成! 用时: {time.time() - episode_start:.1f}秒")
                    
                    # 保存当前episode数据
                    if len(self.data_saver.trajectory_buffer) > 0:
                        self.data_saver.save_episode()
                    
                    # 重置episode
                    self._reset_episode()
                    
                    # 重新开始计时
                    episode_start = time.time()
                    frame_count = 0
                    
                    time.sleep(1.0)
                    print("[新Episode] 开始新的Episode\n")
                    continue
                
                # 按指定频率收集数据
                current_time = time.time()
                if current_time - last_frame_time >= 1.0 / config.COLLECT_FREQUENCY:
                    frame_data = self.collect_frame_data()
                    if frame_data:
                        self.data_saver.add_frame(frame_data)
                        frame_count += 1
                    
                    last_frame_time = current_time
                
                # 定期显示状态
                if current_time - last_status_time >= 2.0:
                    elapsed = current_time - episode_start
                    speed = utils.get_speed(self.vehicle)
                    distance_to_end = self.vehicle.get_location().distance(self.end_location)
                    
                    # 统计AI车辆状态
                    static_count = 0
                    moving_count = 0
                    for ai_info in self.ai_vehicles:
                        try:
                            if ai_info['config'].get('model') == 'static_obstacle':
                                static_count += 1
                            elif utils.get_speed(ai_info['actor']) > 1.0:
                                moving_count += 1
                        except:
                            pass
                    
                    print(f"[状态] 时间: {elapsed:.1f}s | 帧数: {frame_count} | "
                          f"速度: {speed:.1f} km/h | 距终点: {distance_to_end:.1f}m | "
                          f"静止: {static_count} | 移动: {moving_count}")
                    last_status_time = current_time
                
                # 检查最大episode时间
                if current_time - episode_start > config.EPISODE_LENGTH:
                    print(f"\n[超时] Episode超时，强制重置...")
                    if len(self.data_saver.trajectory_buffer) > 0:
                        self.data_saver.save_episode()
                    self._reset_episode()
                    episode_start = current_time
                    frame_count = 0
        
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
        
        # 销毁AI车辆
        for ai_vehicle_info in self.ai_vehicles:
            try:
                ai_vehicle_info['actor'].destroy()
            except:
                pass
        
        # 销毁主车辆
        if self.vehicle:
            self.vehicle.destroy()
        
        print("[完成] 清理完成")

# 为了兼容原有代码，保留DataCollector类名
DataCollector = StraightRoadDataCollector