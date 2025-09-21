"""工具函数 - 直道变道环境版本"""
import carla
import numpy as np
import os
import json

def get_actor_blueprints(world, filter, generation='All'):
    """获取特定类型的蓝图"""
    bps = world.get_blueprint_library().filter(filter)
    if generation.lower() == "all":
        return bps
    
    # 处理没有generation属性的情况
    bps_with_gen = []
    for bp in bps:
        if bp.has_attribute('generation'):
            if int(bp.get_attribute('generation')) == generation:
                bps_with_gen.append(bp)
        else:
            bps_with_gen.append(bp)  # 没有generation属性的也包含
    
    return bps_with_gen if bps_with_gen else bps

def create_transform(location, rotation=(0,0,0)):
    """创建变换对象"""
    return carla.Transform(
        carla.Location(x=location[0], y=location[1], z=location[2]),
        carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2])
    )

def get_speed(vehicle):
    """获取车辆速度 km/h"""
    vel = vehicle.get_velocity()
    return 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def get_acceleration(vehicle):
    """获取车辆加速度"""
    acc = vehicle.get_acceleration()
    return np.array([acc.x, acc.y, acc.z])

def setup_traffic(world, num_vehicles=2, num_walkers=0):
    """
    为直道环境生成背景交通流 - 已弃用
    
    注意：这个函数在直道环境中已被data_collector中的
    _spawn_ai_vehicles()方法替代，用于更精确的AI车辆控制
    """
    print("Warning: setup_traffic() 已被直道环境专用的AI车辆生成逻辑替代")
    return []

def setup_straight_road_traffic(world, ego_vehicle, ai_configs):
    """
    为直道环境设置特定的AI车辆（备用函数）
    
    Args:
        world: CARLA world对象
        ego_vehicle: 主控车辆
        ai_configs: AI车辆配置列表
    
    Returns:
        list: 生成的AI车辆列表
    """
    ai_vehicles = []
    bp_library = world.get_blueprint_library()
    ego_transform = ego_vehicle.get_transform()
    
    try:
        for i, config in enumerate(ai_configs):
            # 获取车辆蓝图
            vehicle_bp = bp_library.filter(config['model'])[0]
            
            # 设置随机颜色
            if vehicle_bp.has_attribute('color'):
                color = np.random.choice(vehicle_bp.get_attribute('color').recommended_values)
                vehicle_bp.set_attribute('color', color)
            
            # 计算生成位置
            offset = config['start_offset']
            spawn_location = carla.Location(
                x=ego_transform.location.x + offset[0],
                y=ego_transform.location.y + offset[1],
                z=ego_transform.location.z + offset[2]
            )
            
            spawn_transform = carla.Transform(
                spawn_location,
                carla.Rotation(pitch=0, yaw=0, roll=0)
            )
            
            # 生成AI车辆
            ai_vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
            ai_vehicle.set_autopilot(True)
            
            ai_vehicles.append(ai_vehicle)
            print(f"生成AI车辆 {i+1}: {config['model']}")
            
    except Exception as e:
        print(f"生成AI车辆失败: {e}")
    
    return ai_vehicles

def find_nearby_vehicles(world, ego_vehicle, radius=100):
    """查找附近车辆 - 扩大搜索范围适应直道环境"""
    ego_loc = ego_vehicle.get_location()
    nearby = []
    
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.id == ego_vehicle.id:
            continue
        
        loc = actor.get_location()
        dist = ego_loc.distance(loc)
        
        if dist < radius:
            relative_pos = np.array([
                loc.x - ego_loc.x,
                loc.y - ego_loc.y,
                loc.z - ego_loc.z
            ])
            
            vel = actor.get_velocity()
            relative_vel = np.array([vel.x, vel.y, vel.z])
            
            # 计算相对方位
            angle = np.arctan2(relative_pos[1], relative_pos[0])
            angle_deg = np.degrees(angle)
            
            # 判断车辆位置（前方、后方、左侧、右侧）
            if relative_pos[0] > 0:  # 前方
                if abs(relative_pos[1]) < 2.0:  # 同车道
                    position = "front_same"
                elif relative_pos[1] > 0:  # 左前方
                    position = "front_left"
                else:  # 右前方
                    position = "front_right"
            else:  # 后方
                if abs(relative_pos[1]) < 2.0:  # 同车道
                    position = "rear_same"
                elif relative_pos[1] > 0:  # 左后方
                    position = "rear_left"
                else:  # 右后方
                    position = "rear_right"
            
            nearby.append({
                'id': actor.id,
                'distance': dist,
                'relative_position': relative_pos.tolist(),
                'relative_velocity': relative_vel.tolist(),
                'speed': get_speed(actor),
                'angle': angle_deg,
                'position': position
            })
    
    return sorted(nearby, key=lambda x: x['distance'])

def calculate_lane_change_metrics(ego_vehicle, nearby_vehicles):
    """
    计算变道相关指标
    
    Args:
        ego_vehicle: 主车
        nearby_vehicles: 附近车辆列表
    
    Returns:
        dict: 变道安全指标
    """
    metrics = {
        'left_lane_clear': True,
        'right_lane_clear': True,
        'left_gap_front': float('inf'),
        'left_gap_rear': float('inf'),
        'right_gap_front': float('inf'),
        'right_gap_rear': float('inf'),
        'safe_to_change_left': False,
        'safe_to_change_right': False
    }
    
    # 安全距离阈值
    SAFE_DISTANCE_FRONT = 30.0  # 米
    SAFE_DISTANCE_REAR = 20.0   # 米
    
    for vehicle in nearby_vehicles:
        pos = vehicle['position']
        dist = vehicle['distance']
        
        # 左车道
        if 'left' in pos:
            metrics['left_lane_clear'] = False
            if 'front' in pos:
                metrics['left_gap_front'] = min(metrics['left_gap_front'], dist)
            else:
                metrics['left_gap_rear'] = min(metrics['left_gap_rear'], dist)
        
        # 右车道
        elif 'right' in pos:
            metrics['right_lane_clear'] = False
            if 'front' in pos:
                metrics['right_gap_front'] = min(metrics['right_gap_front'], dist)
            else:
                metrics['right_gap_rear'] = min(metrics['right_gap_rear'], dist)
    
    # 判断是否安全变道
    metrics['safe_to_change_left'] = (
        metrics['left_gap_front'] > SAFE_DISTANCE_FRONT and
        metrics['left_gap_rear'] > SAFE_DISTANCE_REAR
    )
    
    metrics['safe_to_change_right'] = (
        metrics['right_gap_front'] > SAFE_DISTANCE_FRONT and
        metrics['right_gap_rear'] > SAFE_DISTANCE_REAR
    )
    
    return metrics

def get_highway_waypoints(world_map, start_location, distance=500):
    """
    获取高速公路路点序列
    
    Args:
        world_map: CARLA地图对象
        start_location: 起始位置
        distance: 路径长度
    
    Returns:
        list: 路点列表
    """
    waypoints = []
    current_waypoint = world_map.get_waypoint(start_location)
    
    accumulated_distance = 0
    while accumulated_distance < distance:
        waypoints.append(current_waypoint)
        
        # 获取下一个路点
        next_waypoints = current_waypoint.next(2.0)  # 每2米一个路点
        
        if not next_waypoints:
            break
        
        current_waypoint = next_waypoints[0]
        accumulated_distance += 2.0
    
    return waypoints

def analyze_driving_behavior(trajectory_data):
    """
    分析驾驶行为
    
    Args:
        trajectory_data: 轨迹数据DataFrame
    
    Returns:
        dict: 驾驶行为分析结果
    """
    if len(trajectory_data) < 10:
        return {}
    
    # 计算基本统计
    avg_speed = trajectory_data['speed'].mean()
    max_speed = trajectory_data['speed'].max()
    speed_variance = trajectory_data['speed'].var()
    
    # 计算加速度统计
    avg_acceleration = trajectory_data['ax'].mean()
    max_acceleration = trajectory_data['ax'].max()
    min_acceleration = trajectory_data['ax'].min()
    
    # 计算转向统计
    avg_steering = trajectory_data['steer'].abs().mean()
    max_steering = trajectory_data['steer'].abs().max()
    steering_changes = np.diff(trajectory_data['steer']).std()
    
    # 变道次数估计（基于车道ID变化）
    lane_changes = 0
    if 'lane_id' in trajectory_data.columns:
        lane_ids = trajectory_data['lane_id'].values
        for i in range(1, len(lane_ids)):
            if abs(lane_ids[i] - lane_ids[i-1]) >= 1:
                lane_changes += 1
    
    # 行为分类
    if avg_speed > 80 and max_acceleration > 3.0:
        driving_style = "激进"
    elif avg_speed < 60 and max_acceleration < 2.0:
        driving_style = "保守"
    else:
        driving_style = "正常"
    
    return {
        'average_speed': avg_speed,
        'max_speed': max_speed,
        'speed_variance': speed_variance,
        'average_acceleration': avg_acceleration,
        'max_acceleration': max_acceleration,
        'min_acceleration': min_acceleration,
        'average_steering': avg_steering,
        'max_steering': max_steering,
        'steering_smoothness': 1.0 / (1.0 + steering_changes),
        'estimated_lane_changes': lane_changes,
        'driving_style': driving_style
    }