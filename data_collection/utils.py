"""工具函数"""
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

def setup_traffic(world, num_vehicles=30, num_walkers=0):
    """生成背景交通流"""
    vehicles = []
    
    try:
        # 获取车辆蓝图
        vehicle_bps = world.get_blueprint_library().filter('vehicle.*')
        vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]
        
        # 获取生成点
        spawn_points = world.get_map().get_spawn_points()
        np.random.shuffle(spawn_points)
        
        # 生成车辆
        for i in range(min(num_vehicles, len(spawn_points)-1)):  # 留一个给主车
            bp = np.random.choice(vehicle_bps)
            
            # 设置随机颜色
            if bp.has_attribute('color'):
                color = np.random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            
            # 尝试生成
            try:
                vehicle = world.spawn_actor(bp, spawn_points[i])
                vehicle.set_autopilot(True)
                vehicles.append(vehicle)
            except:
                continue
        
        print(f"Successfully spawned {len(vehicles)} traffic vehicles")
        
    except Exception as e:
        print(f"Error spawning traffic: {e}")
    
    return vehicles

def find_nearby_vehicles(world, ego_vehicle, radius=50):
    """查找附近车辆"""
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
            
            nearby.append({
                'id': actor.id,
                'distance': dist,
                'relative_position': relative_pos.tolist(),
                'relative_velocity': relative_vel.tolist(),
                'speed': get_speed(actor)
            })
    
    return sorted(nearby, key=lambda x: x['distance'])[:10]