"""配置文件 - 简化版直道变道超车环境（优化版）"""
import os
from datetime import datetime

# CARLA连接配置
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
TIMEOUT = 30.0

# 地图和天气
MAP_NAME = 'Town02' 
WEATHER_PRESET = 'ClearNoon'

# 车辆配置
VEHICLE_MODEL = 'vehicle.tesla.model3'

# 直道变道环境配置（优化版）
STRAIGHT_ROAD_CONFIG = {
    # 使用spawn point索引
    'start_spawn_index': 73,      # 使用测试验证的生成点349
    'end_location_coords': {'x': 132.0, 'y': 201.0, 'z': 0.0},
    'completion_distance': 10.0,   # 到达终点的判定距离
    
    # 随机交通配置 - 调整数量避免过多车辆
    'random_traffic': {
        'num_vehicles': 10,        # 减少随机交通车辆数量（避免卡顿）
        'spawn_radius': 2000,       # 生成半径（米）
        'preserve_on_reset': True, # 重置时保留随机交通车辆
    }
}

# 数据收集配置
COLLECT_FREQUENCY = 20  # Hz
EPISODE_LENGTH = 300    # 秒 (增加到5分钟，给更多时间完成)
TRAFFIC_VEHICLES = 10   # 随机交通车辆数量（与上面保持一致）
TRAFFIC_PEDESTRIANS = 0  # 不生成行人

# 数据保存选项
SAVE_SENSOR_H5 = False   # 是否保存传感器H5文件
SAVE_CAMERA_H5 = False   # 是否保存相机H5文件（关闭以节省空间）

# 显示配置
DISPLAY_CONFIG = {
    'width': 1920,
    'height': 1080,
    'mirror_config': {
        'left_mirror': {
            'size': (160, 120),
            'position': (10, 10)
        },
        'right_mirror': {
            'size': (160, 120), 
            'position': (1750, 10)
        }
        # 已删除rear_mirror配置
    }
}

# 传感器配置（删除后视镜相机）
SENSORS = {
    'main_camera': {
        'width': 1920,
        'height': 1080,
        'fov': 90,
        'position': (1.6, 0.0, 1.7),
        'rotation': (0, 0, 0)
    },
    'left_mirror_camera': {
        'width': 320,
        'height': 240,
        'fov': 70,
        'position': (0.5, -1.8, 1.5),
        'rotation': (0, -160, 0)
    },
    'right_mirror_camera': {
        'width': 320,
        'height': 240,
        'fov': 70,
        'position': (0.5, 1.8, 1.5),
        'rotation': (0, 160, 0)
    },
    # 后视镜相机已删除
    'gnss': {
        'position': (0.0, 0.0, 0.0)
    },
    'imu': {
        'position': (0.0, 0.0, 0.0)
    }
}

# 数据保存路径
DATA_ROOT = './collected_data'
SESSION_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
SAVE_PATH = os.path.join(DATA_ROOT, SESSION_ID)

# 方向盘配置文件路径
WHEEL_CONFIG_PATH = 'wheel_config.ini'

# 重置策略配置
RESET_CONFIG = {
    'reset_ego_vehicle': True,          # 重置主车位置
    'reset_static_obstacle': True,      # 重置静止障碍车
    'reset_random_traffic': False,      # 不重置随机交通（让它们继续跑）
}

# 性能优化配置
PERFORMANCE_CONFIG = {
    'traffic_manager_port': 8000,       # Traffic Manager端口
    'traffic_manager_sync': False,      # 是否同步模式
    'traffic_hybrid_mode': True,        # 混合物理模式
    'global_distance_to_leading': 2.0,  # 全局车距设置
}