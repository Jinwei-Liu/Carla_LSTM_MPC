"""配置文件 - 删除后视镜相机配置"""
import os
from datetime import datetime

# CARLA连接配置
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
TIMEOUT = 30.0

# 地图和天气
MAP_NAME = 'Town07'  # 使用当前地图
WEATHER_PRESET = 'ClearNoon'

# 车辆配置
VEHICLE_MODEL = 'vehicle.tesla.model3'
SPAWN_POINT_INDEX = None  # None表示随机

# 数据收集配置
COLLECT_FREQUENCY = 20  # Hz
EPISODE_LENGTH = 300  # 秒
TRAFFIC_VEHICLES = 10
TRAFFIC_PEDESTRIANS = 0

# 数据保存选项
SAVE_SENSOR_H5 = False   # 是否保存 sensors.h5 文件
SAVE_CAMERA_H5 = False   # 是否保存 cameras.h5 文件

# 多相机配置
DISPLAY_CONFIG = {
    'width': 1280,
    'height': 720,
    # 左右后视镜显示区域配置（删除后视镜）
    'mirror_config': {
        'left_mirror': {
            'size': (160, 120),
            'position': (10, 10)  # 左上角
        },
        'right_mirror': {
            'size': (160, 120), 
            'position': (1110, 10)  # 右上角
        }
        # 删除了 rear_mirror 配置
    }
}

# 传感器配置 - 删除后视镜相机
SENSORS = {
    # 主驾驶员相机
    'main_camera': {
        'width': 1280,
        'height': 720,
        'fov': 90,
        'position': (1.6, 0.0, 1.7),
        'rotation': (0, 0, 0)
    },
    # 左后视镜相机
    'left_mirror_camera': {
        'width': 320,
        'height': 240,
        'fov': 70,
        'position': (0.5, -1.8, 1.5),  # 左侧位置
        'rotation': (0, -160, 0)  # 向左后方
    },
    # 右后视镜相机
    'right_mirror_camera': {
        'width': 320,
        'height': 240,
        'fov': 70,
        'position': (0.5, 1.8, 1.5),   # 右侧位置
        'rotation': (0, 160, 0)   # 向右后方
    },
    # 删除了 rear_mirror_camera 配置
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