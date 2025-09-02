"""传感器管理器"""
import carla
import numpy as np
import weakref
from collections import defaultdict
import config

class SensorManager:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.sensors = {}
        self.data = defaultdict(lambda: None)
        
    def setup_sensors(self):
        """初始化所有传感器"""
        # RGB相机
        if 'rgb_camera' in config.SENSORS:
            self._create_rgb_camera()
        
        # LIDAR
        if 'lidar' in config.SENSORS:
            self._create_lidar()
        
        # GNSS
        if 'gnss' in config.SENSORS:
            self._create_gnss()
        
        # IMU
        if 'imu' in config.SENSORS:
            self._create_imu()
    
    def _create_rgb_camera(self):
        """创建RGB相机"""
        cfg = config.SENSORS['rgb_camera']
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(cfg['width']))
        bp.set_attribute('image_size_y', str(cfg['height']))
        bp.set_attribute('fov', str(cfg['fov']))
        
        transform = carla.Transform(
            carla.Location(x=cfg['position'][0], y=cfg['position'][1], z=cfg['position'][2]),
            carla.Rotation(pitch=cfg['rotation'][0], yaw=cfg['rotation'][1], roll=cfg['rotation'][2])
        )
        
        camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        camera.listen(lambda image: self._process_image(image))
        self.sensors['rgb_camera'] = camera
    
    def _create_lidar(self):
        """创建激光雷达"""
        cfg = config.SENSORS['lidar']
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('channels', str(cfg['channels']))
        bp.set_attribute('range', str(cfg['range']))
        bp.set_attribute('rotation_frequency', str(cfg['rotation_frequency']))
        bp.set_attribute('points_per_second', str(cfg['points_per_second']))
        
        transform = carla.Transform(
            carla.Location(x=cfg['position'][0], y=cfg['position'][1], z=cfg['position'][2]),
            carla.Rotation(pitch=cfg['rotation'][0], yaw=cfg['rotation'][1], roll=cfg['rotation'][2])
        )
        
        lidar = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        lidar.listen(lambda data: self._process_lidar(data))
        self.sensors['lidar'] = lidar
    
    def _create_gnss(self):
        """创建GNSS"""
        cfg = config.SENSORS['gnss']
        bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        transform = carla.Transform(carla.Location(x=cfg['position'][0], y=cfg['position'][1], z=cfg['position'][2]))
        
        gnss = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        gnss.listen(lambda data: self._process_gnss(data))
        self.sensors['gnss'] = gnss
    
    def _create_imu(self):
        """创建IMU"""
        cfg = config.SENSORS['imu']
        bp = self.world.get_blueprint_library().find('sensor.other.imu')
        transform = carla.Transform(carla.Location(x=cfg['position'][0], y=cfg['position'][1], z=cfg['position'][2]))
        
        imu = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        imu.listen(lambda data: self._process_imu(data))
        self.sensors['imu'] = imu
    
    def _process_image(self, image):
        """处理图像数据"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # RGB
        self.data['rgb_camera'] = array
    
    def _process_lidar(self, data):
        """处理激光雷达数据"""
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 4])
        self.data['lidar'] = points[:, :3]  # XYZ
    
    def _process_gnss(self, data):
        """处理GNSS数据"""
        self.data['gnss'] = [data.latitude, data.longitude, data.altitude]
    
    def _process_imu(self, data):
        """处理IMU数据"""
        self.data['imu'] = {
            'accelerometer': [data.accelerometer.x, data.accelerometer.y, data.accelerometer.z],
            'gyroscope': [data.gyroscope.x, data.gyroscope.y, data.gyroscope.z],
            'compass': data.compass
        }
    
    def get_data(self):
        """获取所有传感器数据"""
        return dict(self.data)
    
    def destroy(self):
        """销毁所有传感器"""
        for sensor in self.sensors.values():
            sensor.destroy()