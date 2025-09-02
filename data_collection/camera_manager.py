"""多相机管理器 - 主视角+左右后视镜（删除后视镜）"""
import carla
import numpy as np
import weakref
import pygame
import config

class CameraManager:
    """管理主相机和左右后视镜相机"""
    
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        
        # 相机对象
        self.cameras = {}
        self.camera_images = {}
        
        # 初始化所有相机图像为None（删除rear_mirror_camera）
        self.camera_images = {
            'main_camera': None,
            'left_mirror_camera': None,
            'right_mirror_camera': None
        }
        
        # 设置所有相机
        self._setup_all_cameras()
    
    def _setup_all_cameras(self):
        """设置所有相机"""
        bp_library = self.world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        
        # 为每个相机创建传感器
        for camera_name, camera_config in config.SENSORS.items():
            if not camera_name.endswith('_camera'):
                continue
            
            # 跳过后视镜相机（已从config中删除，但为了安全起见保留检查）
            if camera_name == 'rear_mirror_camera':
                continue
                
            # 配置相机参数
            camera_bp.set_attribute('image_size_x', str(camera_config['width']))
            camera_bp.set_attribute('image_size_y', str(camera_config['height']))
            camera_bp.set_attribute('fov', str(camera_config['fov']))
            
            # 设置相机位置和旋转
            camera_transform = carla.Transform(
                carla.Location(
                    x=camera_config['position'][0],
                    y=camera_config['position'][1],
                    z=camera_config['position'][2]
                ),
                carla.Rotation(
                    pitch=camera_config['rotation'][0],
                    yaw=camera_config['rotation'][1],
                    roll=camera_config['rotation'][2]
                )
            )
            
            # 创建相机传感器
            camera_sensor = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )
            
            # 设置回调函数
            weak_self = weakref.ref(self)
            camera_sensor.listen(
                lambda image, cam_name=camera_name: 
                CameraManager._parse_image(weak_self, image, cam_name)
            )
            
            self.cameras[camera_name] = camera_sensor
            print(f"✓ Camera setup: {camera_name}")
    
    @staticmethod
    def _parse_image(weak_self, image, camera_name):
        """解析相机图像"""
        self = weak_self()
        if not self:
            return
            
        # 转换图像数据
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # 移除Alpha通道
        array = array[:, :, ::-1]  # BGR -> RGB
        
        # 转换为pygame surface
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        # 存储图像
        self.camera_images[camera_name] = surface
    
    def render_all_cameras(self, screen):
        """在屏幕上渲染所有相机视图"""
        display_config = config.DISPLAY_CONFIG
        
        # 渲染主相机（全屏背景）
        if self.camera_images['main_camera'] is not None:
            screen.blit(self.camera_images['main_camera'], (0, 0))
        else:
            screen.fill((50, 50, 50))  # 灰色背景
        
        # 渲染左右后视镜（不包括后视镜）
        self._render_mirrors(screen)
        
        # 添加后视镜边框
        self._draw_mirror_borders(screen)
    
    def _render_mirrors(self, screen):
        """渲染左右后视镜（删除后视镜）"""
        mirror_config = config.DISPLAY_CONFIG['mirror_config']
        
        # 左后视镜
        if self.camera_images['left_mirror_camera'] is not None:
            left_surface = pygame.transform.scale(
                self.camera_images['left_mirror_camera'],
                mirror_config['left_mirror']['size']
            )
            screen.blit(left_surface, mirror_config['left_mirror']['position'])
        
        # 右后视镜
        if self.camera_images['right_mirror_camera'] is not None:
            right_surface = pygame.transform.scale(
                self.camera_images['right_mirror_camera'],
                mirror_config['right_mirror']['size']
            )
            screen.blit(right_surface, mirror_config['right_mirror']['position'])
        
        # 删除了后视镜渲染代码
    
    def _draw_mirror_borders(self, screen):
        """绘制左右后视镜边框（删除后视镜）"""
        mirror_config = config.DISPLAY_CONFIG['mirror_config']
        border_color = (255, 255, 255)
        border_width = 2
        
        # 左后视镜边框
        left_rect = pygame.Rect(
            mirror_config['left_mirror']['position'],
            mirror_config['left_mirror']['size']
        )
        pygame.draw.rect(screen, border_color, left_rect, border_width)
        
        # 右后视镜边框
        right_rect = pygame.Rect(
            mirror_config['right_mirror']['position'],
            mirror_config['right_mirror']['size']
        )
        pygame.draw.rect(screen, border_color, right_rect, border_width)
        
        # 删除了后视镜边框代码
        
        # 添加标签
        font = pygame.font.Font(None, 16)
        
        # 左镜标签
        left_label = font.render("Left", True, (255, 255, 255))
        screen.blit(left_label, (
            mirror_config['left_mirror']['position'][0] + 5,
            mirror_config['left_mirror']['position'][1] + 
            mirror_config['left_mirror']['size'][1] + 2
        ))
        
        # 右镜标签
        right_label = font.render("Right", True, (255, 255, 255))
        screen.blit(right_label, (
            mirror_config['right_mirror']['position'][0] + 5,
            mirror_config['right_mirror']['position'][1] + 
            mirror_config['right_mirror']['size'][1] + 2
        ))
        
        # 删除了后镜标签代码
    
    def get_camera_data(self):
        """获取所有相机的图像数据"""
        camera_data = {}
        
        for camera_name, surface in self.camera_images.items():
            if surface is not None:
                # 转换pygame surface为numpy array
                array = pygame.surfarray.array3d(surface)
                array = array.swapaxes(0, 1)
                array = array[:, :, ::-1]  # RGB -> BGR
                camera_data[camera_name] = array
            else:
                camera_data[camera_name] = None
        
        return camera_data
    
    def destroy(self):
        """销毁所有相机"""
        for camera in self.cameras.values():
            if camera is not None:
                camera.destroy()
        print("All cameras destroyed")