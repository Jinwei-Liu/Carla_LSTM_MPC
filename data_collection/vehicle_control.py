"""车辆控制器 - 包含左右后视镜显示（删除后视镜）"""
import pygame
import carla
import numpy as np
import math
import os
import config
from configparser import ConfigParser
from camera_manager import CameraManager

class WheelControl:
    """方向盘控制器 - 包含左右后视镜显示"""
    
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        
        # 初始化pygame
        pygame.init()
        pygame.joystick.init()
        
        # 检查方向盘连接
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            raise RuntimeError("No steering wheel/joystick found! Please connect your wheel.")
        
        if joystick_count > 1:
            print("Warning: Multiple joysticks detected, using the first one")
        
        # 初始化方向盘
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        
        print(f"Connected wheel: {self._joystick.get_name()}")
        print(f"  Axes: {self._joystick.get_numaxes()}")
        print(f"  Buttons: {self._joystick.get_numbuttons()}")
        
        # 加载方向盘配置
        self._load_wheel_config()
        
        # 初始化控制
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        
        # 设置显示窗口
        display_config = config.DISPLAY_CONFIG
        self.screen = pygame.display.set_mode((display_config['width'], display_config['height']))
        pygame.display.set_caption("CARLA - Steering Wheel Control with Left/Right Mirrors")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        
        # 初始化多相机管理器
        print("Setting up cameras and mirrors...")
        self.camera_manager = CameraManager(world, vehicle)
        
        print("\n" + "="*50)
        print("方向盘控制已启动 (左右后视镜)")
        print("="*50)
        print("显示布局：")
        print("  - 主视角：全屏驾驶员视角")
        print("  - 左后视镜：左上角")
        print("  - 右后视镜：右上角")
        print("  注意：已移除后视镜显示")
        print("控制说明：")
        print("  - 方向盘: 转向")
        print("  - 油门踏板: 加速")
        print("  - 刹车踏板: 减速")
        print("  - 手刹按钮: 手刹")
        print("  - ESC/Q: 退出")
        print("="*50 + "\n")
    
    def _load_wheel_config(self):
        """加载方向盘配置"""
        self._parser = ConfigParser()
        
        # 尝试加载配置文件
        if os.path.exists(config.WHEEL_CONFIG_PATH):
            self._parser.read(config.WHEEL_CONFIG_PATH)
            section = 'G29 Racing Wheel'  # 默认section名
        else:
            # 创建默认配置
            self._create_default_config()
            section = 'G29 Racing Wheel'
        
        try:
            self._steer_idx = int(self._parser.get(section, 'steering_wheel'))
            self._throttle_idx = int(self._parser.get(section, 'throttle'))
            self._brake_idx = int(self._parser.get(section, 'brake'))
            self._reverse_idx = int(self._parser.get(section, 'reverse'))
            self._handbrake_idx = int(self._parser.get(section, 'handbrake'))
        except:
            # 使用默认值
            self._steer_idx = 0
            self._throttle_idx = 1
            self._brake_idx = 2
            self._reverse_idx = 5
            self._handbrake_idx = 0
            print("Warning: Using default wheel axis mapping")
    
    def _create_default_config(self):
        """创建默认配置文件"""
        config_content = """[G29 Racing Wheel]
steering_wheel = 0
throttle = 1
brake = 2
reverse = 5
handbrake = 0
"""
        with open(config.WHEEL_CONFIG_PATH, 'w') as f:
            f.write(config_content)
        print(f"Created default wheel config: {config.WHEEL_CONFIG_PATH}")
    
    def _parse_vehicle_wheel(self):
        """解析方向盘输入"""
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # 方向盘控制 - 使用参考文件的算法
        K1 = 1.0
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])
        steerCmd = max(-1.0, min(1.0, steerCmd))  # 限制范围

        # 油门控制
        K2 = 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        throttleCmd = max(0.0, min(1.0, throttleCmd))

        # 刹车控制
        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        brakeCmd = max(0.0, min(1.0, brakeCmd))

        # 设置控制值
        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd
        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])
        
        # 倒车控制
        if len(jsButtons) > self._reverse_idx:
            if jsButtons[self._reverse_idx]:
                self._control.gear = 1 if self._control.reverse else -1
    
    def update(self):
        """更新控制 - 主循环调用"""
        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return True
        
        # 解析方向盘输入
        self._parse_vehicle_wheel()
        
        # 设置倒车模式
        self._control.reverse = self._control.gear < 0
        
        # 应用控制到车辆
        self.vehicle.apply_control(self._control)
        
        # 渲染画面
        self._render()
        
        # 控制帧率
        self.clock.tick(60)
        
        return False
    
    def _render(self):
        """渲染画面 - 包含左右后视镜"""
        # 使用相机管理器渲染所有视图
        self.camera_manager.render_all_cameras(self.screen)
        
        # 显示HUD信息
        self._render_hud()
        
        pygame.display.flip()
    
    def _render_hud(self):
        """显示HUD信息"""
        # 获取车辆状态
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        transform = self.vehicle.get_transform()
        
        # HUD信息
        info_lines = [
            f"Speed: {speed:.1f} km/h",
            f"Throttle: {self._control.throttle:.2f}",
            f"Brake: {self._control.brake:.2f}", 
            f"Steer: {self._control.steer:.2f}",
            f"Handbrake: {self._control.hand_brake}",
            f"Reverse: {self._control.reverse}",
            f"Location: ({transform.location.x:.1f}, {transform.location.y:.1f})"
        ]
        
        # 半透明背景 - 放在左下角避免遮挡后视镜
        hud_height = len(info_lines) * 22 + 10
        info_surface = pygame.Surface((280, hud_height))
        info_surface.set_alpha(180)
        info_surface.fill((0, 0, 0))
        
        # HUD位置：左下角
        hud_y = self.screen.get_height() - hud_height - 10
        self.screen.blit(info_surface, (10, hud_y))
        
        # 渲染文字
        for i, text in enumerate(info_lines):
            color = (255, 255, 255)
            # 高亮活跃控制
            if "Throttle" in text and self._control.throttle > 0:
                color = (0, 255, 0)
            elif "Brake" in text and self._control.brake > 0:
                color = (255, 0, 0)
            elif "Steer" in text and abs(self._control.steer) > 0.1:
                color = (255, 255, 0)
                
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (15, hud_y + 5 + i * 22))
        
        # 显示设备信息 - 右下角
        device_text = f"Device: {self._joystick.get_name()[:25]}"
        device_surface = self.font.render(device_text, True, (200, 200, 200))
        device_width = device_surface.get_width()
        self.screen.blit(device_surface, (
            self.screen.get_width() - device_width - 10,
            self.screen.get_height() - 25
        ))
        
        # 显示镜像标识（删除后视镜说明）
        mirror_info = self.font.render("Mirrors: L=Left, R=Right (Rear mirror removed)", True, (200, 200, 200))
        mirror_width = mirror_info.get_width()
        self.screen.blit(mirror_info, (
            (self.screen.get_width() - mirror_width) // 2,
            145  # 在后视镜下方位置
        ))
    
    def get_control_data(self):
        """获取控制数据"""
        return {
            'throttle': self._control.throttle,
            'brake': self._control.brake,
            'steer': self._control.steer,
            'hand_brake': self._control.hand_brake,
            'reverse': self._control.reverse
        }
    
    def get_camera_data(self):
        """获取所有相机数据"""
        return self.camera_manager.get_camera_data()
    
    def destroy(self):
        """清理资源"""
        if hasattr(self, 'camera_manager'):
            self.camera_manager.destroy()
        pygame.quit()