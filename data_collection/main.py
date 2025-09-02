"""主程序入口 - 终端控制版"""
import sys
import time
import carla
from data_collector import DataCollector
import config

def check_carla_connection():
    """检查CARLA连接"""
    try:
        client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
        client.set_timeout(5.0)
        version = client.get_server_version()
        print(f"✓ CARLA服务器已连接 (版本: {version})")
        return True
    except:
        print("✗ 无法连接到CARLA服务器")
        return False

def main():
    """主函数"""
    print("\n" + "="*60)
    print(" " * 15 + "CARLA 数据收集工具")
    print(" " * 15 + "  (终端控制版)")
    print("="*60)
    
    # 检查连接
    print("\n正在检查CARLA连接...")
    if not check_carla_connection():
        print("\n请先启动CARLA服务器：")
        print("  Windows: CarlaUE4.exe")
        print("  Linux: ./CarlaUE4.sh")
        return
    
    # 显示配置信息
    print("\n当前配置：")
    print(f"  - 地图: {config.MAP_NAME or '默认'}")
    print(f"  - 车辆: {config.VEHICLE_MODEL}")
    print(f"  - 天气: {config.WEATHER_PRESET}")
    print(f"  - 采集频率: {config.COLLECT_FREQUENCY} Hz")
    print(f"  - Episode长度: {config.EPISODE_LENGTH} 秒")
    print(f"  - 背景车辆: {config.TRAFFIC_VEHICLES} 辆")
    print(f"  - 数据保存: {config.SAVE_PATH}")
    
    # 初始化数据收集器
    collector = DataCollector()
    
    try:
        print("\n正在初始化环境...")
        print("-" * 60)
        collector.setup()
        
        # 开始收集
        print("\n准备就绪！")
        print("-" * 60)
        
        # 倒计时
        for i in range(3, 0, -1):
            print(f"开始收集倒计时: {i}...")
            time.sleep(1)
        
        # 运行主循环
        collector.run()
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n" + "="*60)
        print("数据收集已结束")
        if hasattr(collector, 'data_saver') and collector.data_saver:
            print(f"数据保存位置: {collector.data_saver.save_path}")
        print("="*60)

if __name__ == '__main__':
    main()