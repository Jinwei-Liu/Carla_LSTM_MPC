"""主程序入口 - 带个人信息收集"""
import sys
import time
import os
import json
from datetime import datetime
import carla
from data_collector import DataCollector
import config

# 尝试导入tkinter
try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    print("提示: tkinter未安装，将使用终端输入模式")

class UserInfoDialog:
    """用户信息收集对话框"""
    
    def __init__(self):
        self.user_info = {}
        self.window_closed = False
        
    def show_gui_dialog(self):
        """GUI对话框"""
        root = tk.Tk()
        root.title("用户信息收集")
        root.geometry("400x300")
        root.resizable(False, False)
        
        # 居中窗口
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")
        
        # 主框架
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="请填写个人信息", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 输入字段
        fields = [
            ("姓名:", "name", "请输入您的姓名"),
            ("年龄:", "age", "请输入您的年龄"),
            ("性别:", "gender", "男/女/其他"),
            ("驾龄(年):", "driving_years", "驾驶经验年数，无驾照填0")
        ]
        
        entries = {}
        
        for i, (label, key, placeholder) in enumerate(fields, start=1):
            # 标签
            ttk.Label(main_frame, text=label).grid(row=i, column=0, 
                                                   sticky=tk.W, pady=5)
            
            # 输入框
            if key == "gender":
                # 性别下拉框
                entry = ttk.Combobox(main_frame, width=25, 
                                     values=["男", "女", "其他"])
                entry.set("请选择")
            else:
                entry = ttk.Entry(main_frame, width=27)
                entry.insert(0, placeholder)
                
                # 焦点事件
                def on_focus_in(event, entry=entry, placeholder=placeholder):
                    if entry.get() == placeholder:
                        entry.delete(0, tk.END)
                        entry.config(foreground='black')
                
                def on_focus_out(event, entry=entry, placeholder=placeholder):
                    if entry.get() == '':
                        entry.insert(0, placeholder)
                        entry.config(foreground='grey')
                
                entry.bind('<FocusIn>', on_focus_in)
                entry.bind('<FocusOut>', on_focus_out)
                entry.config(foreground='grey')
            
            entry.grid(row=i, column=1, pady=5, padx=(10, 0))
            entries[key] = entry
        
        # 备注框
        ttk.Label(main_frame, text="备注:").grid(row=len(fields)+1, column=0, 
                                                 sticky=tk.W, pady=5)
        notes_text = tk.Text(main_frame, width=27, height=3)
        notes_text.grid(row=len(fields)+1, column=1, pady=5, padx=(10, 0))
        notes_text.insert(1.0, "可选")
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=len(fields)+2, column=0, columnspan=2, pady=20)
        
        def save_info():
            """保存信息"""
            for key, entry in entries.items():
                value = entry.get()
                if key == "gender":
                    if value == "请选择":
                        messagebox.showwarning("提示", "请选择性别")
                        return
                elif value in ["请输入您的姓名", "请输入您的年龄", 
                              "驾驶经验年数，无驾照填0"]:
                    messagebox.showwarning("提示", f"请填写{fields[[f[1] for f in fields].index(key)][0][:-1]}")
                    return
                self.user_info[key] = value
            
            # 备注
            notes = notes_text.get(1.0, tk.END).strip()
            if notes != "可选":
                self.user_info["notes"] = notes
            
            self.user_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            root.destroy()
        
        def cancel():
            """取消"""
            if messagebox.askyesno("确认", "确定要取消吗？程序将退出。"):
                self.window_closed = True
                root.destroy()
        
        ttk.Button(button_frame, text="确定", command=save_info, 
                  width=15).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="取消", command=cancel, 
                  width=15).grid(row=0, column=1, padx=5)
        
        # 窗口关闭事件
        def on_closing():
            self.window_closed = True
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
        
        return not self.window_closed
    
    def show_terminal_dialog(self):
        """终端输入"""
        print("\n" + "="*60)
        print(" " * 20 + "用户信息收集")
        print("="*60)
        print("\n请填写以下个人信息：\n")
        
        try:
            self.user_info["name"] = input("姓名: ").strip() or "未填写"
            
            # 年龄验证
            while True:
                age_input = input("年龄: ").strip()
                if age_input.isdigit():
                    self.user_info["age"] = age_input
                    break
                elif age_input == "":
                    self.user_info["age"] = "未填写"
                    break
                else:
                    print("请输入有效的年龄数字")
            
            # 性别
            print("性别 (1-男, 2-女, 3-其他): ", end="")
            gender_choice = input().strip()
            gender_map = {"1": "男", "2": "女", "3": "其他"}
            self.user_info["gender"] = gender_map.get(gender_choice, "未填写")
            
            # 驾龄
            driving_input = input("驾龄(年，无驾照填0): ").strip()
            self.user_info["driving_years"] = driving_input if driving_input else "0"
            
            self.user_info["notes"] = input("备注(可选): ").strip() or ""
            self.user_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 确认
            print("\n" + "-"*40)
            print("您填写的信息：")
            for key, value in self.user_info.items():
                if key != "timestamp" and value:
                    print(f"  {self.get_field_name(key)}: {value}")
            print("-"*40)
            
            confirm = input("\n确认信息无误？(y/n): ").strip().lower()
            return confirm == 'y'
            
        except KeyboardInterrupt:
            print("\n\n用户取消输入")
            return False
    
    def get_field_name(self, key):
        """字段中文名"""
        field_names = {
            "name": "姓名",
            "age": "年龄",
            "gender": "性别",
            "driving_years": "驾龄",
            "notes": "备注"
        }
        return field_names.get(key, key)
    
    def save_to_file(self, save_path):
        """保存到文件"""
        os.makedirs(save_path, exist_ok=True)
        
        # 文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_path, f"user_info_{timestamp}.txt")
        
        # 保存txt
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write(" " * 15 + "用户信息记录\n")
            f.write("="*50 + "\n\n")
            
            for key, value in self.user_info.items():
                if value:
                    f.write(f"{self.get_field_name(key)}: {value}\n")
            
            f.write("\n" + "="*50 + "\n")
        
        # 保存json
        json_filename = os.path.join(save_path, f"user_info_{timestamp}.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.user_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 用户信息已保存到: {filename}")
        return filename

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
    
    # 收集用户信息
    user_dialog = UserInfoDialog()
    
    if HAS_GUI:
        if not user_dialog.show_gui_dialog():
            print("\n用户取消，程序退出。")
            return
    else:
        if not user_dialog.show_terminal_dialog():
            print("\n用户取消，程序退出。")
            return
    
    # 保存信息
    user_info_file = user_dialog.save_to_file(config.SAVE_PATH)
    
    # 检查连接
    print("\n正在检查CARLA连接...")
    if not check_carla_connection():
        print("\n请先启动CARLA服务器：")
        print("  Windows: CarlaUE4.exe")
        print("  Linux: ./CarlaUE4.sh")
        return
    
    # 显示配置
    print("\n当前配置：")
    print(f"  - 地图: {config.MAP_NAME or '默认'}")
    print(f"  - 车辆: {config.VEHICLE_MODEL}")
    print(f"  - 天气: {config.WEATHER_PRESET}")
    print(f"  - 采集频率: {config.COLLECT_FREQUENCY} Hz")
    print(f"  - Episode长度: {config.EPISODE_LENGTH} 秒")
    print(f"  - 背景车辆: {config.TRAFFIC_VEHICLES} 辆")
    print(f"  - 数据保存: {config.SAVE_PATH}")
    print(f"  - 用户信息: {os.path.basename(user_info_file)}")
    
    # 初始化收集器
    collector = DataCollector()
    
    if hasattr(collector, 'set_user_info'):
        collector.set_user_info(user_dialog.user_info)
    
    try:
        print("\n正在初始化环境...")
        print("-" * 60)
        collector.setup()
        
        print("\n准备就绪！")
        print("-" * 60)
        
        # 倒计时
        for i in range(3, 0, -1):
            print(f"开始收集倒计时: {i}...")
            time.sleep(1)
        
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
        print(f"用户信息文件: {user_info_file}")
        print("="*60)

if __name__ == '__main__':
    main()