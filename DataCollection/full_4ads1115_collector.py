#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整4个ADS1115数据采集脚本
采集8个TEC1-04906热电芯片的电压数据
采集频率：10秒一次
数据保存：data.csv
"""

import time
import csv
from datetime import datetime
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# 配置参数
SAMPLE_INTERVAL = 10  # 采集间隔（秒）
DATA_FILE = 'data.csv'  # 数据文件路径

# 4个ADS1115的I2C地址
ADS1115_ADDRESSES = {
    'ADS1': 0x48,  # ADDR -> GND
    'ADS2': 0x49,  # ADDR -> VDD
    'ADS3': 0x4A,  # ADDR -> SDA
    'ADS4': 0x4B,  # ADDR -> SCL
}

def setup_all_ads1115():
    """初始化所有4个ADS1115"""
    ads_devices = {}
    i2c = None
    
    try:
        # 创建I2C总线
        i2c = busio.I2C(board.SCL, board.SDA)
        
        # 初始化每个ADS1115
        for name, address in ADS1115_ADDRESSES.items():
            try:
                ads = ADS.ADS1115(i2c, address=address)
                # 设置增益为16，量程±0.256V
                ads.gain = 16
                ads_devices[name] = ads
                print(f"{name} 初始化成功，地址: 0x{address:02X}")
            except Exception as e:
                print(f"{name} (0x{address:02X}) 初始化失败: {e}")
                
        if len(ads_devices) == 0:
            print("错误：没有成功初始化任何ADS1115设备")
            return None
        elif len(ads_devices) < 4:
            print(f"警告：只成功初始化了 {len(ads_devices)}/4 个ADS1115设备")
            
        return ads_devices
        
    except Exception as e:
        print(f"I2C总线初始化失败: {e}")
        return None

def read_differential_voltage(ads, channel_pair):
    """
    读取差分电压
    
    参数:
        ads: ADS1115对象
        channel_pair: 通道对，'0-1' 或 '2-3'
    
    返回:
        电压值（V）
    """
    try:
        if channel_pair == '0-1':
            chan = AnalogIn(ads, ADS.P0, ADS.P1)
        elif channel_pair == '2-3':
            chan = AnalogIn(ads, ADS.P2, ADS.P3)
        else:
            raise ValueError("无效的通道对")
        
        return chan.voltage
    except Exception as e:
        print(f"读取通道{channel_pair}失败: {e}")
        return None

def read_all_voltages(ads_devices):
    """
    读取所有8个TEC芯片的电压
    
    参数:
        ads_devices: ADS1115设备字典
    
    返回:
        包含8个电压值的列表，如果读取失败则对应位置为None
    """
    voltages = []
    
    # ADS1 - TEC1 和 TEC2
    if 'ADS1' in ads_devices:
        voltages.append(read_differential_voltage(ads_devices['ADS1'], '0-1'))  # TEC1
        voltages.append(read_differential_voltage(ads_devices['ADS1'], '2-3'))  # TEC2
    else:
        voltages.extend([None, None])
    
    # ADS2 - TEC3 和 TEC4
    if 'ADS2' in ads_devices:
        voltages.append(read_differential_voltage(ads_devices['ADS2'], '0-1'))  # TEC3
        voltages.append(read_differential_voltage(ads_devices['ADS2'], '2-3'))  # TEC4
    else:
        voltages.extend([None, None])
    
    # ADS3 - TEC5 和 TEC6
    if 'ADS3' in ads_devices:
        voltages.append(read_differential_voltage(ads_devices['ADS3'], '0-1'))  # TEC5
        voltages.append(read_differential_voltage(ads_devices['ADS3'], '2-3'))  # TEC6
    else:
        voltages.extend([None, None])
    
    # ADS4 - TEC7 和 TEC8
    if 'ADS4' in ads_devices:
        voltages.append(read_differential_voltage(ads_devices['ADS4'], '0-1'))  # TEC7
        voltages.append(read_differential_voltage(ads_devices['ADS4'], '2-3'))  # TEC8
    else:
        voltages.extend([None, None])
    
    return voltages

def initialize_csv_file(filename):
    """初始化CSV文件，写入表头"""
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            header = ['Timestamp', 'DateTime',
                     'TEC1_Voltage(V)', 'TEC2_Voltage(V)',
                     'TEC3_Voltage(V)', 'TEC4_Voltage(V)',
                     'TEC5_Voltage(V)', 'TEC6_Voltage(V)',
                     'TEC7_Voltage(V)', 'TEC8_Voltage(V)']
            writer.writerow(header)
        print(f"CSV文件已初始化: {filename}")
    except Exception as e:
        print(f"CSV文件初始化失败: {e}")

def save_data(filename, timestamp, datetime_str, voltages):
    """
    保存数据到CSV文件
    
    参数:
        filename: 文件名
        timestamp: Unix时间戳
        datetime_str: 格式化的日期时间字符串
        voltages: 包含8个电压值的列表
    """
    try:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [timestamp, datetime_str] + voltages
            writer.writerow(row)
    except Exception as e:
        print(f"数据保存失败: {e}")

def display_voltages(sample_count, datetime_str, voltages):
    """
    显示电压数据
    
    参数:
        sample_count: 样本计数
        datetime_str: 时间字符串
        voltages: 电压列表
    """
    print(f"[{sample_count:04d}] {datetime_str}")
    
    # 分4行显示，每行显示2个TEC芯片（对应一个ADS1115）
    tec_names = ['TEC1', 'TEC2', 'TEC3', 'TEC4', 'TEC5', 'TEC6', 'TEC7', 'TEC8']
    
    for i in range(0, 8, 2):
        if voltages[i] is not None and voltages[i+1] is not None:
            print(f"  {tec_names[i]}: {voltages[i]:8.6f} V ({voltages[i]*1000:8.3f} mV)  |  "
                  f"{tec_names[i+1]}: {voltages[i+1]:8.6f} V ({voltages[i+1]*1000:8.3f} mV)")
        else:
            print(f"  {tec_names[i]}: 读取失败  |  {tec_names[i+1]}: 读取失败")
    
    print()

def main():
    """主函数"""
    print("=" * 80)
    print("完整4个ADS1115数据采集系统")
    print("=" * 80)
    print(f"采集间隔: {SAMPLE_INTERVAL}秒")
    print(f"数据文件: {DATA_FILE}")
    print(f"采集通道: 8个TEC芯片")
    print("  ADS1115 #1 (0x48): TEC1 (A0-A1), TEC2 (A2-A3)")
    print("  ADS1115 #2 (0x49): TEC3 (A0-A1), TEC4 (A2-A3)")
    print("  ADS1115 #3 (0x4A): TEC5 (A0-A1), TEC6 (A2-A3)")
    print("  ADS1115 #4 (0x4B): TEC7 (A0-A1), TEC8 (A2-A3)")
    print("按Ctrl+C停止采集")
    print("=" * 80)
    
    # 初始化所有ADS1115
    ads_devices = setup_all_ads1115()
    if ads_devices is None or len(ads_devices) == 0:
        print("程序退出")
        return
    
    print()
    
    # 初始化CSV文件
    initialize_csv_file(DATA_FILE)
    
    # 数据采集计数器
    sample_count = 0
    
    try:
        print("\n开始采集数据...\n")
        
        while True:
            # 记录当前时间
            timestamp = time.time()
            datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 读取所有TEC芯片的电压
            voltages = read_all_voltages(ads_devices)
            
            # 检查是否至少有一个成功读取
            if any(v is not None for v in voltages):
                # 保存数据
                save_data(DATA_FILE, timestamp, datetime_str, voltages)
                
                # 显示数据
                sample_count += 1
                display_voltages(sample_count, datetime_str, voltages)
            else:
                print(f"[{datetime_str}] 所有通道读取失败，跳过本次采集")
            
            # 等待下一个采集周期
            time.sleep(SAMPLE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("数据采集已停止")
        print(f"总共采集了 {sample_count} 个样本")
        print(f"数据已保存至: {DATA_FILE}")
        print("=" * 80)
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("程序异常退出")

if __name__ == "__main__":
    main()
