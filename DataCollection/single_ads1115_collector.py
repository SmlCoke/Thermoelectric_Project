#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单ADS1115数据采集脚本
采集2个TEC1-04906热电芯片的电压数据
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
ADS1115_ADDRESS = 0x48  # ADS1115的I2C地址

def setup_ads1115():
    """初始化ADS1115"""
    try:
        # 创建I2C总线
        i2c = busio.I2C(board.SCL, board.SDA)
        
        # 创建ADS1115对象
        ads = ADS.ADS1115(i2c, address=ADS1115_ADDRESS)
        
        # 设置增益为16，量程±0.256V，适合TEC芯片的mV级信号
        # GAIN = 16: ±0.256V, 1 bit = 0.0078125mV
        ads.gain = 16
        
        print(f"ADS1115初始化成功，地址: 0x{ADS1115_ADDRESS:02X}")
        return ads
    except Exception as e:
        print(f"ADS1115初始化失败: {e}")
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
            raise ValueError("无效的通道对，应为 '0-1' 或 '2-3'")
        
        return chan.voltage
    except Exception as e:
        print(f"读取通道{channel_pair}失败: {e}")
        return None

def initialize_csv_file(filename):
    """初始化CSV文件，写入表头"""
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头：时间戳、采集时间、TEC1电压、TEC2电压
            writer.writerow(['Timestamp', 'DateTime', 'TEC1_Voltage(V)', 'TEC2_Voltage(V)'])
        print(f"CSV文件已初始化: {filename}")
    except Exception as e:
        print(f"CSV文件初始化失败: {e}")

def save_data(filename, timestamp, datetime_str, voltage1, voltage2):
    """
    保存数据到CSV文件
    
    参数:
        filename: 文件名
        timestamp: Unix时间戳
        datetime_str: 格式化的日期时间字符串
        voltage1: TEC1电压
        voltage2: TEC2电压
    """
    try:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, datetime_str, voltage1, voltage2])
    except Exception as e:
        print(f"数据保存失败: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("单ADS1115数据采集系统")
    print("=" * 60)
    print(f"采集间隔: {SAMPLE_INTERVAL}秒")
    print(f"数据文件: {DATA_FILE}")
    print(f"采集通道: TEC1 (A0-A1), TEC2 (A2-A3)")
    print("按Ctrl+C停止采集")
    print("=" * 60)
    
    # 初始化ADS1115
    ads = setup_ads1115()
    if ads is None:
        print("程序退出")
        return
    
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
            
            # 读取两个TEC芯片的电压
            voltage1 = read_differential_voltage(ads, '0-1')
            voltage2 = read_differential_voltage(ads, '2-3')
            
            # 检查读取是否成功
            if voltage1 is not None and voltage2 is not None:
                # 保存数据
                save_data(DATA_FILE, timestamp, datetime_str, voltage1, voltage2)
                
                # 显示数据
                sample_count += 1
                print(f"[{sample_count:04d}] {datetime_str}")
                print(f"  TEC1: {voltage1:8.6f} V ({voltage1*1000:8.3f} mV)")
                print(f"  TEC2: {voltage2:8.6f} V ({voltage2*1000:8.3f} mV)")
                print()
            else:
                print(f"[{datetime_str}] 数据读取失败，跳过本次采集")
            
            # 等待下一个采集周期
            time.sleep(SAMPLE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("数据采集已停止")
        print(f"总共采集了 {sample_count} 个样本")
        print(f"数据已保存至: {DATA_FILE}")
        print("=" * 60)
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("程序异常退出")

if __name__ == "__main__":
    main()
