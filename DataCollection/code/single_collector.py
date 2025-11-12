#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import csv
from datetime import datetime
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

SAMPLE_INTERVAL = 10
DATA_FILE = 'data.csv'
ADS1115_ADDRESS = 0x48

def setup_ads1115():
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c, address=ADS1115_ADDRESS)
        ads.gain = 16  # ±0.256V
        print(f"ADS1115初始化成功，地址: 0x{ADS1115_ADDRESS:02X}")
        return ads
    except Exception as e:
        print(f"ADS1115初始化失败: {e}")
        return None

def read_differential_voltage(ads, channel_pair):
    try:
        if channel_pair == '0-1':
            chan = AnalogIn(ads, 0, 1)
        elif channel_pair == '2-3':
            chan = AnalogIn(ads, 2, 3)
        else:
            raise ValueError("无效的通道对，应为 '0-1' 或 '2-3'")
        return chan.voltage
    except Exception as e:
        print(f"读取通道{channel_pair}失败: {e}")
        return None

def initialize_csv_file(filename):
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'DateTime', 'TEC1_Voltage(V)', 'TEC2_Voltage(V)'])
        print(f"CSV文件已初始化: {filename}")
    except Exception as e:
        print(f"CSV文件初始化失败: {e}")

def save_data(filename, timestamp, datetime_str, voltage1, voltage2):
    """保存数据到CSV文件，立即写入SD卡确保数据持久化"""
    try:
        # buffering=1 表示行缓冲，每写入一行立即刷新到磁盘
        with open(filename, 'a', newline='', buffering=1) as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, datetime_str, voltage1, voltage2])
            # 显式刷新缓冲区，确保数据写入SD卡
            f.flush()
            # 同步到磁盘，防止掉电丢失数据
            import os
            os.fsync(f.fileno())
    except Exception as e:
        print(f"数据保存失败: {e}")

def main():
    print("=" * 60)
    print("单ADS1115数据采集系统")
    print("=" * 60)
    print(f"采集间隔: {SAMPLE_INTERVAL}秒")
    print(f"数据文件: {DATA_FILE}")
    print(f"采集通道: TEC1 (A0-A1), TEC2 (A2-A3)")
    print("按Ctrl+C停止采集")
    print("=" * 60)

    ads = setup_ads1115()
    if ads is None:
        print("程序退出")
        return

    initialize_csv_file(DATA_FILE)
    sample_count = 0

    try:
        print("\n开始采集数据...\n")
        while True:
            timestamp = time.time()
            datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            voltage1 = read_differential_voltage(ads, '0-1')
            voltage2 = read_differential_voltage(ads, '2-3')

            if voltage1 is not None and voltage2 is not None:
                save_data(DATA_FILE, timestamp, datetime_str, voltage1, voltage2)
                sample_count += 1
                print(f"[{sample_count:04d}] {datetime_str}")
                print(f"  TEC1: {voltage1:8.6f} V ({voltage1*1000:8.3f} mV)")
                print(f"  TEC2: {voltage2:8.6f} V ({voltage2*1000:8.3f} mV)")
                print()
            else:
                print(f"[{datetime_str}] 数据读取失败，跳过本次采集")

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
