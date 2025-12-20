#!/usr/bin/env python3
"""
模拟数据采集脚本 (mock_collector.py)

该脚本用于在没有 ADS1115 硬件的情况下，模拟 Full_collector.py 的行为：
- 每 10 秒生成一次模拟的 8 通道电压数据
- 将数据写入 CSV 文件，格式与 Full_collector.py 完全一致
- 可以与 pi_sender.py 配合使用，验证 HTTP 转发和 GUI 是否正常工作

使用方式：
    python mock_collector.py
    
    # 指定输出目录
    python mock_collector.py --output-dir /path/to/output
    
    # 指定采集间隔
    python mock_collector.py --interval 5

在树莓派上的部署：
    1. 将此脚本放到与 Full_collector.py 相同的目录
    2. 运行：python mock_collector.py
    3. 同时运行 pi_sender.py 来转发数据
"""

import os
import sys
import csv
import time
import math
import random
import argparse
from datetime import datetime
from pathlib import Path

# 配置参数
DEFAULT_SAMPLE_INTERVAL = 10  # 采集间隔（秒）
DEFAULT_OUTPUT_DIR = "."  # 默认输出目录

# 增益配置（与 Full_collector.py 保持一致）
GAINS_TO_TEST = [16, 8, 4, 2, 1]
COLUMNS_PER_TEC = len(GAINS_TO_TEST) + 2  # 各增益值 + 最优值 + 最优增益


class MockVoltageGenerator:
    """
    模拟电压生成器
    
    生成具有以下特征的模拟电压数据：
    - 基准值在 0.3V ~ 0.7V 之间
    - 包含缓慢变化的趋势
    - 包含随机噪声
    - 偶尔出现突变（模拟环境变化）
    """
    
    # 噪声和突变参数常量
    NOISE_STDDEV = 0.005      # 5mV 标准差的高斯噪声
    SPIKE_PROBABILITY = 0.05  # 5% 突变概率
    SPIKE_STDDEV = 0.02       # 20mV 突变标准差
    
    def __init__(self, num_channels: int = 8):
        """
        初始化模拟电压生成器
        
        参数:
            num_channels: int, 通道数量
        """
        self.num_channels = num_channels
        
        # 每个通道的基准电压（mV 级别的热电效应电压）
        # CSV 中 TEC 通道与颜色的对应关系：
        # TEC1 → Blue, TEC2 → Green, TEC3 → Yellow, TEC4 → Violet
        # TEC5 → Red, TEC6 → Infrared, TEC7 → Ultraviolet, TEC8 → Transparent
        self.base_voltages = [
            0.47,   # TEC1 - Blue
            0.50,   # TEC2 - Green
            0.45,   # TEC3 - Yellow
            0.49,   # TEC4 - Violet
            0.55,   # TEC5 - Red
            0.48,   # TEC6 - Infrared
            0.52,   # TEC7 - Ultraviolet
            0.53,   # TEC8 - Transparent
        ]
        
        # 趋势参数（模拟温度缓慢变化）
        self.trend_phase = [random.uniform(0, 2 * math.pi) for _ in range(num_channels)]
        self.trend_amplitude = [random.uniform(0.02, 0.05) for _ in range(num_channels)]
        self.trend_period = [random.uniform(300, 600) for _ in range(num_channels)]  # 5-10分钟周期
        
        # 计数器
        self.sample_count = 0
        
    def generate(self) -> list:
        """
        生成一组模拟电压数据
        
        返回:
            list of dict, 每个元素包含该通道的多增益电压值和最优值
        """
        self.sample_count += 1
        current_time = time.time()
        
        voltages = []
        for i in range(self.num_channels):
            voltage_data = self._generate_channel_data(i, current_time)
            voltages.append(voltage_data)
        
        return voltages
    
    def _generate_channel_data(self, channel_idx: int, current_time: float) -> dict:
        """
        生成单个通道的电压数据
        
        参数:
            channel_idx: int, 通道索引
            current_time: float, 当前时间戳
        
        返回:
            dict, 包含各增益下的电压值和最优值
        """
        # 计算基准电压（包含趋势）
        trend = self.trend_amplitude[channel_idx] * math.sin(
            2 * math.pi * current_time / self.trend_period[channel_idx] + self.trend_phase[channel_idx]
        )
        base_voltage = self.base_voltages[channel_idx] + trend
        
        # 添加随机噪声
        noise = random.gauss(0, self.NOISE_STDDEV)
        
        # 偶尔添加突变
        spike = 0
        if random.random() < self.SPIKE_PROBABILITY:
            spike = random.gauss(0, self.SPIKE_STDDEV)
        
        optimal_voltage = base_voltage + noise + spike
        
        # 生成各增益下的电压值（模拟真实 ADC 行为）
        result = {}
        for gain in GAINS_TO_TEST:
            # 模拟不同增益下的读数差异
            gain_noise = random.gauss(0, 0.001 / gain)  # 高增益噪声更小
            result[f'gain_{gain}'] = optimal_voltage + gain_noise
        
        # 设置最优值
        result['optimal_voltage'] = optimal_voltage
        result['optimal_gain'] = 16  # 假设增益16是最优的
        
        return result


def initialize_csv_file(filename: str) -> None:
    """
    初始化 CSV 文件，写入表头
    
    参数:
        filename: str, CSV 文件路径
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Timestamp', 'DateTime']
        
        # 为每个 TEC 添加多增益列和最优值列
        for i in range(1, 9):
            tec_name = f'TEC{i}'
            # 各增益下的电压值
            for gain in GAINS_TO_TEST:
                header.append(f'{tec_name}_Gain{gain}(V)')
            # 最优电压值和对应的增益
            header.append(f'{tec_name}_Optimal(V)')
            header.append(f'{tec_name}_OptimalGain')
        
        writer.writerow(header)
    print(f"CSV 文件已初始化: {filename}")


def save_data(filename: str, timestamp: float, datetime_str: str, voltages: list) -> None:
    """
    保存数据到 CSV 文件
    
    参数:
        filename: str, CSV 文件路径
        timestamp: float, Unix 时间戳
        datetime_str: str, 格式化的时间字符串
        voltages: list, 8个通道的电压数据
    """
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [timestamp, datetime_str]
        
        # 为每个 TEC 添加数据
        for voltage_data in voltages:
            if voltage_data is None:
                row.extend([''] * COLUMNS_PER_TEC)
            else:
                # 添加各增益下的电压值
                for gain in GAINS_TO_TEST:
                    v = voltage_data.get(f'gain_{gain}')
                    row.append("" if v is None else f"{v:.9f}")
                
                # 添加最优电压值
                opt_v = voltage_data.get('optimal_voltage')
                row.append("" if opt_v is None else f"{opt_v:.9f}")
                
                # 添加最优增益
                opt_g = voltage_data.get('optimal_gain')
                row.append("" if opt_g is None else str(opt_g))
        
        writer.writerow(row)


def display_voltages(sample_count: int, datetime_str: str, voltages: list) -> None:
    """
    在控制台显示电压数据
    
    参数:
        sample_count: int, 采样计数
        datetime_str: str, 时间字符串
        voltages: list, 电压数据
    """
    # TEC 通道与颜色的对应关系
    # TEC1 → Blue, TEC2 → Green, TEC3 → Yellow, TEC4 → Violet
    # TEC5 → Red, TEC6 → Infrared, TEC7 → Ultraviolet, TEC8 → Transparent
    tec_names = ['TEC1/Blue', 'TEC2/Green', 'TEC3/Yellow', 'TEC4/Violet',
                 'TEC5/Red', 'TEC6/IR', 'TEC7/UV', 'TEC8/Trans']
    
    print(f"\n[{sample_count:04d}] {datetime_str}")
    print("-" * 60)
    
    for i in range(0, 8, 2):
        v1 = voltages[i].get('optimal_voltage', 0)
        v2 = voltages[i+1].get('optimal_voltage', 0)
        print(f"  {tec_names[i]:12s}: {v1:8.6f} V ({v1*1000:8.3f} mV)")
        print(f"  {tec_names[i+1]:12s}: {v2:8.6f} V ({v2*1000:8.3f} mV)")
        if i < 6:
            print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模拟数据采集脚本')
    
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'数据文件输出目录 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--interval', type=float, default=DEFAULT_SAMPLE_INTERVAL,
                       help=f'采集间隔(秒) (默认: {DEFAULT_SAMPLE_INTERVAL})')
    parser.add_argument('--quiet', action='store_true',
                       help='安静模式，不显示每次采集的详细信息')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成数据文件名
    start_time = datetime.now()
    data_file = output_dir / f"TEC_multi_gain_data_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
    
    print("=" * 80)
    print("模拟数据采集系统")
    print("=" * 80)
    print(f"采集间隔: {args.interval}秒")
    print(f"数据文件: {data_file}")
    print(f"采集通道: 8个模拟TEC芯片")
    print("按 Ctrl+C 停止采集")
    print("=" * 80)
    
    # 初始化 CSV 文件
    initialize_csv_file(str(data_file))
    
    # 创建电压生成器
    generator = MockVoltageGenerator()
    
    sample_count = 0
    try:
        print("\n开始模拟数据采集...\n")
        while True:
            timestamp = time.time()
            datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 生成模拟电压数据
            voltages = generator.generate()
            
            # 保存数据
            save_data(str(data_file), timestamp, datetime_str, voltages)
            sample_count += 1
            
            # 显示数据
            if not args.quiet:
                display_voltages(sample_count, datetime_str, voltages)
            else:
                print(f"[{sample_count:04d}] {datetime_str} - 数据已保存")
            
            # 等待下一次采集
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("模拟数据采集已停止")
        print(f"总共采集了 {sample_count} 个样本")
        print(f"数据已保存至: {data_file}")
        print("=" * 80)


if __name__ == "__main__":
    main()
