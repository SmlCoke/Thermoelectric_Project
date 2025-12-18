#!/usr/bin/env python3
"""
生成用于测试的合成数据文件

基于真实数据 ../data/data1122.csv 的模式生成多个合成数据文件。
数据遵循对数衰减模式，模拟热电芯片辐射强度随时间的变化。
"""

import csv
import math
import random
from datetime import datetime, timedelta


def load_real_data(filename):
    """加载真实数据以分析模式"""
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data


def generate_logarithmic_decay(num_points, initial_value, decay_rate, noise_level=0.05):
    """
    生成对数衰减序列
    
    参数:
        num_points: 数据点数量
        initial_value: 初始值
        decay_rate: 衰减率（越大衰减越快）
        noise_level: 噪声水平（相对于当前值的比例）
    
    返回:
        values: 衰减序列
    """
    values = []
    for i in range(num_points):
        # 对数衰减：y = initial * exp(-decay_rate * log(x + 1))
        # 或者等价于：y = initial / ((x + 1) ^ decay_rate)
        base_value = initial_value / math.pow(i + 1, decay_rate)
        
        # 添加噪声
        noise = random.gauss(0, base_value * noise_level)
        value = max(0.0, base_value + noise)  # 确保非负
        
        values.append(value)
    
    return values


def generate_synthetic_data(
    date_str,
    start_time,
    num_samples,
    interval_seconds=10,
    num_channels=8,
    base_timestamp=1763784663
):
    """
    生成合成数据文件
    
    参数:
        date_str: 日期字符串，如 "1201"
        start_time: 开始时间字符串，如 "10:00:00"
        num_samples: 样本数量
        interval_seconds: 采样间隔（秒）
        num_channels: 通道数量
        base_timestamp: 基准时间戳
    """
    # 解析时间
    hour, minute, second = map(int, start_time.split(':'))
    
    # 生成每个通道的初始值和衰减率（基于真实数据的统计）
    # 真实数据的初始值范围大约在 0.002 到 0.008 之间
    channel_params = []
    for i in range(num_channels):
        initial_value = random.uniform(0.003, 0.008)
        decay_rate = random.uniform(0.15, 0.25)  # 衰减率有所变化
        noise_level = random.uniform(0.03, 0.08)  # 噪声水平
        channel_params.append((initial_value, decay_rate, noise_level))
    
    # 生成数据
    rows = []
    current_timestamp = base_timestamp + random.randint(-100000, 100000)  # 添加随机偏移
    
    for i in range(num_samples):
        # 计算时间
        total_seconds = i * interval_seconds
        current_hour = hour + total_seconds // 3600
        current_minute = minute + (total_seconds % 3600) // 60
        current_second = second + total_seconds % 60
        
        # 处理进位
        while current_second >= 60:
            current_second -= 60
            current_minute += 1
        while current_minute >= 60:
            current_minute -= 60
            current_hour += 1
        
        time_str = f"{current_hour:02d}:{current_minute:02d}:{current_second:02d}"
        
        # 生成每个通道的值
        row = {
            'Timestamp': current_timestamp,
            'DateTime': time_str
        }
        
        for ch in range(num_channels):
            initial, decay, noise = channel_params[ch]
            # 使用对数衰减
            base_value = initial / math.pow(i + 1, decay)
            noise_val = random.gauss(0, base_value * noise)
            value = max(0.0, base_value + noise_val)
            
            # 保留足够的小数位
            row[f'TEC{ch+1}_Optimal(V)'] = f"{value:.8f}"
        
        rows.append(row)
        current_timestamp += interval_seconds
    
    # 写入CSV文件
    filename = f"data{date_str}.csv"
    fieldnames = ['Timestamp', 'DateTime'] + [f'TEC{i+1}_Optimal(V)' for i in range(num_channels)]
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ 生成文件: {filename} ({num_samples} 样本)")
    return filename


def main():
    """主函数：生成多个测试数据文件"""
    print("=" * 60)
    print("生成合成测试数据")
    print("=" * 60)
    
    # 定义要生成的数据文件
    # 格式：(日期, 开始时间, 样本数)
    datasets = [
        ("1123", "10:00:00", 250),   # 11月23日，10:00-10:41，250个样本
        ("1124", "11:30:00", 300),   # 11月24日，11:30-12:20，300个样本
        ("1125", "13:00:00", 280),   # 11月25日，13:00-13:46，280个样本
        ("1126", "09:30:00", 320),   # 11月26日，09:30-10:23，320个样本
        ("1127", "14:00:00", 260),   # 11月27日，14:00-14:43，260个样本
        ("1128", "12:00:00", 290),   # 11月28日，12:00-12:48，290个样本
        ("1130", "10:30:00", 310),   # 11月30日，10:30-11:21，310个样本
        ("1201", "13:30:00", 270),   # 12月1日，13:30-14:15，270个样本
        ("1202", "11:00:00", 300),   # 12月2日，11:00-11:50，300个样本
        ("1203", "09:00:00", 330),   # 12月3日，09:00-09:55，330个样本
        ("1204", "14:30:00", 280),   # 12月4日，14:30-15:16，280个样本
        ("1205", "10:00:00", 350),   # 12月5日，10:00-11:00，350个样本
        ("1206", "12:30:00", 290),   # 12月6日，12:30-13:18，290个样本
        ("1207", "11:30:00", 310),   # 12月7日，11:30-12:21，310个样本
        ("1208", "13:00:00", 300),   # 12月8日，13:00-13:50，300个样本
    ]
    
    print(f"\n将生成 {len(datasets)} 个数据文件\n")
    
    generated_files = []
    for date_str, start_time, num_samples in datasets:
        filename = generate_synthetic_data(
            date_str=date_str,
            start_time=start_time,
            num_samples=num_samples,
            interval_seconds=10,
            num_channels=8
        )
        generated_files.append(filename)
    
    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)
    print(f"\n生成的文件列表：")
    for f in generated_files:
        print(f"  - {f}")
    
    print(f"\n总计：{len(generated_files)} 个文件")
    print(f"总样本数：{sum(ds[2] for ds in datasets)}")
    print("\n这些数据可用于：")
    print("  1. 训练模型：足够的数据量进行有效训练")
    print("  2. 验证模型：独立的测试片段")
    print("  3. 测试推理：不同时间段的预测测试")
    print("\n使用方法：")
    print("  cd src")
    print("  python train.py --model gru --num_epochs 100")
    print("  python predict.py --model_path ../checkpoints/best_model.pth \\")
    print("                    --csv_path ../data1205.csv --plot")
    print("=" * 60)


if __name__ == '__main__':
    main()
