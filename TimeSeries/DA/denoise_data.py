#!/usr/bin/env python3
"""
时序数据降噪脚本 (Time Series Data Denoising)

对CSV数据集进行降噪处理，减少测量噪声和不合理的跳变。
支持两种降噪方法：
1. 异常值修正：检测并修正相对于相邻点的异常值
2. 滑动平均平滑：使用滑动窗口对数据进行平滑处理

降噪顺序：先异常值修正，再滑动平均平滑
支持5秒和10秒采样间隔的数据

作者：GitHub Copilot
日期：2024-12-14
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def detect_outliers_zscore(series: pd.Series, window_size: int = 5, threshold: float = 3.0) -> pd.Series:
    """
    使用局部Z-score检测异常值
    
    Args:
        series: 数据序列
        window_size: 滑动窗口大小
        threshold: Z-score阈值，超过此值视为异常
    
    Returns:
        布尔序列，True表示异常值
    """
    # 计算滚动均值和标准差
    rolling_mean = series.rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window=window_size, center=True, min_periods=1).std()
    
    # 避免除以零
    rolling_std = rolling_std.replace(0, 1e-10)
    
    # 计算局部Z-score
    z_scores = np.abs((series - rolling_mean) / rolling_std)
    
    # 标记异常值
    outliers = z_scores > threshold
    
    return outliers


def correct_outliers(df: pd.DataFrame, data_columns: List[str], 
                     window_size: int = 5, threshold: float = 3.0) -> pd.DataFrame:
    """
    修正异常值（方法1）
    
    检测相对于相邻点的异常跳变，并用局部均值替换
    
    Args:
        df: 原始数据DataFrame
        data_columns: 需要处理的数据列名列表
        window_size: 检测窗口大小（默认5）
        threshold: 异常值阈值（Z-score，默认3.0）
    
    Returns:
        修正后的DataFrame
    """
    df_corrected = df.copy()
    
    total_outliers = 0
    
    for col in data_columns:
        # 检测异常值
        outliers = detect_outliers_zscore(df_corrected[col], window_size, threshold)
        num_outliers = outliers.sum()
        total_outliers += num_outliers
        
        if num_outliers > 0:
            # 使用局部均值替换异常值
            rolling_mean = df_corrected[col].rolling(window=window_size, center=True, min_periods=1).mean()
            df_corrected.loc[outliers, col] = rolling_mean[outliers]
    
    return df_corrected, total_outliers


def moving_average_smooth(df: pd.DataFrame, data_columns: List[str], 
                         window_size: int = 3) -> pd.DataFrame:
    """
    滑动平均平滑（方法2）
    
    使用滑动窗口对数据进行平滑，减少短期噪声
    
    Args:
        df: 原始数据DataFrame
        data_columns: 需要处理的数据列名列表
        window_size: 滑动窗口大小（默认3，表示前后各1个点）
    
    Returns:
        平滑后的DataFrame
    """
    df_smoothed = df.copy()
    
    for col in data_columns:
        # 应用滑动平均，center=True表示当前点位于窗口中心
        df_smoothed[col] = df_smoothed[col].rolling(
            window=window_size, 
            center=True, 
            min_periods=1
        ).mean()
    
    return df_smoothed


def calculate_time_interval(df: pd.DataFrame) -> Optional[int]:
    """
    自动检测数据的时间间隔（5秒或10秒）
    
    Args:
        df: 包含Timestamp列的DataFrame
    
    Returns:
        时间间隔（秒），如果无法确定则返回None
    """
    if 'Timestamp' not in df.columns or len(df) < 2:
        return None
    
    try:
        # 确保Timestamp列是数值类型
        timestamps = pd.to_numeric(df['Timestamp'], errors='coerce')
        
        # 计算前几个时间间隔
        intervals = []
        for i in range(min(10, len(timestamps) - 1)):
            if pd.notna(timestamps.iloc[i]) and pd.notna(timestamps.iloc[i + 1]):
                interval = timestamps.iloc[i + 1] - timestamps.iloc[i]
                intervals.append(interval)
        
        if len(intervals) == 0:
            return None
        
        # 取中位数作为典型间隔
        median_interval = np.median(intervals)
        
        # 判断是5秒还是10秒（允许±2秒的误差）
        if abs(median_interval - 5) <= 2:
            return 5
        elif abs(median_interval - 10) <= 2:
            return 10
        else:
            return int(round(median_interval))
    except Exception:
        return None


def denoise_csv(
    input_path: str,
    output_path: str,
    method: str = 'both',
    outlier_window: int = 5,
    outlier_threshold: float = 3.0,
    smooth_window: int = 3,
    time_interval: Optional[int] = None
) -> Tuple[int, int, int]:
    """
    对单个CSV文件进行降噪处理
    
    Args:
        input_path: 输入CSV文件路径
        output_path: 输出CSV文件路径
        method: 降噪方法 ('outlier', 'smooth', 'both')
        outlier_window: 异常值检测窗口大小
        outlier_threshold: 异常值阈值
        smooth_window: 滑动平均窗口大小
        time_interval: 时间间隔（秒），None表示自动检测
    
    Returns:
        (原始样本数, 修正的异常值数, 检测到的时间间隔)
    """
    # 读取CSV
    df = pd.read_csv(input_path)
    original_samples = len(df)
    
    # 自动检测时间间隔
    if time_interval is None:
        detected_interval = calculate_time_interval(df)
    else:
        detected_interval = time_interval
    
    # 确定数据列（排除时间列）
    data_columns = [col for col in df.columns if col not in ['Timestamp', 'DateTime']]
    
    if len(data_columns) == 0:
        raise ValueError(f"未找到数据列！文件：{input_path}")
    
    # 应用降噪方法
    total_outliers = 0
    
    if method in ['outlier', 'both']:
        # 方法1：修正异常值
        df, total_outliers = correct_outliers(df, data_columns, outlier_window, outlier_threshold)
    
    if method in ['smooth', 'both']:
        # 方法2：滑动平均平滑
        df = moving_average_smooth(df, data_columns, smooth_window)
    
    # 保存结果
    df.to_csv(output_path, index=False)
    
    return original_samples, total_outliers, detected_interval


def process_single_file(
    input_path: str,
    output_dir: str,
    method: str = 'both',
    outlier_window: int = 5,
    outlier_threshold: float = 3.0,
    smooth_window: int = 3,
    time_interval: Optional[int] = None,
    suffix: str = '_denoised'
) -> None:
    """
    处理单个CSV文件
    
    Args:
        input_path: 输入文件路径
        output_dir: 输出目录
        method: 降噪方法
        outlier_window: 异常值检测窗口
        outlier_threshold: 异常值阈值
        smooth_window: 平滑窗口
        time_interval: 时间间隔
        suffix: 输出文件后缀
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构造输出文件名
    input_file = Path(input_path)
    output_filename = input_file.stem + suffix + input_file.suffix
    output_path = os.path.join(output_dir, output_filename)
    
    # 降噪处理
    try:
        samples, outliers, interval = denoise_csv(
            input_path, output_path, method, 
            outlier_window, outlier_threshold, smooth_window, time_interval
        )
        
        interval_str = f"{interval}秒" if interval else "未知"
        print(f"✓ 处理成功: {input_file.name}")
        print(f"  - 样本数: {samples}")
        print(f"  - 时间间隔: {interval_str}")
        print(f"  - 修正异常值: {outliers}个")
        print(f"  - 输出文件: {output_filename}")
        
    except Exception as e:
        print(f"✗ 处理失败: {input_file.name}")
        print(f"  错误: {str(e)}")


def process_directory(
    input_dir: str,
    output_dir: str,
    pattern: str = '*.csv',
    method: str = 'both',
    outlier_window: int = 5,
    outlier_threshold: float = 3.0,
    smooth_window: int = 3,
    time_interval: Optional[int] = None,
    suffix: str = '_denoised'
) -> None:
    """
    批量处理目录下的所有CSV文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        pattern: 文件匹配模式
        method: 降噪方法
        outlier_window: 异常值检测窗口
        outlier_threshold: 异常值阈值
        smooth_window: 平滑窗口
        time_interval: 时间间隔
        suffix: 输出文件后缀
    """
    # 查找所有CSV文件
    input_path = Path(input_dir)
    csv_files = list(input_path.glob(pattern))
    
    if len(csv_files) == 0:
        print(f"错误：在 {input_dir} 中未找到匹配 {pattern} 的文件")
        return
    
    print(f"\n找到 {len(csv_files)} 个CSV文件")
    print(f"降噪方法: {method}")
    print(f"异常值检测窗口: {outlier_window}, 阈值: {outlier_threshold}")
    print(f"平滑窗口: {smooth_window}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个文件
    success_count = 0
    total_outliers = 0
    
    for csv_file in csv_files:
        print(f"\n处理文件: {csv_file.name}")
        
        try:
            # 构造输出路径
            output_filename = csv_file.stem + suffix + csv_file.suffix
            output_path = os.path.join(output_dir, output_filename)
            
            # 降噪处理
            samples, outliers, interval = denoise_csv(
                str(csv_file), output_path, method,
                outlier_window, outlier_threshold, smooth_window, time_interval
            )
            
            interval_str = f"{interval}秒" if interval else "未知"
            print(f"  ✓ 样本数: {samples}")
            print(f"  ✓ 时间间隔: {interval_str}")
            print(f"  ✓ 修正异常值: {outliers}个")
            
            success_count += 1
            total_outliers += outliers
            
        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
    
    print("\n" + "=" * 80)
    print(f"处理完成！成功处理 {success_count}/{len(csv_files)} 个文件")
    print(f"共修正 {total_outliers} 个异常值")
    print(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='时序数据降噪工具 - 减少测量噪声和异常跳变',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 处理单个文件（默认：异常值修正 + 滑动平均）
  python denoise_data.py -i data1122.csv -o ./denoised

  # 批量处理目录
  python denoise_data.py -d ../Prac_data -o ./denoised_data

  # 仅使用异常值修正
  python denoise_data.py -d ../Prac_data -o ./output -m outlier

  # 仅使用滑动平均
  python denoise_data.py -d ../Prac_data -o ./output -m smooth

  # 自定义窗口大小
  python denoise_data.py -d ../Prac_data -o ./output --outlier-window 7 --smooth-window 5

  # 指定时间间隔（5秒或10秒）
  python denoise_data.py -d ../Prac_data -o ./output --time-interval 5

降噪方法说明:
  - outlier: 仅异常值修正（检测并替换异常跳变）
  - smooth: 仅滑动平均（平滑数据曲线）
  - both: 两种方法都使用（推荐，先修正异常值再平滑）

推荐配置:
  - 5秒间隔数据: --outlier-window 5 --smooth-window 3
  - 10秒间隔数据: --outlier-window 5 --smooth-window 3（默认）
        """
    )
    
    # 输入输出参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help='输入CSV文件路径（单文件模式）')
    input_group.add_argument('-d', '--directory', type=str, help='输入目录路径（批量模式）')
    
    parser.add_argument('-o', '--output', type=str, required=True, help='输出目录路径')
    
    # 降噪方法参数
    parser.add_argument('-m', '--method', type=str, default='both',
                       choices=['outlier', 'smooth', 'both'],
                       help='降噪方法：outlier=异常值修正, smooth=滑动平均, both=两者都用（默认）')
    
    # 异常值检测参数
    parser.add_argument('--outlier-window', type=int, default=5,
                       help='异常值检测窗口大小（默认: 5）')
    parser.add_argument('--outlier-threshold', type=float, default=3.0,
                       help='异常值阈值（Z-score，默认: 3.0）')
    
    # 滑动平均参数
    parser.add_argument('--smooth-window', type=int, default=3,
                       help='滑动平均窗口大小（默认: 3）')
    
    # 其他参数
    parser.add_argument('--time-interval', type=int, default=None,
                       choices=[5, 10],
                       help='时间间隔（秒），留空则自动检测')
    parser.add_argument('-p', '--pattern', type=str, default='*.csv',
                       help='文件匹配模式，仅用于目录模式（默认: *.csv）')
    parser.add_argument('-s', '--suffix', type=str, default='_denoised',
                       help='输出文件名后缀（默认: _denoised）')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.outlier_window < 3:
        print("错误：异常值检测窗口大小至少为3")
        sys.exit(1)
    
    if args.smooth_window < 2:
        print("错误：滑动平均窗口大小至少为2")
        sys.exit(1)
    
    # 执行处理
    if args.input:
        # 单文件模式
        if not os.path.exists(args.input):
            print(f"错误：输入文件不存在: {args.input}")
            sys.exit(1)
        
        print(f"\n单文件处理模式")
        print(f"输入文件: {args.input}")
        print(f"降噪方法: {args.method}")
        print("=" * 80 + "\n")
        
        process_single_file(
            args.input, args.output, args.method,
            args.outlier_window, args.outlier_threshold,
            args.smooth_window, args.time_interval, args.suffix
        )
    else:
        # 批量模式
        if not os.path.exists(args.directory):
            print(f"错误：输入目录不存在: {args.directory}")
            sys.exit(1)
        
        print(f"\n批量处理模式")
        print(f"输入目录: {args.directory}")
        print(f"文件模式: {args.pattern}")
        
        process_directory(
            args.directory, args.output, args.pattern,
            args.method, args.outlier_window, args.outlier_threshold,
            args.smooth_window, args.time_interval, args.suffix
        )
    
    print("\n全部完成！")


if __name__ == '__main__':
    main()
