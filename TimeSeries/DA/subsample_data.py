# filepath: 
#!/usr/bin/env python3
"""
时序数据降采样脚本 (Temporal Subsampling for Data Augmentation)

修正版功能：
当采样率为 N 时，生成 N 个不同的子数据集。
例如 N=3：
1. Offset 0: 取索引 0, 3, 6...
2. Offset 1: 取索引 1, 4, 7...
3. Offset 2: 取索引 2, 5, 8...

作者：GitHub Copilot
日期：2024-12-14
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple


def subsample_dataframe(df: pd.DataFrame, rate: int, offset: int) -> pd.DataFrame:
    """
    对DataFrame进行降采样
    
    Args:
        df: 原始DataFrame
        rate: 降采样率
        offset: 起始偏移量 (0 到 rate-1)
    
    Returns:
        降采样后的DataFrame
    """
    # 从 offset 开始，每隔 rate 取一个点
    df_subsampled = df.iloc[offset::rate, :].copy()
    
    # 重置索引
    df_subsampled.reset_index(drop=True, inplace=True)
    
    return df_subsampled


def process_single_file(
    input_path: str, 
    output_dir: str, 
    subsample_rates: List[int],
    min_samples: int = 100,
    interval: int = 10,
) -> List[Tuple[str, int, int]]:
    """
    处理单个CSV文件，生成多个降采样版本
    """
    # 获取原始文件名（不含扩展名）
    base_name = Path(input_path).stem
    
    # 读取CSV (只读取一次，提高效率)
    try:
        df_original = pd.read_csv(input_path)
    except Exception as e:
        print(f"读取文件失败 {input_path}: {e}")
        return []

    # 存储生成的文件信息
    generated_files = []
    
    for rate in subsample_rates:
        # === 核心修改：遍历所有可能的起始偏移量 ===
        # 如果 rate=3, range(3) 会产生 0, 1, 2
        for offset in range(rate):
            
            # 降采样
            df_subsampled = subsample_dataframe(df_original, rate, offset)
            num_samples = len(df_subsampled)
            
            # 检查样本数是否足够
            if num_samples < min_samples:
                # 仅在 offset=0 时打印警告，避免刷屏
                if offset == 0:
                    print(f"  ⚠️  N={rate}: 样本不足 ({num_samples} < {min_samples})，跳过")
                continue
            
            # 构建输出文件名
            # 格式: 原名_N{采样率}_offset{偏移量}.csv
            # 例如: data_N3_offset0.csv, data_N3_offset1.csv
            if rate == 1:
                output_filename = f"{base_name}_N1.csv"
            else:
                output_filename = f"{base_name}_N{rate}_offset{offset}.csv"
            
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存文件
            df_subsampled.to_csv(output_path, index=False)
            
            # 记录信息
            generated_files.append((output_path, rate, num_samples))
            
            # 仅打印一次该采样率的信息，或者打印详细信息
            # 这里选择打印简略信息
            # print(f"    -> 生成: offset={offset}, 样本数={num_samples}")

        # 计算实际时间跨度 (基于 offset=0 的样本数估算)
        time_interval = interval * rate
        est_samples = len(df_original) // rate
        time_span_minutes = (est_samples * time_interval) / 60
        print(f"  ✓ N={rate}: 生成 {rate} 个文件 (Offset 0-{rate-1}), 间隔{time_interval}秒")
    
    return generated_files


def process_directory(
    input_dir: str,
    output_dir: str,
    subsample_rates: List[int],
    min_samples: int = 100,
    file_pattern: str = "*.csv",
    interval: int = 10
) -> None:
    """
    批量处理目录中的所有CSV文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有CSV文件
    input_path = Path(input_dir)
    csv_files = list(input_path.glob(file_pattern))
    
    if not csv_files:
        print(f"错误：在 {input_dir} 中未找到匹配 {file_pattern} 的文件")
        return
    
    print(f"\n找到 {len(csv_files)} 个CSV文件")
    print(f"降采样率: {subsample_rates}")
    print(f"策略: 对每个采样率 N，生成 N 个不同相位的数据集")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    
    all_generated_files = []
    
    for csv_file in csv_files:
        print(f"\n处理文件: {csv_file.name}")
        
        # 处理文件
        generated_files = process_single_file(
            str(csv_file),
            output_dir,
            subsample_rates,
            min_samples,
            interval
        )
        
        all_generated_files.extend(generated_files)
    
    # 打印总结
    print("\n" + "=" * 80)
    print(f"处理完成！共生成 {len(all_generated_files)} 个数据集文件")
    
    # 简单统计
    for rate in subsample_rates:
        count = sum(1 for f in all_generated_files if f[1] == rate)
        print(f"  N={rate}: 共生成 {count} 个文件")


def main():
    parser = argparse.ArgumentParser(
        description="时序数据降采样增强脚本 - 生成多相位数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help='输入CSV文件路径')
    input_group.add_argument('-d', '--directory', type=str, help='输入目录路径')
    parser.add_argument('-interval', '--interval', type=int, default=10, help='数据间隔时间 (秒)')
    
    # 输出参数
    parser.add_argument('-o', '--output', type=str, required=True, help='输出目录路径')
    
    
    # 降采样率
    parser.add_argument('-r', '--rates', type=int, nargs='+', default=[1, 2, 3, 5], help='降采样率列表')
    
    # 最小样本数
    parser.add_argument('-m', '--min-samples', type=int, default=100, help='最小样本数阈值')
    parser.add_argument('-p', '--pattern', type=str, default='*.csv', help='文件匹配模式')
    
    args = parser.parse_args()
    
    # 验证降采样率
    if not all(r > 0 for r in args.rates):
        print("错误：降采样率必须大于0")
        sys.exit(1)
    
    subsample_rates = sorted(set(args.rates))
    
    if args.input:
        if not os.path.exists(args.input):
            print(f"错误：输入文件不存在: {args.input}")
            sys.exit(1)
        os.makedirs(args.output, exist_ok=True)
        process_single_file(args.input, args.output, subsample_rates, args.min_samples, args.interval)
    else:
        if not os.path.isdir(args.directory):
            print(f"错误：输入目录不存在: {args.directory}")
            sys.exit(1)
        process_directory(args.directory, args.output, subsample_rates, args.min_samples, args.pattern, args.interval)

if __name__ == "__main__":
    main()