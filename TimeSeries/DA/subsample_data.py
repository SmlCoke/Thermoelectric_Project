#!/usr/bin/env python3
"""
时序数据降采样脚本 (Temporal Subsampling for Data Augmentation)

根据不同的采样率N对CSV数据集进行降采样，生成多个时间尺度的数据集。
原始数据采样间隔为10秒，降采样后间隔变为10×N秒。

用途：
1. 数据增强：将少量数据集扩展为多个不同时间尺度的数据集
2. 多尺度学习：帮助模型学习不同时间尺度的特征
3. 减少噪声：降采样可以减少短期测量噪声的影响

作者：GitHub Copilot
日期：2024-12-12
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple


def subsample_csv(csv_path: str, subsample_rate: int) -> pd.DataFrame:
    """
    对CSV文件进行降采样
    
    Args:
        csv_path: CSV文件路径
        subsample_rate: 降采样率，1表示不降采样，2表示每2个点取1个
    
    Returns:
        降采样后的DataFrame
    """
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    # 进行降采样：每隔subsample_rate个点取一个
    df_subsampled = df.iloc[::subsample_rate, :].copy()
    
    # 重置索引
    df_subsampled.reset_index(drop=True, inplace=True)
    
    return df_subsampled


def process_single_file(
    input_path: str, 
    output_dir: str, 
    subsample_rates: List[int],
    min_samples: int = 100
) -> List[Tuple[str, int, int]]:
    """
    处理单个CSV文件，生成多个降采样版本
    
    Args:
        input_path: 输入CSV文件路径
        output_dir: 输出目录
        subsample_rates: 降采样率列表，如[1, 2, 3, 5]
        min_samples: 最小样本数，小于此值的数据集将被过滤
    
    Returns:
        生成的文件信息列表：[(文件路径, 采样率, 样本数), ...]
    """
    # 获取原始文件名（不含扩展名）
    base_name = Path(input_path).stem
    
    # 存储生成的文件信息
    generated_files = []
    
    for rate in subsample_rates:
        # 降采样
        df_subsampled = subsample_csv(input_path, rate)
        num_samples = len(df_subsampled)
        
        # 检查样本数是否足够
        if num_samples < min_samples:
            print(f"  ⚠️  N={rate}: {num_samples}个样本 (< {min_samples}，跳过)")
            continue
        
        # 构建输出文件名
        if rate == 1:
            output_filename = f"{base_name}_N1_original.csv"
        else:
            output_filename = f"{base_name}_N{rate}_sub{rate}.csv"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存文件
        df_subsampled.to_csv(output_path, index=False)
        
        # 记录信息
        generated_files.append((output_path, rate, num_samples))
        
        # 计算实际时间跨度
        time_interval = 10 * rate  # 秒
        time_span_minutes = (num_samples * time_interval) / 60
        
        print(f"  ✓ N={rate}: {num_samples}个样本, 间隔{time_interval}秒, 跨度{time_span_minutes:.1f}分钟")
    
    return generated_files


def process_directory(
    input_dir: str,
    output_dir: str,
    subsample_rates: List[int],
    min_samples: int = 100,
    file_pattern: str = "*.csv"
) -> None:
    """
    批量处理目录中的所有CSV文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        subsample_rates: 降采样率列表
        min_samples: 最小样本数
        file_pattern: 文件匹配模式
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
    print(f"最小样本数阈值: {min_samples}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    
    all_generated_files = []
    
    for csv_file in csv_files:
        print(f"\n处理文件: {csv_file.name}")
        
        # 读取原始文件信息
        df_original = pd.read_csv(csv_file)
        print(f"  原始样本数: {len(df_original)}")
        
        # 处理文件
        generated_files = process_single_file(
            str(csv_file),
            output_dir,
            subsample_rates,
            min_samples
        )
        
        all_generated_files.extend(generated_files)
    
    # 打印总结
    print("\n" + "=" * 80)
    print(f"处理完成！共生成 {len(all_generated_files)} 个数据集文件")
    print("\n文件列表:")
    
    # 按采样率分组显示
    for rate in subsample_rates:
        files_with_rate = [f for f in all_generated_files if f[1] == rate]
        if files_with_rate:
            total_samples = sum(f[2] for f in files_with_rate)
            print(f"\n  N={rate} (间隔{10*rate}秒): {len(files_with_rate)}个文件, 共{total_samples}个样本")
            for file_path, _, num_samples in files_with_rate:
                print(f"    - {Path(file_path).name}: {num_samples}个样本")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="时序数据降采样脚本 - 根据采样率N生成多尺度数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 处理单个文件（使用默认采样率1,2,3,5）:
   python subsample_data.py -i ../data/data1122.csv -o ./output

2. 处理目录中的所有CSV文件:
   python subsample_data.py -d ../Prac_data -o ./output

3. 自定义采样率:
   python subsample_data.py -d ../Prac_data -o ./output -r 1 2 4 6 10

4. 设置最小样本数阈值:
   python subsample_data.py -d ../Prac_data -o ./output -m 150

5. 快速验证（只生成N=2）:
   python subsample_data.py -i ../data/data1122.csv -o ./test -r 1 2

推荐配置:
  - 快速验证: -r 1 2 3
  - 平衡方案: -r 1 2 3 5 (推荐)
  - 激进方案: -r 1 2 4 6 10
        """
    )
    
    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-i', '--input',
        type=str,
        help='输入CSV文件路径（处理单个文件）'
    )
    input_group.add_argument(
        '-d', '--directory',
        type=str,
        help='输入目录路径（批量处理目录中的所有CSV文件）'
    )
    
    # 输出参数
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='输出目录路径'
    )
    
    # 降采样率
    parser.add_argument(
        '-r', '--rates',
        type=int,
        nargs='+',
        default=[1, 2, 3, 5],
        help='降采样率列表 (默认: 1 2 3 5)'
    )
    
    # 最小样本数
    parser.add_argument(
        '-m', '--min-samples',
        type=int,
        default=100,
        help='最小样本数阈值，低于此值的数据集将被过滤 (默认: 100)'
    )
    
    # 文件匹配模式（仅用于目录模式）
    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default='*.csv',
        help='文件匹配模式，仅在目录模式下使用 (默认: *.csv)'
    )
    
    args = parser.parse_args()
    
    # 验证降采样率
    if not all(r > 0 for r in args.rates):
        print("错误：降采样率必须大于0")
        sys.exit(1)
    
    # 确保降采样率是升序且唯一
    subsample_rates = sorted(set(args.rates))
    
    # 处理文件
    if args.input:
        # 单文件模式
        if not os.path.exists(args.input):
            print(f"错误：输入文件不存在: {args.input}")
            sys.exit(1)
        
        # 创建输出目录
        os.makedirs(args.output, exist_ok=True)
        
        print(f"\n处理单个文件: {args.input}")
        print(f"降采样率: {subsample_rates}")
        print(f"最小样本数阈值: {args.min_samples}")
        print(f"输出目录: {args.output}")
        print("=" * 80)
        
        # 读取原始文件
        df_original = pd.read_csv(args.input)
        print(f"\n原始文件: {Path(args.input).name}")
        print(f"  原始样本数: {len(df_original)}")
        
        # 处理文件
        generated_files = process_single_file(
            args.input,
            args.output,
            subsample_rates,
            args.min_samples
        )
        
        print("\n" + "=" * 80)
        print(f"处理完成！共生成 {len(generated_files)} 个数据集文件")
        
    else:
        # 目录模式
        if not os.path.isdir(args.directory):
            print(f"错误：输入目录不存在: {args.directory}")
            sys.exit(1)
        
        process_directory(
            args.directory,
            args.output,
            subsample_rates,
            args.min_samples,
            args.pattern
        )


if __name__ == "__main__":
    main()
