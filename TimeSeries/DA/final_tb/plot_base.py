import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
from datetime import datetime, timedelta
from matplotlib.font_manager import FontProperties
import matplotlib.dates as mdates

# 定义字体
T_16 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 16)    # Times New Roman
T_14 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 14)    # Times New Roman
T_12 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 12)    # Times New Roman

# 设置绘图风格
plt.rcParams['axes.unicode_minus'] = False

# 通道颜色映射
CHANNEL_COLORS = {
    "Yellow": "gold",
    "Ultraviolet": "purple",
    "Infrared": "firebrick",
    "Red": "red",
    "Green": "green",
    "Blue": "blue",
    "Transparent": "gray",
    "Violet": "violet",
}


def get_channel_color(channel):
    """Return the configured color for a channel, fall back to black."""
    return CHANNEL_COLORS.get(channel, "black")


def build_time_axis(total_len, start_time_str, interval_seconds):
    """Construct a datetime x-axis if time info is provided."""
    if start_time_str is None and interval_seconds is None:
        return None
    if not start_time_str or interval_seconds is None:
        print("错误：必须同时提供 --start_time 与 --time_interval 才能显示时间轴。")
        sys.exit(1)
    try:
        base_time = datetime.strptime(start_time_str, "%H:%M:%S")
    except ValueError:
        print("错误：--start_time 格式必须为 时:分:秒，例如 12:36:26")
        sys.exit(1)
    if interval_seconds <= 0:
        print("错误：--time_interval 必须大于 0")
        sys.exit(1)
    step = timedelta(seconds=interval_seconds)
    return [base_time + i * step for i in range(total_len)]


def format_time_axis(ax, x_is_time=False):
    """Apply HH:MM formatter when using datetime x-axis."""
    if x_is_time:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

def plot_single_channel(ax, x_axis, slice_orig, channel, show_legend=True, x_is_time=False):
    """辅助函数：在指定的 ax 上绘制单个通道"""
    if channel in slice_orig.columns:
        # 绘制原始数据
        ax.plot(x_axis, slice_orig[channel], color=get_channel_color(channel), alpha=0.8, linewidth=1, label='Original(Raw)')
        
        # 使用 T_14 设置子标题
        ax.set_title(f'{channel} Channel', fontproperties=T_14)
        ax.set_ylabel('mV', fontproperties=T_12)
        ax.grid(True, alpha=0.3)
        format_time_axis(ax, x_is_time)
        
        # 设置坐标轴刻度字体为 T_12
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(T_12)
        
        # 显示图例
        if show_legend:
            ax.legend(loc='upper right', prop=T_12)
    else:
        ax.text(0.5, 0.5, f'Column {channel} not found', ha='center', va='center', fontproperties=T_12)

def plot(csv_path, start_ratio=0.0, end_ratio=1.0, target_channel=None, save_path=None,
         start_time=None, time_interval=None):
    # 1. 读取数据
    if not os.path.exists(csv_path) :
        print("错误：找不到输入文件。")
        return

    df = pd.read_csv(csv_path)


    # 2. 确定切片范围
    total_len = len(df)
    time_axis_full = build_time_axis(total_len, start_time, time_interval)
    start_idx = int(total_len * start_ratio)
    end_idx = int(total_len * end_ratio)

    # 边界检查
    if start_idx < 0: start_idx = 0
    if end_idx > total_len: end_idx = total_len
    if start_idx >= end_idx:
        print("错误：起始点必须小于结束点。")
        return

    print(f"正在绘图... 显示范围: {start_ratio*100}% - {end_ratio*100}% (索引: {start_idx} - {end_idx})")

    # 切片数据
    slice = df.iloc[start_idx:end_idx].reset_index(drop=True)

    # 生成X轴坐标
    x_axis = time_axis_full[start_idx:end_idx] if time_axis_full else list(range(start_idx, end_idx))
    x_is_time = time_axis_full is not None

    # 3. 绘图逻辑
    if target_channel:
        # --- 单通道模式 ---
        if target_channel not in slice.columns:
            print(f"错误：在数据中找不到通道 '{target_channel}'")
            print(f"可用通道: {list(slice.columns)}")
            return
        # 创建单张大图
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_single_channel(ax, x_axis, slice,  target_channel, show_legend=True, x_is_time=x_is_time)
        
        title_text = f'Radiation Variation - {target_channel} ({start_ratio}-{end_ratio})'
    else:
        # --- 全通道模式 (8张图) ---
        channels = [
            "Yellow",
            "Ultraviolet",
            "Infrared",
            "Red",
            "Green",
            "Blue",
            "Transparent",
            "Violet"
        ]
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        axes = axes.flatten() 

        for i, channel in enumerate(channels):
            # 只在第一张图显示图例
            plot_single_channel(axes[i], x_axis, slice, channel, show_legend=(i==0), x_is_time=x_is_time)

        title_text = f'Radiation Variation ({start_ratio}-{end_ratio} range)'

    # 4. 布局调整与保存
    plt.suptitle(title_text, fontproperties=T_16)
    
    # 根据模式调整布局
    if target_channel:
        plt.tight_layout()
    else:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图片已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化降噪效果对比脚本")
    
    # 必须参数
    parser.add_argument('--csv_path', "-p", type=str, required=True, help='CSV文件路径')

    # 可选参数
    parser.add_argument('--start', "-s", type=float, default=0.0, help='起始时间比例 (0.0 - 1.0)')
    parser.add_argument('--end', "-e", type=float, default=1.0, help='结束时间比例 (0.0 - 1.0)')
    parser.add_argument('--channel', "-c", type=str, default=None, help='指定绘制的单通道名称 (例如 Blue)，不指定则绘制所有')
    parser.add_argument('--save', "-sv", type=str, default=None, help='保存图片的路径 (可选)')
    parser.add_argument('--start_time', type=str, default=None, help='起始时间点，格式 时:分:秒，例如 12:36:26')
    parser.add_argument('--time_interval', type=float, default=None, help='相邻两个数据点的时间间隔（秒）')

    args = parser.parse_args()

    plot(
        csv_path=args.csv_path,
        start_ratio=args.start,
        end_ratio=args.end,
        target_channel=args.channel,
        save_path=args.save,
        start_time=args.start_time,
        time_interval=args.time_interval
    )