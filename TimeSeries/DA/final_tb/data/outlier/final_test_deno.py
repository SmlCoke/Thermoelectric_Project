import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
import re
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

def find_files_in_folder(folder_path):
    """在文件夹中查找原始文件与降噪文件列表"""
    if not os.path.isdir(folder_path):
        print("错误：提供的路径不是文件夹。")
        return None, []

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    raw_pattern = re.compile(r"data\d+_\d?\.csv$")
    denoise_pattern = re.compile(r"data\d+_\d?_n(\d+(?:\.\d+)?)\.csv$")

    raw_files = [f for f in files if raw_pattern.match(f)]
    denoise_files = []
    for f in files:
        match = denoise_pattern.match(f)
        if match:
            threshold = float(match.group(1))
            denoise_files.append((threshold, f))

    raw_path = None
    if raw_files:
        raw_files.sort()
        raw_path = os.path.join(folder_path, raw_files[0])
        if len(raw_files) > 1:
            print(f"警告：发现多个原始文件，将使用 {raw_files[0]}")

    denoise_files.sort(key=lambda x: x[0])  # 按阈值排序，便于对比
    denoise_paths = [(thr, os.path.join(folder_path, f)) for thr, f in denoise_files]
    return raw_path, denoise_paths

def plot_channel_with_multiple(ax, x_axis, slices, channel, show_legend=True, x_is_time=False):
    """在指定 ax 上绘制原始与多组降噪数据"""
    has_data = False
    for slice_item in slices:
        if channel in slice_item["data"].columns:
            line_color = slice_item["color"]
            if "Original" in slice_item["label"]:
                line_color = get_channel_color(channel)
            ax.plot(
                x_axis,
                slice_item["data"][channel],
                color=line_color,
                alpha=0.9,
                linewidth=1,
                label=slice_item["label"],
            )
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5, f'Column {channel} not found', ha='center', va='center', fontproperties=T_12)

    ax.set_title(f'{channel} Channel', fontproperties=T_14)
    ax.set_ylabel('mV', fontproperties=T_12)
    ax.grid(True, alpha=0.3)
    format_time_axis(ax, x_is_time)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(T_12)
    if show_legend:
        ax.legend(loc='upper right', prop=T_12)

def plot_multiple(data_entries, start_ratio=0.0, end_ratio=1.0, target_channel=None, save_path=None, show_diff=False,
                 start_time=None, time_interval=None):
    """绘制原始与多组降噪数据的对比图"""
    if not data_entries:
        print("错误：未提供任何数据。")
        return

    # 2. 确定切片范围，使用所有数据集长度的最小值确保对齐
    min_len = min(len(entry["df"]) for entry in data_entries)
    if min_len == 0:
        print("错误：数据为空，无法绘图。")
        return

    time_axis_full = build_time_axis(min_len, start_time, time_interval)

    start_idx = int(min_len * start_ratio)
    end_idx = int(min_len * end_ratio)

    if start_idx < 0:
        start_idx = 0
    if end_idx > min_len:
        end_idx = min_len
    if start_idx >= end_idx:
        print("错误：起始点必须小于结束点。")
        return

    print(f"正在绘图... 显示范围: {start_ratio*100}% - {end_ratio*100}% (索引: {start_idx} - {end_idx})")

    slices = []
    for entry in data_entries:
        sliced_df = entry["df"].iloc[start_idx:end_idx].reset_index(drop=True)
        slices.append({
            "label": entry["label"],
            "color": entry["color"],
            "data": sliced_df,
        })

    x_axis = time_axis_full[start_idx:end_idx] if time_axis_full else list(range(start_idx, end_idx))
    x_is_time = time_axis_full is not None

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

    if target_channel:
        if target_channel not in slices[0]["data"].columns:
            print(f"错误：在数据中找不到通道 '{target_channel}'")
            print(f"可用通道: {list(slices[0]['data'].columns)}")
            return
        # ---- 多个降噪文件：每个降噪一列（主图+残差） ----
        if len(slices) > 1:
            denoise_slices = slices[1:]  # 跳过原始
            n = len(denoise_slices)
            fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 6),
                                     sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            if n == 1:
                axes = axes.reshape(2, 1)
            raw_series = slices[0]["data"][target_channel]
            for idx, slice_item in enumerate(denoise_slices):
                ax_main = axes[0, idx]
                ax_diff = axes[1, idx]
                # 主图：原始 + 当前降噪
                ax_main.plot(x_axis, raw_series, color=get_channel_color(target_channel), linewidth=1, label='Original')
                ax_main.plot(x_axis, slice_item["data"][target_channel],
                             color=slice_item["color"], linewidth=1, label=slice_item["label"])
                ax_main.set_title(f'{target_channel} ({slice_item["label"]})', fontproperties=T_14)
                ax_main.set_ylabel('mV', fontproperties=T_12)
                ax_main.grid(True, alpha=0.3)
                format_time_axis(ax_main, x_is_time)
                if idx == 0:
                    ax_main.legend(prop=T_12)
                for lbl in ax_main.get_xticklabels() + ax_main.get_yticklabels():
                    lbl.set_fontproperties(T_12)
                # 残差：降噪 - 原始
                ax_diff.plot(x_axis,
                             slice_item["data"][target_channel] - raw_series,
                             color=slice_item["color"], linewidth=1, label='Diff')
                ax_diff.axhline(0, color='gray', linewidth=1, alpha=0.6)
                ax_diff.set_ylabel('mV', fontproperties=T_12)
                ax_diff.grid(True, alpha=0.3)
                format_time_axis(ax_diff, x_is_time)
                if idx == 0:
                    ax_diff.legend(prop=T_12)
                for lbl in ax_diff.get_xticklabels() + ax_diff.get_yticklabels():
                    lbl.set_fontproperties(T_12)
            title_text = f'Radiation Variation - {target_channel} ({start_ratio}-{end_ratio})'
            plt.suptitle(title_text, fontproperties=T_16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        else:
            # 只有原始数据时
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_channel_with_multiple(ax, x_axis, slices, target_channel, show_legend=True, x_is_time=x_is_time)
            title_text = f'Radiation Variation - {target_channel} ({start_ratio}-{end_ratio})'
            plt.suptitle(title_text, fontproperties=T_16)
            plt.tight_layout()
    else:
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        axes = axes.flatten()
        for i, channel in enumerate(channels):
            plot_channel_with_multiple(axes[i], x_axis, slices, channel, show_legend=(i==0), x_is_time=x_is_time)
        title_text = f'Radiation Variation ({start_ratio}-{end_ratio} range)'

        plt.suptitle(title_text, fontproperties=T_16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图片已保存至: {save_path}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化降噪效果对比脚本")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv_path', "-p", type=str, help='单个CSV文件路径 (仅绘制该文件)')
    group.add_argument('--folder', "-f", type=str, help='包含原始与降噪文件的文件夹路径')

    parser.add_argument('--start', "-s", type=float, default=0.0, help='起始时间比例 (0.0 - 1.0)')
    parser.add_argument('--end', "-e", type=float, default=1.0, help='结束时间比例 (0.0 - 1.0)')
    parser.add_argument('--channel', "-c", type=str, default=None, help='指定绘制的单通道名称 (例如 Blue)，不指定则绘制所有')
    parser.add_argument('--save', "-sv", type=str, default=None, help='保存图片的路径 (可选)')
    parser.add_argument('--diff', action='store_true', help='单通道模式下同时绘制残差子图')
    parser.add_argument('--start_time', type=str, default=None, help='起始时间点，格式 时:分:秒，例如 12:36:26')
    parser.add_argument('--time_interval', type=float, default=None, help='相邻两个数据点的时间间隔（秒）')

    args = parser.parse_args()

    if args.folder:
        raw_path, denoise_paths = find_files_in_folder(args.folder)
        if not raw_path:
            print("错误：未找到原始文件 (data***.csv)。")
            sys.exit(1)
        data_entries = []

        # 原始数据使用黑色
        raw_df = pd.read_csv(raw_path)
        data_entries.append({"df": raw_df, "label": "Original(Raw)", "color": "black"})

        # 生成颜色序列用于多条降噪曲线
        cmap = plt.cm.get_cmap('tab10')
        for idx, (thr, path) in enumerate(denoise_paths):
            df = pd.read_csv(path)
            color = cmap(idx % cmap.N)
            data_entries.append({
                "df": df,
                "label": f"Denoised n={thr}",
                "color": color,
            })

        if len(denoise_paths) == 0:
            print("警告：未找到降噪文件，仅绘制原始数据。")

        plot_multiple(
            data_entries=data_entries,
            start_ratio=args.start,
            end_ratio=args.end,
            target_channel=args.channel,
            save_path=args.save,
            show_diff=args.diff,
            start_time=args.start_time,
            time_interval=args.time_interval
        )
    else:
        if not os.path.exists(args.csv_path):
            print("错误：找不到输入文件。")
            sys.exit(1)
        single_df = pd.read_csv(args.csv_path)
        plot_multiple(
            data_entries=[{"df": single_df, "label": "Original(Raw)", "color": "black"}],
            start_ratio=args.start,
            end_ratio=args.end,
            target_channel=args.channel,
            save_path=args.save,
            show_diff=args.diff,
            start_time=args.start_time,
            time_interval=args.time_interval
        )