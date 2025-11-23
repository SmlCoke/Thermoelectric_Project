import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime, timedelta
from matplotlib.font_manager import FontProperties

T_16 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 16)    # Times New Roman
T_14 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 14)    # Times New Roman
T_12 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 12)    # Times New Roman

def add_minutes_to_time(time_str, minutes_to_add):
    # 将字符串如 "12:10" 转为 datetime 对象（日期默认为 1900-01-01）
    base_time = datetime.strptime(time_str, "%H:%M")
    # 增加指定分钟数
    new_time = base_time + timedelta(minutes=minutes_to_add)
    # 返回格式化后的时间字符串 "ab:cd"
    return new_time.strftime("%H:%M")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument("N", type=int, default=10, help="Time interval in minutes")
    parser.add_argument("channel", type=int, default=1, help="Channel to analyze")
    parser.add_argument("--plot_mode", type=str, default="plot", choices=["plot", "stem"], help="Plotting mode")
    args = parser.parse_args()

    N = args.N
    channel = f"TEC{args.channel}_Optimal(V)"
    plot_mode = args.plot_mode

    # 读取数据
    df = pd.read_excel("data_clean.xlsx")


    # 构建通道序号-波段范围映射表：
    index2band = {
        1: "yellow",
        2: "Ultraviolet",
        3: "Infrared",
        4: "red",
        5: "green",
        6: "blue",
        7: "Trans",
        8: "violet"
    }

    index2color = {
        1: "gold",          # yellow（比 yellow 更显眼、对比度更佳）
        2: "purple",        # Ultraviolet（UV不可见 → 紫色最接近）
        3: "brown",         # Infrared（红外不可见 → 用深红/棕色表达热感）
        4: "red",           # red
        5: "green",         # green
        6: "blue",          # blue
        7: "gray",          # Trans（透明 → 用 gray 表示中性/无色）
        8: "violet",        # violet（可见紫光）
    }

    # 将 DateTime 转为 pandas datetime （如果不是的话）
    df["DateTime"] = pd.to_datetime(df["DateTime"])

    # 设为索引，便于 resample
    df = df.set_index("DateTime")

    # 按 N 分钟分段计算平均值
    # 例如：N=5 → '5min'
    resampled = df[channel].resample(f"{N}min").mean()

    # 若有尾部不足 N 分钟的段，resample 自动会处理并给出平均值（如果有数据）
    # 去除可能出现的 NaN
    resampled = resampled.dropna()

    # -------------------------------
    # 绘图
    # -------------------------------
    plt.figure(figsize=(10, 5))

    x = np.arange(len(resampled)) + 1  # 段编号 1, 2, 3, ...

    x_time_start = [add_minutes_to_time("12:10", int(N * (i - 1))) for i in x]
    x_time_end = [add_minutes_to_time("12:10", int(N * i)) for i in x]
    x_time = [f"{start} - {end}" for start, end in zip(x_time_start, x_time_end)]
    if plot_mode == "plot":
        plt.plot(x, resampled.values, marker='o', color = index2color[args.channel])
    elif plot_mode == "stem":
        markerline, stemlines, baseline = plt.stem(x, resampled.values,linefmt='k-', markerfmt='ko', basefmt='k-')
        plt.setp(markerline, markersize=6)
        plt.setp(stemlines, linewidth=1)

    # 将刻度设置为时间段，并将字体修改为Times New Roman
    plt.xticks(x, x_time, rotation=45, fontproperties = T_12)
    plt.title(f"TEC{args.channel}: {str(index2band[args.channel])}", fontproperties=T_16)
    plt.xlabel("Time", fontproperties=T_14)
    plt.ylabel("Average Voltage (V)", fontproperties=T_14)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"TEC_{args.channel}_{args.plot_mode}.pdf")
