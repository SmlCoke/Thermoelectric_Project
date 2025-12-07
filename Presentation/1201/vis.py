import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.dates import DateFormatter

T_16 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 16)    # Times New Roman
T_14 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 14)    # Times New Roman
T_12 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 12)    # Times New Roman
if __name__ == "__main__":
    # 读取数据
    df1 = pd.read_excel("data1122.xlsx")
    df2 = pd.read_excel("data1129.xlsx")
    
    # 定义电压列名
    voltage_cols = [f"TEC{i}_Optimal(V)" for i in range(1, 9)]
    
    # (1) 将每一列电压数据转化为numpy数组
    # 使用字典存储，方便后续调用
    data_1122 = {col: df1[col].to_numpy() for col in voltage_cols}
    data_1129 = {col: df2[col].to_numpy() for col in voltage_cols}
    
    # 获取时间轴 (手动生成：起始15:56:00，间隔10s)
    # 原表格时间损坏，故根据已知信息重新生成时间序列
    # 日期部分不影响时:分显示，暂定为任意日期
    start_time = "2025-11-22 15:56:00"
    x_axis = pd.date_range(start=start_time, periods=len(df1), freq='10s')


    # 构建通道序号-波段范围映射表：
    index2band = {
        1: "Yellow",
        2: "Ultraviolet",
        3: "Infrared",
        4: "Red",
        5: "Green",
        6: "Blue",
        7: "Trans",
        8: "Violet"
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

    # (2) 绘图：八个通道电压的比较
    

    for i, col in enumerate(voltage_cols):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 4))
        ax = axes

        # 绘制 1122 数据
        ax.plot(x_axis, 1e3*data_1122[col], label='1122', marker='o', markersize=4, linestyle='-', alpha=0.7)
        
        # 绘制 1129 数据
        ax.plot(x_axis, 1e3*data_1129[col], label='1129', marker='x', markersize=4, linestyle='--', alpha=0.7)
        
        ax.set_title(f"{index2band[i+1]} Channel Voltage Comparison", fontproperties=T_16)
        ax.set_xlabel("Time", fontproperties=T_14)
        ax.set_ylabel("Voltage (mV)", fontproperties=T_14)
        
        # 设置X轴时间格式为 "时:分"
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        # 设置刻度字体为 T_12
        for label in ax.get_xticklabels():
            label.set_fontproperties(T_12)
        for label in ax.get_yticklabels():
            label.set_fontproperties(T_12)

        ax.legend(prop = T_14)
        ax.grid(True)
    
        plt.tight_layout()
        plt.savefig(f"{index2band[i+1]}.svg")