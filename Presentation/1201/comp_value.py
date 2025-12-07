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

    comps = []
    for i in range(8):
        comps.append(data_1129[voltage_cols[i]]/data_1122[voltage_cols[i]])
    
    # for i in range(8):
    #     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 4))
    #     ax = axes

    #     ax.plot(comps[i], color=index2color[i+1], label=f"{index2band[i+1]} Channel", linewidth=1.5)

    #     ax.set_xlabel("Time (10s interval)", fontproperties=T_16)
    #     ax.set_ylabel("Voltage Ratio (1129/1122)", fontproperties=T_16)
    #     ax.set_title(f"Voltage Ratio Comparison for {index2band[i+1]} Channel", fontproperties=T_16)
    #     ax.legend(prop=T_14)
    #     ax.tick_params(axis='both', which='major', labelsize=12)
    #     ax.grid(True)
    
    #     plt.tight_layout()
    #     plt.savefig(f"{index2band[i+1]}_comp.svg")

    for i in range(8):
        print(f"{index2band[i+1]}:")
        print(f"  Mean Ratio: {np.mean(comps[i]):.4f}")
    print("violet mean ratio:", 1e3*data_1129['TEC8_Optimal(V)'].mean())
    print("Trans mean ratio:", 1e3*data_1129['TEC7_Optimal(V)'].mean())