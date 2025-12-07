import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties

# 定义字体
S_16 = FontProperties(fname = r'C:\\Windows\\Fonts\\simsun.ttc', size = 16)  # 宋体标题
S_14 = FontProperties(fname = r'C:\\Windows\\Fonts\\simsun.ttc', size = 14)  # 宋体坐标轴
S_12 = FontProperties(fname = r'C:\\Windows\\Fonts\\simsun.ttc', size = 12)  # 宋体刻度 (原楷体)
S_9  = FontProperties(fname = r'C:\\Windows\\Fonts\\simsun.ttc', size = 9)    # 宋体小字(用于柱状图内文字)

# Task definitions 
# 逻辑修改：允许输入不定长的元组，代码将自动提取首尾作为起止时间
raw_tasks = [
    ("方案初始化", 7, 8),
    ("设备购买", 8, 9),
    ("自动化采集脚本的开发与验证", 8, 9, 10), 
    ("芯片采集阵列的搭建与调试", 9, 10),
    ("天空辐射采集", 10, 11, 13),
    ("辐射数据处理", 13, 14, 15),
    ("时间序列预测", 14, 15), 
]

# 数据清洗：只取任务名、开始时间(第2个元素)、结束时间(最后一个元素)
cleaned_tasks = [(t[0], t[1], t[-1]) for t in raw_tasks]

df = pd.DataFrame(cleaned_tasks, columns=["Task", "Start", "End"])
df["Duration"] = df["End"] - df["Start"] + 1  # inclusive weeks

fig, ax = plt.subplots(figsize=(12, 6))
y_positions = range(len(df))

for i, (idx, row) in enumerate(df.iterrows()):
    ax.barh(i, row["Duration"], left=row["Start"], height=0.6)
    ax.text(row["Start"] - 0.25, i, f"W{int(row['Start'])}", va="center", ha="right", fontsize=9)
    ax.text(row["Start"] + row["Duration"] + 0.2, i, f"W{int(row['End'])}", va="center", ha="left", fontsize=9)

ax.set_yticks(list(y_positions))
# 修复：应用中文字体 (宋体)
ax.set_yticklabels(df["Task"], fontproperties=S_12) 
ax.invert_yaxis()
ax.set_xlim(7, 16)

# 修复：应用字体
ax.set_xlabel("Week number", fontproperties=S_14)
ax.set_title("项目甘特图（按周）", fontproperties=S_16)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)

for i, row in df.iterrows():
    xpos = row["Start"] + row["Duration"] / 2
    # 修复：应用中文字体 (宋体小字)，因为包含汉字"周"
    ax.text(xpos, i, f"{int(row['Duration'])} 周", va="center", ha="center", color="white", fontweight="bold", fontproperties=S_9)

plt.tight_layout()
plt.savefig("gantt.svg", dpi=200)
