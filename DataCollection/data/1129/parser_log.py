import re
import pandas as pd

# 输入输出文件名
log_file = "collector1129.log"
csv_file = "collector1129.csv"

# 用来存储解析结果
records = []

# 正则表达式
time_pattern = re.compile(r"\[\d+\]\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
tec_pattern = re.compile(r"TEC(\d+):.*\(([-\s0-9.]+) mV\)")

with open(log_file, "r", encoding="utf-8") as f:
    current_record = {}
    
    for line in f:
        # 检测时间行
        time_match = time_pattern.search(line)
        if time_match:
            # 如果已经记录了上一条，存入列表
            if current_record:
                records.append(current_record)
            current_record = {"DataTime": time_match.group(1)}
            continue
        
        # 检测TEC电压行
        tec_match = tec_pattern.search(line)
        if tec_match:
            tec_index = int(tec_match.group(1))
            mv_value = float(tec_match.group(2))  # mV 数值
            v_value = mv_value / 1000.0           # 转换为 V
            current_record[f"TEC{tec_index}_Optimal(V)"] = v_value

    # 文件结束后再存一次
    if current_record:
        records.append(current_record)

# 转为 DataFrame
df = pd.DataFrame(records)

# 转换时间列为 pandas datetime
df["DataTime"] = pd.to_datetime(df["DataTime"])

# 输出 CSV
df.to_csv(csv_file, index=False)

print("解析完成，已保存为:", csv_file)
