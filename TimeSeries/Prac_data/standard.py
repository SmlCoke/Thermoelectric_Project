import pandas as pd
import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize a CSV file.")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    # 构建通道序号-波段范围映射表：
    band1129 = {
        "DateTime": "DateTime",
        "TEC1_Optimal(V)": "Yellow",
        "TEC2_Optimal(V)": "Ultraviolet",
        "TEC3_Optimal(V)": "Infrared",
        "TEC4_Optimal(V)": "Red",
        "TEC5_Optimal(V)": "Green",
        "TEC6_Optimal(V)": "Blue",
        "TEC7_Optimal(V)": "Transparent",
        "TEC8_Optimal(V)": "Violet"
    }

    band1210 = {
        "DateTime": "DateTime",
        "TEC1_Optimal(V)": "Blue",
        "TEC2_Optimal(V)": "Green",
        "TEC3_Optimal(V)": "Yellow",
        "TEC4_Optimal(V)": "Violet",
        "TEC5_Optimal(V)": "Red",
        "TEC6_Optimal(V)": "Infrared",
        "TEC7_Optimal(V)": "Ultraviolet",
        "TEC8_Optimal(V)": "Transparent"
    }

    compile = re.compile(r"data(\d+)_standard.csv")
    match = compile.match(args.input_file)
    if not match:
        raise ValueError("Invalid input file format.")
    data_id = int(match.group(1))
    if data_id <= 1129:
        df.rename(columns=band1129, inplace=True)
    else:
        df.rename(columns=band1210, inplace=True)
    
    # 更改后的列名以及数据，统一按照band1129中的顺序排序
    df = df[band1129.values()]

    voltage_columns = list(band1129.values())[1:]  # 获取除DateTime外的所有电压列名

    # df中的每一个数据×1000倍，切换单位
    df[voltage_columns] = df[voltage_columns] * 1000

    df.to_csv(f"data{data_id}.csv", index=False)