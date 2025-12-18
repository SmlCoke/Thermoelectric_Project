import pandas as pd
import argparse
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process thermoelectric data CSV.")
    parser.add_argument("input_csv_file", type=str, help="Path to the input CSV file")
    
    args = parser.parse_args()
    # 读取CSV文件
    df = pd.read_csv(args.input_csv_file)

    # (新增) 检查并补齐缺失值
    print("\n正在检查缺失值...")
    if df.isnull().values.any():
        # 找到缺失值的位置
        rows, cols = np.where(df.isnull())
        
        print(f"发现 {len(rows)} 个缺失值，正在处理...")
        
        for r, c in zip(rows, cols):
            col_name = df.columns[c]
            print(f"  - 缺失位置: 第 {r+1} 行, 列 '{col_name}'")
        
        # 使用线性插值填充缺失值 (相当于前后平均)
        # limit_direction='both' 确保首尾缺失也能被处理（虽然首尾无法取前后平均，通常取最近值）
        # 对于数值列进行插值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

        print("缺失值已通过线性插值（前后平均）补齐。")
    else:
        print("未发现缺失值。")