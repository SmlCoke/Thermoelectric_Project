import pandas as pd
import argparse
import os
import sys
import numpy as np

def process_data(input_csv_file):
    # 定义列名映射关系
    band = {
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

    # 检查文件是否存在
    if not os.path.exists(input_csv_file):
        print(f"错误: 文件 '{input_csv_file}' 不存在。")
        return

    try:
        # 读取CSV文件
        df = pd.read_csv(input_csv_file)
        
        # (1) 保留指定列并重命名
        # 检查所需的列是否都在CSV中
        missing_cols = [col for col in band.keys() if col not in df.columns]
        if missing_cols:
            print(f"警告: 输入文件中缺少以下列: {missing_cols}")
            # 只保留存在的列
            existing_keys = [col for col in band.keys() if col in df.columns]
            df_clean = df[existing_keys].copy()
        else:
            df_clean = df[list(band.keys())].copy()
            
        # 重命名列
        df_clean.rename(columns=band, inplace=True)

        # (2) 修改DateTime格式为 时:分:秒
        if 'DateTime' in df_clean.columns:
            print("请确保DateTime列已经通过Excel软件处理完毕！")

        # (3) 将Blue列和Green列的数值取相反数
        for col in ['Blue', 'Green']:
            if col in df_clean.columns:
                df_clean[col] = -df_clean[col]

        # (4) 切换单位（除DateTime列），数值扩大1000倍
        df_clean.loc[:, df_clean.columns != 'DateTime'] *= 1000

        # (5) 列的顺序统一按照band1129的顺序排序
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
        # 确保只选择存在的列
        available_cols = [col for col in band1129.values() if col in df_clean.columns]
        df_clean = df_clean[available_cols]
        
        # (新增) 检查并补齐缺失值
        print("\n正在检查缺失值...")
        if df_clean.isnull().values.any():
            # 找到缺失值的位置
            rows, cols = np.where(df_clean.isnull())
            
            print(f"发现 {len(rows)} 个缺失值，正在处理...")
            
            for r, c in zip(rows, cols):
                col_name = df_clean.columns[c]
                print(f"  - 缺失位置: 第 {r+1} 行, 列 '{col_name}'")
            
            # 使用线性插值填充缺失值 (相当于前后平均)
            # limit_direction='both' 确保首尾缺失也能被处理（虽然首尾无法取前后平均，通常取最近值）
            # 对于数值列进行插值
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear', limit_direction='both')
            
            print("缺失值已通过线性插值（前后平均）补齐。")
        else:
            print("未发现缺失值。")

        # (6) 保存文件
        # 获取原文件路径信息
        dir_name = os.path.dirname(input_csv_file)
        base_name = os.path.basename(input_csv_file)
        file_name_no_ext = os.path.splitext(base_name)[0]
        
        # 构建新文件名
        new_filename = f"{file_name_no_ext}_clean.csv"
        output_path = os.path.join(dir_name, new_filename)
        
        # 保存为CSV，不包含索引
        df_clean.to_csv(output_path, index=False, encoding='utf-8')
        print(f"处理成功！文件已保存至: {output_path}")
        
        # 打印前几行预览
        print("\n数据预览:")
        print(df_clean.head())

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process thermoelectric data CSV.")
    parser.add_argument("input_csv_file", type=str, help="Path to the input CSV file")
    
    args = parser.parse_args()
    process_data(args.input_csv_file)