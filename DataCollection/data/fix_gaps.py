import pandas as pd
import os

def fix_missing_values(file_path):
    print(f"正在处理文件: {file_path}")
    
    if not os.path.exists(file_path):
        print("错误: 文件不存在")
        return

    # 读取CSV
    # 这里的关键是 pandas 会自动将空字段或短行填充为 NaN
    df = pd.read_csv(file_path)
    
    # 打印处理前的缺失情况
    print("\n处理前缺失值统计:")
    print(df.isnull().sum())
    
    # 除去 DateTime 列，其他列都需要插值
    cols_to_interpolate = [col for col in df.columns if col != 'DateTime']
    
    # 确保这些列是数值类型 (防止读取为空字符串时变成 object 类型)
    for col in cols_to_interpolate:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 执行线性插值
    # method='linear': 线性连接前后两个点
    # limit_direction='both': 确保如果开头或结尾有缺失也能补（虽然这里是中间缺失）
    df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method='linear', limit_direction='both')
    
    # 打印处理后的情况
    print("\n处理后缺失值统计:")
    print(df.isnull().sum())
    
    # 预览刚才缺失的那段时间的数据 (13:02:51 - 13:03:26)
    # 简单通过行号切片查看（根据您提供的CSV位置估算）
    # 您提供的缺失段大约在最后几行
    print("\n补齐后的数据预览 (13:02:51 - 13:03:26):")
    print(df.tail(15))

    # 保存回原文件
    df.to_csv(file_path, index=False, encoding='utf-8')
    print(f"\n文件已修复并保存至: {file_path}")

if __name__ == "__main__":
    # 指定您的文件路径
    target_file = r'd:\Courses\Temp\1207\Thermoelectric_Project\DataCollection\data\1214\1214_clean.csv'
    fix_missing_values(target_file)