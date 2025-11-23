import pandas as pd


if __name__ =="__main__":
    df = pd.read_excel("data_1122.xlsx")

    # 打印数据，只打印前一部分
    print(df.head())