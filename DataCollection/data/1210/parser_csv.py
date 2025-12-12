import pandas as pd
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    cols_to_drop = df.filter(regex=r'.*[G,g]ain*').columns
    df = df.drop(columns=cols_to_drop)
    cols_to_drop = df.filter(regex=r'Timestamp').columns
    df = df.drop(columns=cols_to_drop)
    # df["TEC1_Optimal(V)"] = -df["TEC1_Optimal(V)"]
    # df["TEC2_Optimal(V)"] = -df["TEC2_Optimal(V)"]
    df.to_csv(f"data{args.csv_file}", index=False)