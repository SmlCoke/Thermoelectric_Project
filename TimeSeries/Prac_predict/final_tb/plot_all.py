import subprocess
import os  # 新增

if __name__ == "__main__":
    band = ["Blue", "Green", "Infrared", "Red", "Transparent", "Ultraviolet", "Violet", "Yellow"]

    configs = ["raw", "d1"]

    predict_engine_path = "../../src/predict.py"

    model_paths = ["../../Prac_train/final_tb/raw/checkpoints/best_model.pth", "../../Prac_train/final_tb/d1/checkpoints/best_model.pth"]

    csv_paths = ["../../Prac_data/TrainSet/final_tb/tests/raw/data1210_clean.csv", "../../Prac_data/TrainSet/final_tb/tests/d1/data1210_clean.csv"]

    start_time = "12:07:43"

    time_interval = "10"  # 转为字符串，便于传参

    for color in band:
        for i, config in enumerate(configs):
            print(f"开始执行 config = {config}, color = {color} 的推理...")
            model_path = model_paths[i]
            csv_path = csv_paths[i]
            save_path = f"./{config}/cont/data_{color}.npy"
            plot_path = f"./{config}/cont/data_{color}.svg"
            log_path = f"./{config}/cont/data_{color}.log"

            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            cmd = [
                "python", predict_engine_path,
                "--model_path", model_path,
                "--csv_path", csv_path,
                "--save_path", save_path,
                "--custom",
                "--channel", color,
                "--plot_path", plot_path,
                "--start_time", start_time,
                "--time_interval", time_interval,
            ]

            # 将 stdout/stderr 重定向到日志文件
            with open(log_path, "w", encoding="utf-8") as log_f:
                subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, check=True)

            print(f"完成 config = {config}, color = {color} 的推理。\n")