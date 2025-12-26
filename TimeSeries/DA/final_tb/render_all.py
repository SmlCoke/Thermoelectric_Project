import argparse
import sys
from pathlib import Path
from typing import Optional
import importlib.util
import pandas as pd
import matplotlib

# Use non-interactive backend so plt.show() in imported modules does not block
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None  # noqa: E731


def load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_entries_from_outlier(module, folder: Path):
    raw_path, denoise_paths = module.find_files_in_folder(str(folder))
    if not raw_path:
        raise FileNotFoundError(f"未找到原始文件于 {folder}")
    entries = [{"df": pd.read_csv(raw_path), "label": "Original(Raw)", "color": "black"}]
    cmap = plt.cm.get_cmap('tab10')
    for idx, (thr, path) in enumerate(denoise_paths):
        entries.append({
            "df": pd.read_csv(path),
            "label": f"Denoised n={thr}",
            "color": cmap(idx % cmap.N),
        })
    return entries


def build_entries_from_smooth(module, folder: Path):
    raw_path, denoise_paths = module.find_files_in_folder(str(folder))
    if not raw_path:
        raise FileNotFoundError(f"未找到原始文件于 {folder}")
    entries = [{"df": pd.read_csv(raw_path), "label": "Original(Raw)", "color": "black"}]
    cmap = plt.cm.get_cmap('tab10')
    for idx, (L_val, path) in enumerate(denoise_paths):
        entries.append({
            "df": pd.read_csv(path),
            "label": f"Denoised L={L_val}",
            "color": cmap(idx % cmap.N),
        })
    return entries


def build_entries_from_outlier_smooth(module, folder: Path):
    raw_path, denoise_paths = module.find_files_in_folder(str(folder))
    if not raw_path:
        raise FileNotFoundError(f"未找到原始文件于 {folder}")
    entries = [{"df": pd.read_csv(raw_path), "label": "Original(Raw)", "color": "black"}]
    cmap = plt.cm.get_cmap('tab10')
    for idx, (thr, L_val, path) in enumerate(denoise_paths):
        entries.append({
            "df": pd.read_csv(path),
            "label": f"Denoised n={thr} L={L_val}",
            "color": cmap(idx % cmap.N),
        })
    return entries


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def run_tasks(start_time: Optional[str], time_interval: Optional[float], local_end: float):
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"

    plot_base = load_module(base_dir / "plot_base.py", "plot_base_mod")
    mod_outlier = load_module(data_dir / "outlier" / "final_test_deno.py", "outlier_mod")
    mod_smooth = load_module(data_dir / "smooth" / "final_test_deno.py", "smooth_mod")
    mod_outlier_smooth = load_module(data_dir / "outlier_smooth" / "final_test_deno.py", "outlier_smooth_mod")

    raw_csv = data_dir / "outlier" / "data1220_1.csv"
    tasks = [
        # README: 全局原始
        {"runner": "raw", "channel": "Violet", "start": 0.0, "end": 1.0, "save": base_dir / "Violet_global.svg"},
        {"runner": "raw", "channel": "Yellow", "start": 0.0, "end": 1.0, "save": base_dir / "Yellow_global.svg"},
        # README: 0-10% 原始局部
        {"runner": "raw", "channel": "Violet", "start": 0.0, "end": local_end, "save": base_dir / "Violet_raw.svg"},
        {"runner": "raw", "channel": "Yellow", "start": 0.0, "end": local_end, "save": base_dir / "Yellow_raw.svg"},
        # outlier 对比 0-10%
        {"runner": "outlier", "channel": "Violet", "folder": data_dir / "outlier", "start": 0.0, "end": local_end, "save": data_dir / "outlier" / "Violet_outlier.svg"},
        {"runner": "outlier", "channel": "Yellow", "folder": data_dir / "outlier", "start": 0.0, "end": local_end, "save": data_dir / "outlier" / "Yellow_outlier.svg"},
        # smooth 对比 0-10%
        {"runner": "smooth", "channel": "Violet", "folder": data_dir / "smooth", "start": 0.0, "end": local_end, "save": data_dir / "smooth" / "Violet_smooth.svg"},
        {"runner": "smooth", "channel": "Yellow", "folder": data_dir / "smooth", "start": 0.0, "end": local_end, "save": data_dir / "smooth" / "Yellow_smooth.svg"},
        # outlier_smooth 全局
        {"runner": "outlier_smooth", "channel": "Violet", "folder": data_dir / "outlier_smooth", "start": 0.0, "end": 1.0, "save": data_dir / "outlier_smooth" / "Violet_global_d1.svg"},
        {"runner": "outlier_smooth", "channel": "Yellow", "folder": data_dir / "outlier_smooth", "start": 0.0, "end": 1.0, "save": data_dir / "outlier_smooth" / "Yellow_global_d1.svg"},
        # outlier_smooth 0-10%
        {"runner": "outlier_smooth", "channel": "Violet", "folder": data_dir / "outlier_smooth", "start": 0.0, "end": local_end, "save": data_dir / "outlier_smooth" / "Violet_d1.svg"},
        {"runner": "outlier_smooth", "channel": "Yellow", "folder": data_dir / "outlier_smooth", "start": 0.0, "end": local_end, "save": data_dir / "outlier_smooth" / "Yellow_d1.svg"},
        # subsample r=6 全局 (使用与 outlier_smooth 相同的比较逻辑)
        {"runner": "outlier_smooth", "channel": "Violet", "folder": data_dir / "subsample", "start": 0.0, "end": 1.0, "save": data_dir / "subsample" / "Violet_subsample_r6.svg", "sub_factor": 6},
        {"runner": "outlier_smooth", "channel": "Yellow", "folder": data_dir / "subsample", "start": 0.0, "end": 1.0, "save": data_dir / "subsample" / "Yellow_subsample_r6.svg", "sub_factor": 6},
    ]

    failures = []
    for task in tasks:
        try:
            ensure_parent(task["save"])
            if task["runner"] == "raw":
                plot_base.plot(
                    csv_path=str(raw_csv),
                    start_ratio=task["start"],
                    end_ratio=task["end"],
                    target_channel=task["channel"],
                    save_path=str(task["save"]),
                    start_time=start_time,
                    time_interval=time_interval,
                )
            elif task["runner"] == "outlier":
                entries = build_entries_from_outlier(mod_outlier, task["folder"])
                mod_outlier.plot_multiple(
                    data_entries=entries,
                    start_ratio=task["start"],
                    end_ratio=task["end"],
                    target_channel=task["channel"],
                    save_path=str(task["save"]),
                    show_diff=True,
                    start_time=start_time,
                    time_interval=time_interval,
                )
            elif task["runner"] == "smooth":
                entries = build_entries_from_smooth(mod_smooth, task["folder"])
                mod_smooth.plot_multiple(
                    data_entries=entries,
                    start_ratio=task["start"],
                    end_ratio=task["end"],
                    target_channel=task["channel"],
                    save_path=str(task["save"]),
                    show_diff=True,
                    start_time=start_time,
                    time_interval=time_interval,
                )
            elif task["runner"] == "outlier_smooth":
                entries = build_entries_from_outlier_smooth(mod_outlier_smooth, task["folder"])
                eff_interval = None
                if time_interval is not None:
                    eff_interval = time_interval * task.get("sub_factor", 1)
                mod_outlier_smooth.plot_multiple(
                    data_entries=entries,
                    start_ratio=task["start"],
                    end_ratio=task["end"],
                    target_channel=task["channel"],
                    save_path=str(task["save"]),
                    show_diff=True,
                    start_time=start_time,
                    time_interval=eff_interval,
                )
            print(f"✔ 已生成: {task['save']}")
        except Exception as exc:  # noqa: BLE001
            failures.append((task, exc))
            print(f"✖ 失败: {task['save']} => {exc}")

    if failures:
        print("\n以下任务失败，请检查路径或数据:")
        for task, exc in failures:
            print(f"- {task['save']}: {exc}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="一键生成 README 所需的全部 SVG 图")
    parser.add_argument('--start_time', type=str, default=None, help='起始时间点，格式 时:分:秒，例如 12:36:26')
    parser.add_argument('--time_interval', type=float, default=None, help='相邻两个数据点的时间间隔（秒）')
    parser.add_argument('--local_end', type=float, default=0.1, help='局部视图的结束比例，默认 0.1 (0-10%)')
    args = parser.parse_args()

    run_tasks(start_time=args.start_time, time_interval=args.time_interval, local_end=args.local_end)


if __name__ == "__main__":
    main()
