import numpy as np

def main(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    print(f"文件: {npy_path}")
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            # 打印前最多 10 个元素（展开为一维查看）
            flat = v.reshape(-1)
            preview = flat
            print(f"  preview: {preview}")
        else:
            print(f"{k}: type={type(v)}, value={v}")

if __name__ == "__main__":
    # 替换为你的 npy 路径
    npy_path = r"d1/cont/Red.npy"
    main(npy_path)