## README
本文件夹包含用于展示降噪算法效果的结果图片
我们对`data1220_1`的原始数据集进行了不同方式的降噪处理，并且展示了不同种算法降噪后的数据与原始数据的对比效果
具体包含以下内容：

我们主要针对 `data1220_1` 的紫光 `Violet` 通道以及黄光 `Yellow` 通道展示数据降噪效果，而且主要关注这两个通道的 0~10% 区间内的数据变化情况。
### 文件目录结构
```
Thermoelectric_Project/
├── README.md                          # 本文件，总体说明
├── data/                           
│   ├── outlier/
│   │   ├── readme.md
│   │   ├── data1220_1_n1.4.csv
│   │   ├── data1220_1_n1.8.csv
│   │   ├── data1220_1_n1.75.csv
│   │   ├── data1220_1.csv
│   │   ├── final_test_deno.py
│   │   ├── Violet_outlier.svg
│   │   └── Yellow_outlier.svg
│   ├── smooth/
│   │   ├── readme.md
│   │   ├── data1220_1_L3.csv
│   │   ├── data1220_1_L5.csv
│   │   ├── data1220_1_L7.csv
│   │   ├── data1220_1.csv
│   │   ├── final_test_deno.py
│   │   ├── Violet_smooth.svg  
│   │   └── Yellow_smooth.svg  
│   ├── outlier_smooth/
│   │   ├── readme.md
│   │   ├── data1220_1_denoised.csv
│   │   ├── data1220_1.csv
│   │   ├── final_test_deno.py
│   │   ├── Violet_d1.svg  
│   │   ├── Violet_global_d1.svg
│   │   ├── Yellow_d1.svg  
│   │   └── Yellow_global_d1.svg  
│   └── subsample/
│       ├── readme.md
│       ├── sub6/
│       ├── data1220_1_denoised.csv
│       ├── data1220_1.csv
│       ├── final_test_deno.py
│       └── Violet_d1.svg  
└── DataCollection/                    

```

### 数据降噪的图片展示索引
数据降噪方案的效果展示，包含以下图片需要展示：
1. data1220_1 两个通道 全局图：`./Violet_global.svg`, `./Yellow_global.svg`
2. data1220_1 两个通道 时域区间 0~10% 局部图：`./Violet_raw.svg`, `./Yellow_raw.svg`
3. data1220_1 两个通道 时域区间 0~10% 不同降噪阈值的对比图：`./data/outlier/Violet_outlier.svg`, `./data/outlier/Yellow_outlier.svg`
4. data1220_1 两个通道 时域区间 0~10% 不同滑动窗长的对比图：`./data/smooth/Violet_smooth.svg`, `./data/smooth/Yellow_smooth.svg`
5. data1220_1 两个通道 全局 d1 降噪前后对比图：`./data/outlier_smooth/Violet_global_d1.svg`, `./data/outlier_smooth/Yellow_global_d1.svg`
6. data1220_1 两个通道 时域区间 0~10% d1 降噪前后对比图：`./data/outlier_smooth/Violet_d1.svg`, `./data/outlier_smooth/Yellow_d1.svg`
7. data1220_1 两个通道 全局降采样前后对比图($r=6$)