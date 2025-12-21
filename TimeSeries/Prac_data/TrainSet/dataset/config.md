## 数据预处理说明

### 1. 关于降采样因子的说明
模型步长：输入60 输出10
如果间隔为10s，则实现的最终效果为10min预测100s
如果间隔扩大6倍，即为60s，则实现的最终效果为60min预测10min
但是这样的话数据集长度就会缩减为原来的1/6，一组至少要70个点，那么原始数据集一组最少需要420个点

### 2. `dataset`————真正用来训练的数据集
#### 2.1 sub6:
未进行`denoise`，仅进行$r=6$的降采样

#### 2.2 sub8
未进行`denoise`，仅进行$r=8$的降采样

#### 2.3 d1_sub6:
先进行`denoise1`，然后进行$r=6$的降采样

#### 2.4 d1_sub8:
先进行`denoise1`，然后进行$r=8$的降采样

#### 2.5 d2_sub6:
先进行`denoise2`，然后进行$r=6$的降采样

#### 2.6 d2_sub8:
先进行`denoise2`，然后进行$r=8$的降采样

#### 2.7 sub6_d3
先进行$r=6$的降采样，然后进行`--outlier_window 5 --outlier_threshold 1.75 --smooth_window 3`

#### 2.8 sub8_d4
先进行$r=8$的降采样，然后进行`--outlier_window 5 --outlier_threshold 1.75 --smooth_window 3`

### 3. `preprocess`————中间数据集，一般只经过了denoise
#### 3.1 denoise1
**denoise配置**
outlier_threshold = 1.75
outlier_window = 5
smooth_window = 5

**终端输出**
```bash
批量处理模式
输入目录: D:\Courses\Thermoelectric_Project\TimeSeries\Prac_data\raw
文件模式: *.csv

找到 9 个CSV文件
降噪方法: both
异常值检测窗口: 5, 阈值: 1.75
平滑窗口: 5
输出目录: D:\Courses\Thermoelectric_Project\TimeSeries\Prac_data\TrainSet\temp\denoise1
================================================================================

处理文件: data1122.csv
  ✓ 样本数: 1650
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 134个

处理文件: data1129.csv
  ✓ 样本数: 91
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 7个

处理文件: data1210_clean.csv
  ✓ 样本数: 1322
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 164个

处理文件: data1214_1.csv
  ✓ 样本数: 588
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 21个

处理文件: data1214_2.csv
  ✓ 样本数: 587
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 17个

处理文件: data1217_1.csv
  ✓ 样本数: 660
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 73个

处理文件: data1217_2.csv
  ✓ 样本数: 660
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 76个

处理文件: data1220_1.csv
  ✓ 样本数: 1160
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 143个

处理文件: data1220_2.csv
  ✓ 样本数: 1159
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 141个

================================================================================
处理完成！成功处理 9/9 个文件
共修正 776 个异常值
输出目录: D:\Courses\Thermoelectric_Project\TimeSeries\Prac_data\TrainSet\temp\denoise1

全部完成！
```

异常钳位：平均修复了9.1%个数据，最高12.4%，最低2.8%


#### 3.2 denoise1
**denoise配置**
outlier_threshold = 1.5
outlier_window = 5
smooth_window = 5


**终端输出**
```bash
批量处理模式
输入目录: D:\Courses\Thermoelectric_Project\TimeSeries\Prac_data\raw
文件模式: *.csv

找到 9 个CSV文件
降噪方法: both
异常值检测窗口: 5, 阈值: 1.5
平滑窗口: 5
输出目录: D:\Courses\Thermoelectric_Project\TimeSeries\Prac_data\TrainSet\preprocess\denoise2
================================================================================

处理文件: data1122.csv
  ✓ 样本数: 1650
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 440个

处理文件: data1129.csv
  ✓ 样本数: 91
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 42个

处理文件: data1210_clean.csv
  ✓ 样本数: 1322
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 667个

处理文件: data1214_1.csv
  ✓ 样本数: 588
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 240个

处理文件: data1214_2.csv
  ✓ 样本数: 587
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 286个

处理文件: data1217_1.csv
  ✓ 样本数: 660
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 344个

处理文件: data1217_2.csv
  ✓ 样本数: 660
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 365个

处理文件: data1220_1.csv
  ✓ 样本数: 1160
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 587个

处理文件: data1220_2.csv
  ✓ 样本数: 1159
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 551个

================================================================================
处理完成！成功处理 9/9 个文件
共修正 3522 个异常值
输出目录: D:\Courses\Thermoelectric_Project\TimeSeries\Prac_data\TrainSet\preprocess\denoise2

全部完成！
```

异常钳位：平均修复了46.4%个数据，最高55.3%，最低26.6%