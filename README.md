# Thermoelectric_Project

该仓库用于存储《芯片发电技术基础与应用》的课程材料，包括项目方案书、数据采集系统和神经网络代码。

## 一. 项目概述
本项目旨在搭建一个基于Raspberry Pi的TEC（Thermoelectric Cooler）热电芯片数据采集与分析系统。系统使用Raspberry Pi 3B作为主控，通过I2C总线连接多个ADS1115高精度ADC模块，实现多路热电电压的自动化采集1、存储和分析。后端集成了实时采集-推理-可视化的GUI系统，并基于深度学习模型对采集数据进行预测与分析。

### 核心技术
- **硬件平台**: Raspberry Pi 3B
- **传感器**: TEC1-04906热电芯片
- **ADC模块**: ADS1115（16位分辨率，I2C接口）
- **数据采集**: 10秒/次，支持7天连续采集
- **数据处理**: 基于深度学习的电压预测与分析
- **模型训练**: LSTM/GRU模型
- **实时系统**: 采集-推理-可视化闭环，通过PyQt实现GUI展示


### 目录结构

```
Thermoelectric_Project/
├── README.md                               # 本文件，项目总体说明
├── Initial/                                # 项目方案书
│   └── 完整方案书.md                        # 项目设计文档
├── Code_1/                                 # 时间序列预测模型（模拟）
│   ├── full_version/                       # 完整版深度学习模型
│   ├── lightweight_version/                # 轻量级模型
│   └── lightweight_copy/                   # 轻量级模型副本
├── DataCollectCode/                        # 数据采集系统（核心目录）
│   ├── docs/                               # 采集系统文档
│   │   ├── readme.md                       # 数据采集系统详细说明
│   │   ├── 快速开始指南.md                  # 新手快速上手指南
│   │   ├── requirements.txt                # Python 依赖库列表
│   │   ├── LowPower.md                     # 低功耗优化配置（新增）
│   │   ├── automation.md                   # 自动化采集方案（新增）
│   │   ├── DataPersistence.md              # 数据持久化说明（新增）
│   │   ├── 单ADS1115连线方案.md             # 单个 ADS1115 连线方案
│   │   ├── 完整4ADS1115连线方案.md          # 完整4个 ADS1115 连线方案
│   │   └── 树莓派远程连接和基本操作指南.pdf   # 利用 SSH 远程连接 Windows11 与 Pi 的说明
│   ├── single_collector.py                 # 单 ADS1115 采集脚本
│   ├── Full_collector.py                   # 完整4 ADS1115 采集脚本
│   ├── mock_collector.py                   # 模拟数据采集脚本（主要用于 GUI 实时系统搭建的尝试）
│   ├── pi_sender.py                        # 数据转发脚本（将数据发送到主机，工作于 GUI 系统）
│   └── tec-collector.txt                   # Raspberry systemd 服务配置示例
├── RealTimeSystem/                         # 实时采集-推理-可视化系统
│   ├── docs/                               # 实时系统文档
│   │   ├── README.md                       # 实时系统说明
│   │   ├── init.md                         # 实时系统构建的 prompt
│   │   ├── FIX_NOTES.md                    # 实时系统已知问题与修复说明
│   │   ├── TESTING_GUIDE.md                # 实时系统测试指南
│   │   └── ...                             # RealTimeSystem/中的 python 脚本的说明文档
│   ├── Raspberry_Pi/                       # Raspberry Pi 端脚本
│   │   ├── pi_sender.py                    # Pi 端数据发送脚本
│   │   ├── mock_collector.py               # Pi 端模拟采集脚本
│   │   └── ...                             # 其他 Pi 端脚本
│   ├── realtime_data/                      # 实时数据存储目录（重要，GUI显示的中介）
│   ├── server.py                           # 主机端数据接收服务器
│   ├── gui_app.py                          # 主机端 GUI 应用
│   └── inference_engine.py                 # 主机端推理引擎
├── TimeSeries/                             # 时间序列预测模型（核心目录）
│   ├── docs/                               # 时间序列模型文档
│   ├── DA/                                 # 数据处理与增强模块
│   ├── Sim_data/                           # 模拟数据
│   ├── src/                                # 模型训练与推理代码
│   ├── requirements.txt                    # Python 依赖库列表
│   ├── README.md                           # 时间序列模型说明
│   └── IMPLEMENTATION_SUMMARY.md           # 模型实现细节总结
└── FigureProcess/                          # 天空图像处理脚本
    └── FPU.py                              # 天空图像处理主脚本


```

---

## 二. 快速开始（Rasp控制环境）

### 2.1 前置条件

1. **硬件准备**:
   - Raspberry Pi 3B（及以上版本，已烧录Raspberry Pi OS）
   - 4个ADS1115 ADC模块
   - 8个TEC1-04906热电芯片
   - 面包板、杜邦线等连接材料

2. **软件准备**:
   - Raspberry Pi end:
     - Python 3.7+
     - 已启用I2C接口
   - Host end:
     - Python 3.7+
     - PyQt5等依赖库（见 `RealTimeSystem/docs/README.md`）
     - 完整的 PyTorch 环境（见 `TimeSeries/requirements.txt`）
     - 本仓库完整代码 

### 2.2 安装步骤

#### 1. 克隆仓库

```bash
cd /home/pi/dev
git clone https://github.com/SmlCoke/Thermoelectric_Project.git
```

然后将 `Thermoelectric_Project/DataCollectCode` 下的所有代码复制到 `/home/pi/dev/ads1115_project/Themoelectric` 目录下：

#### 2. 创建Python虚拟环境（推荐）

```bash
python3 -m venv /home/pi/dev/ads1115_project/py311
source /home/pi/dev/ads1115_project/py311/bin/activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：
```bash
pip install adafruit-circuitpython-ads1x15
pip install adafruit-blinka
```

#### 4. 启用I2C接口

```bash
sudo raspi-config
# 选择: 3 Interface Options → I5 I2C → Yes
sudo reboot
```

#### 5. 硬件连线

- 单ADS1115方案：参见 [`DataCollectCode/docs/单ADS1115连线方案.md`](DataCollectCode/docs/单ADS1115连线方案.md)
- 完整方案：参见 [`DataCollectCode/docs/完整4ADS1115连线方案.md`](DataCollectCode/docs/完整4ADS1115连线方案.md)

#### 6. 测试运行

```bash
cd /home/pi/dev/ads1115_project/Themoelectric
python3 Full_collector.py
```

查看数据文件：
```bash
tail -f data.csv
```

---

## 三. 核心功能

本项目主要包含四大核心功能模块，涵盖了从数据采集、实时监测、深度学习预测到图像分析的完整流程。

### 1. 数据采集系统 (Data Collection)

**核心目录**: [`DataCollectCode/`](DataCollectCode/)

该模块是项目的基础，负责在 Raspberry Pi 上稳定、精确地采集热电芯片电压数据。

*   **多通道支持**: 支持 1-4 个 ADS1115 模块级联，实现 2-8 路电压同步采集。
*   **高可靠性**: 采用行缓冲与 `fsync` 机制，确保断电数据不丢失（详见 [DataPersistence.md](DataCollectCode/docs/DataPersistence.md)）。
*   **自动化运行**: 集成 systemd 服务配置，实现开机自启、故障自动重启（详见 [automation.md](DataCollectCode/docs/automation.md)）。
*   **低功耗优化**: 针对 Raspberry Pi 3B 进行深度优化，关闭非必要外设，适合长期野外部署（详见 [LowPower.md](DataCollectCode/docs/LowPower.md)）。

### 2. 实时监测与预测系统 (Real-Time System)

**核心目录**: [`RealTimeSystem/`](RealTimeSystem/)

基于 "边缘-主机" 架构的实时系统，实现数据的实时上传、推理与可视化。

*   **架构设计**:
    *   **边缘端 (Pi)**: 运行 `pi_sender.py`，实时读取采集数据并通过 HTTP POST 发送至主机。
    *   **主机端 (PC)**: 运行 `server.py` 接收数据，`inference_engine.py` 进行实时推理，`gui_app.py` 展示图表。
*   **实时预测**: 集成训练好的 LSTM/GRU 模型，对未来电压趋势进行实时预测。
*   **可视化界面**: 基于 PyQt5 构建的 GUI，动态展示各通道电压波形与预测结果。

👉 **详细文档**: [RealTimeSystem/docs/README.md](RealTimeSystem/docs/README.md)

### 3. 时间序列预测模型 (Time Series Model)

**核心目录**: [`TimeSeries/`](TimeSeries/)

项目的核心研究模块，基于深度学习对热电电压数据进行建模与分析。

*   **模型架构**: 提供基于 PyTorch 实现的 LSTM 和 GRU 模型。
*   **数据处理**: 包含完整的数据预处理流水线（去噪、降采样、标准化）。
*   **训练平台**: 支持 GPU 加速训练 (CUDA)，集成 TensorBoard 可视化训练过程。
*   **功能模块**:
    *   `src/`: 模型定义与训练代码。
    *   `DA/`: 数据增强与去噪工具。
    *   `Sim_data/`: 合成数据生成工具，用于模型验证。

👉 **详细文档**: [TimeSeries/README.md](TimeSeries/README.md)

### 4. 天空图像处理系统 (Sky Image Processing)

**核心目录**: [`FigureProcess/`](FigureProcess/)

辅助分析工具，用于处理与辐射数据同步拍摄的天空图像。

*   **FPU (Figure Processing Unit)**: 一个基于 Tkinter 的图像分析 GUI 工具 (`FPU.py`)。
*   **功能**: 支持云量概率计算、像素级探针分析，用于探究天空云况与热电芯片发电效率的关联。

---

## 四. Raspberry Pi 3B 迁移与优化

本项目已针对 Raspberry Pi 3B 进行了专门适配，使其能够作为低功耗、长续航的边缘采集节点。

*   **低功耗配置**: 通过 `optimize_pi3b.sh` 脚本可降低约 50% 功耗。
*   **性能适配**: 采集与发送程序 (`Full_collector.py`, `pi_sender.py`) 占用资源极低，在 1GB 内存的 Pi 3B 上运行流畅。
*   **网络要求**: 实时系统需要 Pi 与主机处于同一局域网（或通过网线直连），离线采集模式则无需网络。

---

## 五. 常用命令速查

### 1. 数据采集 (Raspberry Pi)

```bash
# 启动采集服务
sudo systemctl start tec-collector.service
sudo systemctl stop tec-collector.service

# 查看实时日志
sudo journalctl -u tec-collector.service -f

# 启动实时数据发送 (用于实时系统)
cd ~/ads1115_project/Themoelectric/RealTimeSystem/Raspberry
python3 pi_sender.py
```

### 2. 实时系统 (Host PC)

```bash
# 1. 启动数据接收服务器
python RealTimeSystem/server.py

# 2. 启动 GUI 界面 (新终端)
python RealTimeSystem/gui_app.py
```

### 3. 模型训练 (Host PC)

```bash
# 训练 GRU 模型
cd TimeSeries/src
python train.py --model gru --epochs 100

# 启动 TensorBoard 查看训练曲线
tensorboard --logdir=runs
```

---

## 六. 文档导航

| 模块 | 文档名称 | 说明 |
| :--- | :--- | :--- |
| **采集** | [快速开始指南](DataCollectCode/docs/快速开始指南.md) | 从零搭建采集硬件 |
| **采集** | [自动化方案](DataCollectCode/docs/automation.md) | 设置开机自启与守护进程 |
| **实时** | [实时系统主页](RealTimeSystem/docs/README.md) | 实时系统架构与部署 |
| **实时** | [测试指南](RealTimeSystem/docs/TESTING_GUIDE.md) | 模拟数据测试流程 |
| **模型** | [模型训练](TimeSeries/docs/train.md) | 深度学习模型训练参数说明 |
| **模型** | [数据增强](TimeSeries/docs/data_augmentation_subsampling.md) | 数据预处理方法 |

---

## 七. 数据格式

### 1. 采集原始数据 (CSV)

路径: `DataCollectCode/data.csv` 或 `RealTimeSystem/realtime_data/received_data.csv`

```csv
Timestamp,DateTime,TEC1_Voltage(V),...,TEC8_Voltage(V)
1699876543.123,2024-11-13 10:15:43,0.003245,...,0.003002
```

### 2. 预测结果数据 (CSV)

路径: `RealTimeSystem/realtime_data/predictions.csv`

```csv
timestamp,prediction
1699876553.123,0.003310
```

---

## 八. 故障排除

### 常见问题

**Q: 实时系统 GUI 没有显示数据？**
*   检查 Pi 端 `pi_sender.py` 是否运行且无报错。
*   检查主机端 `server.py` 是否收到 POST 请求（控制台有日志）。
*   检查防火墙设置，确保主机端口（默认 5000）开放。

**Q: 模型训练提示显存不足？**
*   在 `train.py` 或配置文件中减小 `batch_size`（如设为 16 或 32）。

**Q: Raspberry Pi 找不到 I2C 设备？**
*   运行 `i2cdetect -y 1`。如果全为横杠，检查 ADS1115 接线及供电。

---

## 九. 系统要求

### 边缘采集端
*   **硬件**: Raspberry Pi 3B/4B/5
*   **系统**: Raspberry Pi OS (Bullseye/Bookworm)
*   **环境**: Python 3.7+, 依赖库见 `DataCollectCode/requirements.txt`

### 主机分析端
*   **系统**: Windows 10/11 或 Linux
*   **环境**: Python 3.8+
*   **依赖**:
    *   PyQt5 (界面)
    *   PyTorch (且建议配备 NVIDIA 显卡 + CUDA 以加速训练)
    *   Flask (服务器)
    *   完整依赖见 `TimeSeries/requirements.txt`

---

## 十. 联系方式

*   **项目仓库**: [Thermoelectric_Project](https://github.com/SmlCoke/Thermoelectric_Project)
*   **维护者**: SmlCoke
*   **邮箱**: j.feng.st05@gmail.com
