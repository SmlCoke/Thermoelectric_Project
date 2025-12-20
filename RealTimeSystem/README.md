# 热电芯片电压实时监测与预测系统

## 系统概述

本系统实现了热电芯片电压的实时采集、传输、预测和可视化功能。系统由两个独立的程序组成：

1. **边缘端 (Raspberry Pi)**: 负责数据采集和发送
2. **主机端 (Windows/Linux PC)**: 负责数据接收、模型推理和可视化

## 系统架构

```
┌─────────────────────────────────────┐         HTTP POST          ┌─────────────────────────────┐
│         Raspberry Pi 3B             │  ─────────────────────────> │        主机端 PC           │
│                                     │         /data               │                             │
│  ┌───────────────────────────────┐  │                             │  ┌───────────────────────┐  │
│  │   Full_collector.py           │  │    {                        │  │    Flask Server       │  │
│  │   (或 mock_collector.py)      │  │      "timestamp": "...",    │  │    (server.py)        │  │
│  │   - 采集8通道电压数据           │  │      "values": [v1...v8]    │  └───────────┬───────────┘  │
│  │   - 写入CSV文件                │  │    }                        │              │              │
│  └───────────────┬───────────────┘  │                             │              ▼              │
│                  │                  │                             │  ┌───────────────────────┐  │
│                  ▼ (读取CSV)        │                             │  │   Sliding Window      │  │
│  ┌───────────────────────────────┐  │                             │  │   (60 time points)    │  │
│  │        pi_sender.py           │──┼────────────────────────────>│  └───────────┬───────────┘  │
│  │   - 读取CSV最新数据            │  │                             │              │              │
│  │   - HTTP POST转发              │  │                             │              ▼              │
│  └───────────────────────────────┘  │                             │  ┌───────────────────────┐  │
│                                     │                             │  │  Inference Engine     │  │
└─────────────────────────────────────┘                             │  │  (LSTM/GRU Model)     │  │
                                                                    │  └───────────┬───────────┘  │
      Pi 与主机通过网线连接                                           │              │              │
                                                                    │              ▼              │
                                                                    │  ┌───────────────────────┐  │
                                                                    │  │    GUI Application    │  │
                                                                    │  │    (PyQt5 + MPL)      │  │
                                                                    │  └───────────────────────┘  │
                                                                    └─────────────────────────────┘
```

## 通信方式

### HTTP POST 数据传输

- **协议**: HTTP
- **方法**: POST
- **端点**: `/data`
- **内容类型**: `application/json`

**请求格式**:
```json
{
  "timestamp": "2025-01-12 14:23:10",
  "values": [0.52, 0.61, 0.55, 0.53, 0.58, 0.54, 0.56, 0.57]
}
```

**通道顺序** (values 数组的顺序):
1. Yellow (黄色) - 对应 CSV 中的 TEC3
2. Ultraviolet (紫外) - 对应 CSV 中的 TEC7
3. Infrared (红外) - 对应 CSV 中的 TEC6
4. Red (红色) - 对应 CSV 中的 TEC5
5. Green (绿色) - 对应 CSV 中的 TEC2
6. Blue (蓝色) - 对应 CSV 中的 TEC1
7. Transparent (透明) - 对应 CSV 中的 TEC8
8. Violet (紫色) - 对应 CSV 中的 TEC4

**CSV 文件中 TEC 通道与颜色的映射关系**:
- TEC1_Optimal(V) → Blue
- TEC2_Optimal(V) → Green
- TEC3_Optimal(V) → Yellow
- TEC4_Optimal(V) → Violet
- TEC5_Optimal(V) → Red
- TEC6_Optimal(V) → Infrared
- TEC7_Optimal(V) → Ultraviolet
- TEC8_Optimal(V) → Transparent

## 文件结构

```
RealTimeSystem/
├── README.md                      # 本文档
├── init.md                        # 任务需求说明
│
├── pi_sender.py                   # 边缘端数据发送模块
├── pi_sender.md                   # 发送模块说明文档
│
├── server.py                      # 主机端 HTTP 接收服务
├── server.md                      # 服务端说明文档
│
├── inference_engine.py            # 推理引擎模块
├── inference_engine.md            # 推理引擎说明文档
│
├── gui_app.py                     # GUI 可视化应用程序
├── gui_app.md                     # GUI 应用说明文档
│
└── DataCollectContrl/             # 数据采集控制脚本
    ├── Full_collector.py          # 真实数据采集脚本 (需要ADS1115硬件)
    ├── mock_collector.py          # 模拟数据采集脚本 (用于测试)
    └── tec-collector.txt          # systemd服务配置文件
```

## 快速启动

### 1. 安装依赖

**主机端 (PC)**:
```bash
pip install flask flask-cors numpy torch PyQt5 matplotlib scikit-learn
```

**边缘端 (Raspberry Pi)**:
```bash
pip install requests
```

### 2. 启动主机端 GUI 程序

```bash
cd RealTimeSystem

# 使用模拟推理引擎（测试模式）
python gui_app.py --port 5000

# 使用训练好的模型
python gui_app.py --port 5000 --model-path ../TimeSeries/Prac_train/checkpoints/best_model.pth
```

### 3. 启动边缘端程序

在 Raspberry Pi 上:

**方式A: 使用模拟数据测试（无需ADS1115硬件）**
```bash
# 终端1: 启动模拟数据采集
cd /home/pi/dev/ads1115_project/Themoelectric
python mock_collector.py --interval 10

# 终端2: 启动数据转发
python pi_sender.py --host <主机IP地址> --port 5000 --csv-dir /home/pi/dev/ads1115_project/Themoelectric
```

**方式B: 使用真实数据（需要ADS1115硬件）**
```bash
# 启动真实数据采集服务
sudo systemctl start tec-collector

# 启动数据转发
python pi_sender.py --host <主机IP地址> --port 5000 --csv-dir /home/pi/dev/ads1115_project/Themoelectric
```

**方式C: 纯测试模式（不需要CSV文件）**
```bash
python pi_sender.py --host <主机IP地址> --port 5000 --test
```

### 4. 网络配置

确保 Pi 和主机在同一网络中：
- 通过网线直连
- 或连接到同一局域网

获取主机 IP 地址：
```bash
# Windows
ipconfig

# Linux
ip addr
```

## 模块说明

### DataCollectContrl/Full_collector.py

真实数据采集脚本，运行在 Raspberry Pi 上，使用 ADS1115 ADC 采集 8 通道热电芯片电压。

- **文件路径**: `/home/pi/dev/ads1115_project/Themoelectric/Full_collector.py`
- **采集间隔**: 10 秒
- **输出文件**: `TEC_multi_gain_data_{timestamp}.csv`

### DataCollectContrl/mock_collector.py

模拟数据采集脚本，用于在没有 ADS1115 硬件的情况下测试系统。

主要功能：
- 生成具有趋势和噪声的模拟电压数据
- CSV 输出格式与 Full_collector.py 完全一致
- 可配置采集间隔和输出目录

### pi_sender.py

负责在 Raspberry Pi 上读取 CSV 数据并通过 HTTP POST 发送到主机端。

主要功能：
- 从 CSV 文件读取最新电压数据
- HTTP POST 数据发送
- 网络异常自动重试
- 本地 CSV 数据备份
- 连接状态检查

### server.py

提供 Flask HTTP 服务，接收并管理数据。

主要功能：
- `/data` 接口接收 POST 数据
- 滑动窗口数据管理 (最近 60 个时间点)
- 数据就绪时触发推理
- 提供状态查询 API

### inference_engine.py

封装模型加载和推理逻辑。

主要功能：
- 加载训练好的 LSTM/GRU 模型
- 支持 1-step 和 multi-step 预测
- 自动数据标准化/反标准化
- 提供模拟推理引擎用于测试

### gui_app.py

PyQt5 图形用户界面应用程序。

主要功能：
- 实时数据可视化 (matplotlib)
- 通道选择 (单通道/全通道)
- 预测步数控制 (1步/10步)
- 系统状态监控
- 内置 HTTP 服务器

## GUI 界面说明

### 系统状态
- **等待数据**: 窗口数据不足 60 个时间点
- **数据就绪**: 数据足够，等待或已完成推理
- **推理中**: 正在执行模型推理
- **推理完成**: 推理完成，图表已更新

### 图表显示
- **历史数据**: 实线，显示最近 60 个时间点
- **预测结果**: 虚线 + 圆点标记
- **颜色**: 各通道使用对应的颜色标识

## API 接口

### POST /data
接收电压数据

### GET /health
健康检查

### GET /status
获取服务器状态

### GET /window
获取当前窗口数据

### GET /latest
获取最新数据点

### GET /prediction
获取最新预测结果

### POST /clear
清空滑动窗口

## 故障排除

### 1. 连接失败

检查：
- 网络连接是否正常
- IP 地址是否正确
- 防火墙是否允许端口访问

### 2. 模型加载失败

检查：
- 模型文件路径是否正确
- PyTorch 版本是否兼容
- 是否有 scaler.pkl 文件

### 3. GUI 无法显示

检查：
- PyQt5 是否正确安装
- 是否在有图形界面的环境中运行
- matplotlib 后端是否为 Qt5Agg

## 扩展开发

### 添加新的预测步数选项

在 `gui_app.py` 的 `_create_control_panel()` 方法中修改 `steps_combo`。

### 自定义图表样式

修改 `_plot_all_channels()` 和 `_plot_single_channel()` 方法。

### 添加数据导出功能

在控制面板中添加导出按钮，实现数据保存到文件的功能。

## 许可证

MIT License
