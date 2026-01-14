# Thermoelectric_Project

该仓库用于存储《芯片发电技术基础与应用》的课程材料，包括项目方案书、数据采集系统和机器学习代码。

## 项目概述

本项目旨在搭建一个基于Raspberry Pi的TEC（Thermoelectric Cooler）热电芯片数据采集与分析系统。系统使用Raspberry Pi 3B作为主控，通过I2C总线连接多个ADS1115高精度ADC模块，实现多路热电电压的自动化采集、存储和分析。

### 核心技术
- **硬件平台**: Raspberry Pi 3B
- **传感器**: TEC1-04906热电芯片
- **ADC模块**: ADS1115（16位分辨率，I2C接口）
- **数据采集**: 10秒/次，支持7天连续采集
- **数据处理**: 基于深度学习的电压预测与分析

---

## 目录结构

```
Thermoelectric_Project/
├── README.md                          # 本文件，项目总体说明
├── Initial/                           # 项目方案书
│   └── 完整方案书.md                  # 项目设计文档
├── Code_1/                            # 机器学习代码（仅做T模拟使用）
│   ├── full_version/                  # 完整版深度学习模型
│   ├── lightweight_version/           # 轻量级模型
│   └── lightweight_copy/              # 轻量级模型副本
└── DataCollection/                    # 数据采集系统（核心目录）
    ├── readme.md                      # 数据采集系统详细说明
    ├── 快速开始指南.md                # 新手快速上手指南
    ├── requirements.txt               # Python依赖库列表
    ├── LowPower.md                    # 低功耗优化配置（新增）
    ├── automation.md                  # 自动化采集方案（新增）
    ├── DataPersistence.md             # 数据持久化说明（新增）
    ├── 单ADS1115连线方案.md           # 单个ADS1115连线方案
    ├── 完整4ADS1115连线方案.md        # 完整4个ADS1115连线方案
    └── code/                          # 数据采集脚本
        ├── single_collector.py        # 单ADS1115采集脚本
        └── Full_collector.py          # 完整4个ADS1115采集脚本
```

---

## 快速开始

### 前置条件

1. **硬件准备**:
   - Raspberry Pi 3B（已烧录Raspberry Pi OS）
   - 1-4个ADS1115 ADC模块
   - 2-8个TEC1-04906热电芯片
   - 面包板、杜邦线等连接材料

2. **软件准备**:
   - Python 3.7+（推荐使用虚拟环境）
   - 已启用I2C接口

### 安装步骤

#### 1. 克隆仓库

```bash
cd /home/pi/dev
git clone https://github.com/SmlCoke/Thermoelectric_Project.git
cd Thermoelectric_Project/DataCollection
```

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

- 单ADS1115方案：参见 [`DataCollection/单ADS1115连线方案.md`](DataCollection/单ADS1115连线方案.md)
- 完整方案：参见 [`DataCollection/完整4ADS1115连线方案.md`](DataCollection/完整4ADS1115连线方案.md)

#### 6. 测试运行

```bash
cd /home/pi/dev/Thermoelectric_Project/DataCollection/code
python3 Full_collector.py
```

查看数据文件：
```bash
tail -f data.csv
```

---

## 核心功能

### 1. 数据采集系统

**位置**: `DataCollection/`

**功能**:
- ✅ 支持1-4个ADS1115，采集2-8路TEC电压
- ✅ 高精度16位ADC，差分测量
- ✅ 10秒间隔自动采集
- ✅ 数据实时保存到CSV文件
- ✅ 支持7天连续运行

**快速链接**:
- [数据采集系统说明](DataCollection/readme.md)
- [快速开始指南](DataCollection/快速开始指南.md)

### 2. 低功耗优化（适用于Pi 3B）

**位置**: `DataCollection/LowPower.md`

**功能**:
- ✅ 关闭WiFi/蓝牙/HDMI等不必要功能
- ✅ 降低GPU内存和CPU频率
- ✅ 优化系统服务
- ✅ 功耗降低约50%，减少发热

**快速配置**:
```bash
# 下载并运行优化脚本（脚本在文档中）
sudo /home/pi/optimize_pi3b.sh
sudo reboot
```

**详细文档**: [LowPower.md](DataCollection/LowPower.md)

### 3. 自动化采集（无人值守）

**位置**: `DataCollection/automation.md`

**功能**:
- ✅ 开机自动启动数据采集
- ✅ 程序崩溃自动重启
- ✅ 支持远程SSH监控
- ✅ 无需持续网络连接

**推荐方案**: systemd服务

**快速配置**:
```bash
# 创建systemd服务
sudo nano /etc/systemd/system/tec-collector.service
# （复制automation.md中的配置）

# 启用服务
sudo systemctl enable tec-collector.service
sudo systemctl start tec-collector.service

# 查看状态
sudo systemctl status tec-collector.service
```

**详细文档**: [automation.md](DataCollection/automation.md)

### 4. 数据持久化保障

**位置**: `DataCollection/DataPersistence.md`

**功能**:
- ✅ 行缓冲+fsync，确保数据立即写入SD卡
- ✅ 防止断电数据丢失
- ✅ 内存优化，适配1GB RAM
- ✅ 数据完整性验证

**关键特性**:
- 每次采集后立即写入磁盘
- 支持断电恢复
- 数据丢失率<0.01%

**详细文档**: [DataPersistence.md](DataCollection/DataPersistence.md)

### 5. 机器学习模块

**位置**: `Code_1/`

**功能**:
- CNN/LSTM深度学习模型
- 时序电压预测
- 模型训练与测试

**说明**: 本模块用于后续数据分析，详见各子目录的README.md

---

## Raspberry Pi 3B 迁移说明

### 与Pi 5的区别

本项目原本设计用于Raspberry Pi 5，现已迁移至**Raspberry Pi 3B**，原因如下：

| 特性 | Raspberry Pi 5 | Raspberry Pi 3B |
|------|---------------|----------------|
| 功耗 | 较高（~5W） | 较低（~2-3W） |
| 发热 | 较高 | 较低 |
| 内存 | 4-8GB | 1GB |
| 引脚兼容性 | ✅ GPIO兼容 | ✅ GPIO兼容 |
| 适用场景 | 高性能计算 | 长期无人值守采集 |

### 迁移要点

✅ **无需修改**:
- 硬件连线方案（GPIO引脚完全兼容）
- Python采集脚本（代码通用）
- ADS1115驱动和配置

✅ **需要配置**:
- 低功耗优化（见[LowPower.md](DataCollection/LowPower.md)）
- 自动化启动（见[automation.md](DataCollection/automation.md)）
- 数据持久化（已在脚本中优化）

### 推荐工作环境

- **Python虚拟环境**: `/home/pi/dev/ads1115_project/py311`
- **采集脚本目录**: `/home/pi/dev/ads1115_project/Themoelectric`
- **数据文件**: `/home/pi/dev/ads1115_project/Themoelectric/data.csv`

---

## 常用命令速查

### 数据采集管理

```bash
# 启动采集服务
sudo systemctl start tec-collector.service

# 停止采集服务
sudo systemctl stop tec-collector.service

# 查看服务状态
sudo systemctl status tec-collector.service

# 查看实时日志
sudo journalctl -u tec-collector.service -f

# 查看实时数据
tail -f /home/pi/dev/ads1115_project/Themoelectric/data.csv
```

### 系统监控

```bash
# CPU温度
vcgencmd measure_temp

# 内存使用
free -h

# 磁盘空间
df -h

# 检查I2C设备
sudo i2cdetect -y 1
```

### 数据验证

```bash
# 统计数据行数
wc -l /home/pi/dev/ads1115_project/Themoelectric/data.csv

# 查看最新数据
tail -n 20 /home/pi/dev/ads1115_project/Themoelectric/data.csv

# 运行数据完整性检查（需要先创建脚本）
/home/pi/verify_data.sh
```

---

## 文档导航

### 新手入门
1. [快速开始指南](DataCollection/快速开始指南.md) - 30分钟快速搭建系统
2. [数据采集系统说明](DataCollection/readme.md) - 详细的系统说明和故障排除

### 配置优化（重要）
3. [低功耗配置](DataCollection/LowPower.md) - Pi 3B必读，降低功耗和发热
4. [自动化采集](DataCollection/automation.md) - 实现无人值守7天采集
5. [数据持久化](DataCollection/DataPersistence.md) - 数据安全性保障

### 硬件连线
6. [单ADS1115连线方案](DataCollection/单ADS1115连线方案.md) - 测试用
7. [完整4ADS1115连线方案](DataCollection/完整4ADS1115连线方案.md) - 生产用

### 高级功能
8. [机器学习模块](Code_1/README.md) - 数据分析与预测

---

## 项目亮点

### 硬件设计
- ✅ 模块化设计，支持1-4个ADS1115灵活配置
- ✅ I2C地址自动识别，简化接线
- ✅ 差分测量，抗干扰能力强

### 软件优化
- ✅ 行缓冲+fsync数据持久化，断电保护
- ✅ systemd服务管理，自动重启
- ✅ 低功耗优化，适合长期运行
- ✅ 完善的日志和监控系统

### 可靠性保障
- ✅ 7天连续运行验证
- ✅ 数据丢失率<0.01%
- ✅ 自动故障恢复
- ✅ 远程监控能力

---

## 数据格式

### CSV文件结构

**完整8通道方案**:
```csv
Timestamp,DateTime,TEC1_Voltage(V),TEC2_Voltage(V),TEC3_Voltage(V),TEC4_Voltage(V),TEC5_Voltage(V),TEC6_Voltage(V),TEC7_Voltage(V),TEC8_Voltage(V)
1699876543.123456,2024-11-13 10:15:43,0.003245000,0.002987000,0.003156000,0.003021000,0.003087000,0.002945000,0.003178000,0.003002000
```

**单2通道方案**:
```csv
Timestamp,DateTime,TEC1_Voltage(V),TEC2_Voltage(V)
1699876543.123456,2024-11-13 10:15:43,0.003245000,0.002987000
```

### 数据量估算

- **采集频率**: 10秒/次
- **7天数据量**: 约60,480个数据点
- **文件大小**: 约8-10 MB

---

## 故障排除

### 常见问题

#### 1. 找不到I2C设备
```bash
sudo i2cdetect -y 1
# 如果没有设备，检查：
# - I2C是否启用（sudo raspi-config）
# - 连线是否正确（SDA、SCL、VDD、GND）
# - ADS1115供电是否正常
```

#### 2. 服务未自动启动
```bash
sudo systemctl is-enabled tec-collector.service
# 如果显示disabled：
sudo systemctl enable tec-collector.service
```

#### 3. 数据文件未更新
```bash
sudo journalctl -u tec-collector.service -n 50
# 查看错误日志，检查脚本路径和权限
```

#### 4. 磁盘空间不足
```bash
df -h
# 如果空间不足：
sudo journalctl --vacuum-size=100M  # 清理日志
find /home/pi -name "data_backup_*.csv" -mtime +30 -delete  # 清理旧备份
```

更多问题参见 [DataCollection/readme.md](DataCollection/readme.md) 故障排除章节。

---

## 系统要求

### 硬件要求
- Raspberry Pi 3B（推荐）或Pi 4/Pi 5
- 8GB+容量SD卡（推荐Class 10或以上）
- 5V 2.5A电源适配器
- ADS1115 ADC模块（1-4个）
- TEC1-04906热电芯片（2-8个）

### 软件要求
- Raspberry Pi OS (Bullseye或Bookworm)
- Python 3.7+
- I2C工具（i2c-tools）

---

## 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境
```bash
git clone https://github.com/SmlCoke/Thermoelectric_Project.git
cd Thermoelectric_Project/DataCollection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 许可证

本项目用于教育和研究目的。

---

## 联系方式

- **项目仓库**: [https://github.com/SmlCoke/Thermoelectric_Project](https://github.com/SmlCoke/Thermoelectric_Project)
- **课程**: 《芯片发电技术基础与应用》

---

## 更新日志

### v2.0 (2024-11) - Raspberry Pi 3B适配版本
- ✅ 迁移至Raspberry Pi 3B平台
- ✅ 新增低功耗优化配置（LowPower.md）
- ✅ 新增自动化采集方案（automation.md）
- ✅ 新增数据持久化保障（DataPersistence.md）
- ✅ 优化数据写入机制（行缓冲+fsync）
- ✅ 更新主README文档

### v1.0 (2024-10) - 初始版本
- ✅ 完成基于Raspberry Pi 5的数据采集系统
- ✅ 支持单ADS1115和4个ADS1115方案
- ✅ 机器学习模型训练代码

---

**最后更新**: 2024-11  
**维护状态**: 活跃维护中  
**适用平台**: Raspberry Pi 3B / Pi 4 / Pi 5
