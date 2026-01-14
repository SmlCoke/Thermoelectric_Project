# 测试指南：验证修复效果

本文档提供详细的测试步骤，帮助您验证修复是否成功解决了单步预测后系统卡死的问题。

## 前置条件

### 主机端（PC）
确保已安装以下依赖：
```bash
pip install flask flask-cors numpy torch PyQt5 matplotlib scikit-learn
```

### 树莓派端
确保已安装以下依赖：
```bash
pip install requests
```

## 测试方案 1：模拟数据测试（推荐首选）

此方案不需要真实硬件，适合快速验证修复效果。

### 步骤 1：启动主机端 GUI

在主机上打开终端，执行：

```bash
cd /path/to/Thermoelectric_Project/RealTimeSystem
python gui_app.py --port 5000
```

**预期输出：**
```
2025-12-21 XX:XX:XX - INFO - 使用模拟推理引擎 (仅用于测试)
2025-12-21 XX:XX:XX - INFO - 滑动窗口初始化完成 (窗口大小: 60)
2025-12-21 XX:XX:XX - INFO - 数据服务器初始化完成 (端口: 5000)
============================================================
热电芯片电压实时监测与预测系统
============================================================
...
GUI 已启动
HTTP 服务: http://0.0.0.0:5000
等待 Raspberry Pi 发送数据...
============================================================
```

GUI 窗口应该正常打开，显示空白图表。

### 步骤 2：在另一台设备上模拟树莓派发送

**选项 A：在同一台主机上模拟（最简单）**

打开另一个终端：

```bash
cd /path/to/Thermoelectric_Project/RealTimeSystem
python pi_sender.py --host 127.0.0.1 --port 5000 --test --interval 10
```

**选项 B：在另一台设备上模拟**

首先获取主机 IP 地址：
```bash
# Windows
ipconfig

# Linux/Mac
ip addr show
# 或
ifconfig
```

然后在另一台设备上执行：
```bash
python pi_sender.py --host <主机IP> --port 5000 --test --interval 10
```

### 步骤 3：切换到单步预测模式

在 GUI 窗口中：
1. 找到"预测设置"面板
2. 在"预测步数"下拉框中选择"1 步预测"

### 步骤 4：观察系统运行

#### 主机端 GUI 应该显示：
- "系统状态"显示"数据就绪"或"推理完成"
- "窗口数据"显示 "60 / 60"（窗口填满后）
- 图表持续更新，显示蓝色历史数据线和红色预测点
- **关键：图表应该流畅更新，没有卡顿**

#### 主机端终端应该持续输出：
```
2025-12-21 XX:XX:XX - INFO - 192.168.X.X - - [21/Dec/2025 XX:XX:XX] "POST /data HTTP/1.1" 200 -
2025-12-21 XX:XX:XX - INFO - 推理完成: 1 步预测
2025-12-21 XX:XX:XX - INFO - 192.168.X.X - - [21/Dec/2025 XX:XX:XX] "POST /data HTTP/1.1" 200 -
2025-12-21 XX:XX:XX - INFO - 推理完成: 1 步预测
...
```

**重点：** 应该连续显示 "POST /data HTTP/1.1" 200，没有中断。

#### Pi 端（或模拟器）终端应该持续输出：
```
2025-12-21 XX:XX:XX - INFO - 数据发送成功 [#1]: 2025-12-21 XX:XX:XX
2025-12-21 XX:XX:XX - INFO - 数据发送成功 [#2]: 2025-12-21 XX:XX:XX
2025-12-21 XX:XX:XX - INFO - 数据发送成功 [#3]: 2025-12-21 XX:XX:XX
...
```

**重点：** 应该全部显示"数据发送成功"，**不应该出现任何 "请求超时" 警告**。

### 步骤 5：验证修复成功

如果满足以下所有条件，说明修复成功：

- ✅ GUI 界面持续流畅更新，没有冻结
- ✅ 主机端持续收到数据（HTTP 200）
- ✅ Pi 端没有出现超时警告
- ✅ 预测点（红色）持续出现在图表上
- ✅ 系统可以运行至少 5 分钟以上不卡死

### 步骤 6：测试 10 步预测（可选）

在 GUI 中切换回"10 步预测"，重复步骤 4-5，验证系统同样运行正常。

---

## 测试方案 2：真实数据测试

如果您有完整的硬件设备（Raspberry Pi + ADS1115），可以使用真实数据进行测试。

### 在树莓派上

#### 启动数据采集（如果还没启动）

```bash
# 方式 1：使用 systemd 服务
sudo systemctl start tec-collector
sudo systemctl status tec-collector

# 方式 2：手动启动
cd /home/pi/dev/ads1115_project/Themoelectric
python Full_collector.py --interval 10
```

#### 启动数据转发

```bash
python pi_sender.py --host <主机IP> --port 5000 --csv-dir /home/pi/dev/ads1115_project/Themoelectric
```

### 在主机上

```bash
cd /path/to/Thermoelectric_Project/RealTimeSystem
python gui_app.py --port 5000 --model-path ../TimeSeries/Prac_train/checkpoints/best_model.pth
```

然后按照方案 1 的步骤 3-6 进行验证。

---

## 问题排查

### 问题 1：Pi 端仍然出现超时

**可能原因：**
1. 网络连接不稳定
2. 防火墙阻止连接
3. IP 地址或端口配置错误

**排查步骤：**
```bash
# 在 Pi 上测试连接
curl http://<主机IP>:5000/health

# 应该返回类似：
# {"receive_count":0,"status":"ok","timestamp":"...","window_size":0}
```

### 问题 2：GUI 启动失败

**可能原因：**
- PyQt5 安装不正确
- 缺少图形界面环境

**解决方案：**
```bash
# 重新安装 PyQt5
pip uninstall PyQt5
pip install PyQt5

# 检查显示环境
echo $DISPLAY  # Linux
```

### 问题 3：图表不更新

**可能原因：**
- 窗口数据不足（需要至少 60 个数据点）
- 推理引擎加载失败

**检查方式：**
- 查看 GUI 左侧"系统状态"面板
- "窗口数据"应该显示 "60 / 60"
- 查看终端日志是否有错误信息

---

## 性能基准

### 预期性能指标

| 指标 | 修复前 | 修复后 | 说明 |
|------|--------|--------|------|
| GUI 更新延迟 | 50-200ms | 5-10ms | 单次图表更新时间 |
| HTTP 响应时间 | 不稳定，经常超时 | 1-2ms | Flask 返回 200 的时间 |
| Pi 超时率 | >50% | 0% | 10s 内未收到响应的比率 |
| CPU 占用 | 高，有峰值 | 低，平稳 | 主线程 CPU 使用率 |

### 监控方式

**在主机上监控 CPU 使用率：**

Linux/Mac:
```bash
top -p $(pgrep -f gui_app.py)
```

Windows:
```
任务管理器 -> 详细信息 -> 找到 python.exe
```

**正常情况下：**
- CPU 使用率应该在 5-15% 之间波动
- 不应该有持续的 100% CPU 占用

---

## 成功标准

测试通过的标准：

1. ✅ **稳定性测试**：系统连续运行 10 分钟以上，无超时、无卡死
2. ✅ **单步预测测试**：切换到 1 步预测，运行 5 分钟无问题
3. ✅ **10步预测测试**：切换到 10 步预测，运行 5 分钟无问题
4. ✅ **通道切换测试**：切换不同通道显示，界面响应流畅
5. ✅ **性能测试**：CPU 占用正常，无异常峰值

如果所有测试都通过，说明修复完全成功！

---

## 反馈

如果遇到任何问题，请记录以下信息：

1. **环境信息**：
   - 操作系统版本
   - Python 版本
   - PyQt5、matplotlib、Flask 版本

2. **日志信息**：
   - 主机端终端完整日志
   - Pi 端终端完整日志
   - 错误截图

3. **重现步骤**：
   - 具体操作步骤
   - 预期行为 vs 实际行为

请将以上信息提供给开发团队进行进一步分析。
