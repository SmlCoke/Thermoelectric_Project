# 主机端 HTTP 接收服务 (server.py)

## 概述

该模块运行在主机端（Windows/Linux PC），提供 Flask HTTP 服务来接收来自 Raspberry Pi 的电压数据。

## 功能特点

1. **HTTP 数据接收** - 提供 `/data` 接口接收 POST 请求
2. **滑动窗口管理** - 维护最近 60 个时间点的数据
3. **推理触发** - 当窗口数据足够时自动触发推理
4. **状态监控** - 提供多个状态查询接口
5. **线程安全** - 支持并发请求处理

## API 接口

### POST /data

接收电压数据。

请求体：
```json
{
  "timestamp": "2025-01-12 14:23:10",
  "values": [v1, v2, v3, v4, v5, v6, v7, v8]
}
```

响应：
```json
{
  "status": "ok",
  "window_size": 45,
  "inference_triggered": false
}
```

### GET /health

健康检查接口。

响应：
```json
{
  "status": "ok",
  "timestamp": "2025-01-12 14:23:10",
  "window_size": 45,
  "receive_count": 100
}
```

### GET /status

获取服务器状态。

响应：
```json
{
  "window_size": 60,
  "window_ready": true,
  "receive_count": 100,
  "latest_timestamp": "2025-01-12 14:23:10",
  "uptime": 3600.5
}
```

### GET /window

获取当前滑动窗口中的所有数据。

### GET /latest

获取最新的数据点。

### GET /prediction

获取最新的预测结果。

### POST /clear

清空滑动窗口数据。

## 使用方法

### 独立运行

```bash
python server.py --port 5000 --window-size 60
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --port | 5000 | 服务端口 |
| --window-size | 60 | 滑动窗口大小 |
| --debug | False | 启用调试模式 |

### 与 GUI 集成

在 GUI 程序中，服务器以后台线程方式运行：

```python
from server import DataServer

# 创建服务器
server = DataServer(port=5000, window_size=60)

# 设置推理回调
server.set_inference_callback(on_inference_ready)

# 在后台线程启动
server.run_in_thread()
```

## SlidingWindow 类

滑动窗口类用于维护最近 N 个时间点的数据。

### 主要方法

- `add(data_point)` - 添加数据点
- `get_data()` - 获取窗口数据 (numpy array)
- `is_ready()` - 检查数据是否足够
- `get_latest_values()` - 获取最新值
- `save_prediction(prediction, timestamp)` - 保存预测结果
- `clear()` - 清空窗口

## 依赖

```
flask>=2.0.0
flask-cors>=3.0.0
numpy>=1.19.0
```

安装依赖：

```bash
pip install flask flask-cors numpy
```

## 日志

服务器会输出详细的日志信息，包括：
- 接收到的数据
- 推理触发时机
- 错误信息

## 注意事项

1. 确保防火墙允许指定端口的入站连接
2. 服务器默认绑定到 `0.0.0.0`，允许所有网络接口访问
3. 生产环境建议使用 gunicorn 或其他 WSGI 服务器
