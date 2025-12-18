"""
主机端 HTTP 接收服务

该模块负责：
1. 提供 Flask HTTP 服务，接收 Pi 发送的电压数据
2. 维护最近 60 个时间点的滑动窗口
3. 当窗口数据足够时触发推理
4. 提供数据接口供 GUI 使用

使用方式：
    python server.py --port 5000 --model-path ../TimeSeries/Prac_train/checkpoints/best_model.pth
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
from datetime import datetime
from collections import deque
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, asdict, field

# 添加 TimeSeries/src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TimeSeries', 'src'))

# 第三方库
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("请安装 Flask 和 flask-cors: pip install flask flask-cors")
    sys.exit(1)

import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """数据点结构"""
    timestamp: str
    values: List[float]
    received_at: float = field(default_factory=time.time)


class SlidingWindow:
    """
    滑动窗口类
    
    维护最近 N 个时间点的电压数据
    """
    
    def __init__(self, window_size: int = 60):
        """
        初始化滑动窗口
        
        参数:
            window_size: int, 窗口大小（时间点数量）
        """
        self.window_size = window_size
        self._data: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
        
        # 上一次预测结果
        self.last_prediction: Optional[np.ndarray] = None
        self.last_prediction_timestamp: Optional[str] = None
        
        # 预测历史（用于对比）
        self.prediction_history: deque = deque(maxlen=100)
        
        logger.info(f"滑动窗口初始化完成 (窗口大小: {window_size})")
    
    def add(self, data_point: DataPoint):
        """
        添加一个数据点到窗口
        
        参数:
            data_point: DataPoint, 数据点
        """
        with self._lock:
            self._data.append(data_point)
            logger.debug(f"添加数据点: {data_point.timestamp}, "
                        f"当前窗口大小: {len(self._data)}")
    
    def get_data(self) -> Optional[np.ndarray]:
        """
        获取当前窗口中的所有数据
        
        返回:
            np.ndarray or None, 形状 [window_size, 8] 或 None（数据不足时）
        """
        with self._lock:
            if len(self._data) < self.window_size:
                return None
            
            # 转换为 numpy 数组
            data = np.array([dp.values for dp in self._data])
            return data
    
    def get_timestamps(self) -> List[str]:
        """获取当前窗口中所有数据点的时间戳"""
        with self._lock:
            return [dp.timestamp for dp in self._data]
    
    def get_current_size(self) -> int:
        """获取当前窗口中的数据点数量"""
        with self._lock:
            return len(self._data)
    
    def is_ready(self) -> bool:
        """检查窗口数据是否足够进行推理"""
        return self.get_current_size() >= self.window_size
    
    def get_latest_values(self) -> Optional[List[float]]:
        """获取最新的一组电压值"""
        with self._lock:
            if len(self._data) == 0:
                return None
            return self._data[-1].values.copy()
    
    def get_latest_timestamp(self) -> Optional[str]:
        """获取最新数据点的时间戳"""
        with self._lock:
            if len(self._data) == 0:
                return None
            return self._data[-1].timestamp
    
    def save_prediction(self, prediction: np.ndarray, timestamp: str):
        """
        保存预测结果
        
        参数:
            prediction: np.ndarray, 预测结果
            timestamp: str, 预测时的时间戳
        """
        with self._lock:
            # 如果有上一次的预测，保存到历史记录
            if self.last_prediction is not None:
                self.prediction_history.append({
                    'timestamp': self.last_prediction_timestamp,
                    'prediction': self.last_prediction,
                    'actual': self.get_latest_values()
                })
            
            self.last_prediction = prediction
            self.last_prediction_timestamp = timestamp
    
    def get_comparison_data(self) -> Optional[Dict]:
        """
        获取预测与实际值的对比数据
        
        返回:
            Dict, 包含上一次预测和对应的实际值
        """
        with self._lock:
            if len(self.prediction_history) == 0:
                return None
            return self.prediction_history[-1].copy()
    
    def clear(self):
        """清空窗口数据"""
        with self._lock:
            self._data.clear()
            self.last_prediction = None
            self.last_prediction_timestamp = None
            self.prediction_history.clear()
            logger.info("滑动窗口已清空")


class DataServer:
    """
    数据服务器类
    
    提供 HTTP 接口接收数据，并管理滑动窗口
    """
    
    # 通道名称
    CHANNEL_NAMES = [
        "Yellow", "Ultraviolet", "Infrared", "Red",
        "Green", "Blue", "Transparent", "Violet"
    ]
    
    def __init__(
        self,
        port: int = 5000,
        window_size: int = 60,
        inference_callback: Optional[Callable] = None
    ):
        """
        初始化数据服务器
        
        参数:
            port: int, 服务端口
            window_size: int, 滑动窗口大小
            inference_callback: Callable, 推理回调函数
        """
        self.port = port
        self.window = SlidingWindow(window_size=window_size)
        self.inference_callback = inference_callback
        
        # 创建 Flask 应用
        self.app = Flask(__name__)
        CORS(self.app)
        
        # 注册路由
        self._register_routes()
        
        # 统计信息
        self.receive_count = 0
        self.start_time = None
        
        logger.info(f"数据服务器初始化完成 (端口: {port})")
    
    def _register_routes(self):
        """注册 Flask 路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查接口"""
            return jsonify({
                'status': 'ok',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'window_size': self.window.get_current_size(),
                'receive_count': self.receive_count
            })
        
        @self.app.route('/data', methods=['POST'])
        def receive_data():
            """接收电压数据"""
            try:
                data = request.get_json()
                
                # 验证数据格式
                if not data:
                    return jsonify({'error': '无效的JSON数据'}), 400
                
                if 'timestamp' not in data or 'values' not in data:
                    return jsonify({'error': '缺少必要字段: timestamp, values'}), 400
                
                values = data['values']
                if not isinstance(values, list) or len(values) != 8:
                    return jsonify({'error': 'values必须是包含8个元素的数组'}), 400
                
                # 创建数据点并添加到窗口
                data_point = DataPoint(
                    timestamp=data['timestamp'],
                    values=values
                )
                self.window.add(data_point)
                self.receive_count += 1
                
                # 检查是否可以进行推理
                inference_triggered = False
                if self.window.is_ready() and self.inference_callback is not None:
                    # 在后台线程中执行推理
                    threading.Thread(target=self._run_inference, daemon=True).start()
                    inference_triggered = True
                
                return jsonify({
                    'status': 'ok',
                    'window_size': self.window.get_current_size(),
                    'inference_triggered': inference_triggered
                })
                
            except Exception as e:
                logger.error(f"处理数据时出错: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """获取服务器状态"""
            return jsonify({
                'window_size': self.window.get_current_size(),
                'window_ready': self.window.is_ready(),
                'receive_count': self.receive_count,
                'latest_timestamp': self.window.get_latest_timestamp(),
                'uptime': time.time() - self.start_time if self.start_time else 0
            })
        
        @self.app.route('/window', methods=['GET'])
        def get_window_data():
            """获取当前窗口数据"""
            data = self.window.get_data()
            if data is None:
                return jsonify({
                    'ready': False,
                    'current_size': self.window.get_current_size(),
                    'required_size': self.window.window_size
                })
            
            return jsonify({
                'ready': True,
                'data': data.tolist(),
                'timestamps': self.window.get_timestamps()
            })
        
        @self.app.route('/latest', methods=['GET'])
        def get_latest():
            """获取最新的数据点"""
            values = self.window.get_latest_values()
            timestamp = self.window.get_latest_timestamp()
            
            if values is None:
                return jsonify({'error': '暂无数据'}), 404
            
            return jsonify({
                'timestamp': timestamp,
                'values': values,
                'channels': dict(zip(self.CHANNEL_NAMES, values))
            })
        
        @self.app.route('/prediction', methods=['GET'])
        def get_prediction():
            """获取最新的预测结果"""
            if self.window.last_prediction is None:
                return jsonify({'error': '暂无预测结果'}), 404
            
            return jsonify({
                'timestamp': self.window.last_prediction_timestamp,
                'prediction': self.window.last_prediction.tolist()
            })
        
        @self.app.route('/clear', methods=['POST'])
        def clear_window():
            """清空滑动窗口"""
            self.window.clear()
            return jsonify({'status': 'ok', 'message': '窗口已清空'})
    
    def _run_inference(self):
        """运行推理"""
        try:
            if self.inference_callback is not None:
                self.inference_callback()
        except Exception as e:
            logger.error(f"推理出错: {e}")
    
    def set_inference_callback(self, callback: Callable):
        """设置推理回调函数"""
        self.inference_callback = callback
    
    def run(self, debug: bool = False, threaded: bool = True):
        """
        启动服务器
        
        参数:
            debug: bool, 是否启用调试模式
            threaded: bool, 是否使用多线程处理请求
        """
        self.start_time = time.time()
        logger.info(f"启动 HTTP 服务器: http://0.0.0.0:{self.port}")
        self.app.run(
            host='0.0.0.0',
            port=self.port,
            debug=debug,
            threaded=threaded
        )
    
    def run_in_thread(self) -> threading.Thread:
        """
        在后台线程中启动服务器
        
        返回:
            threading.Thread, 服务器线程
        """
        self.start_time = time.time()
        
        def run_server():
            # 使用 Werkzeug 服务器
            from werkzeug.serving import make_server
            server = make_server('0.0.0.0', self.port, self.app, threaded=True)
            self._server = server
            logger.info(f"HTTP 服务器启动: http://0.0.0.0:{self.port}")
            server.serve_forever()
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        return thread
    
    def shutdown(self):
        """关闭服务器"""
        if hasattr(self, '_server'):
            self._server.shutdown()
            logger.info("HTTP 服务器已关闭")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='主机端数据接收服务器')
    
    parser.add_argument('--port', type=int, default=5000,
                       help='服务端口 (默认: 5000)')
    parser.add_argument('--window-size', type=int, default=60,
                       help='滑动窗口大小 (默认: 60)')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    
    args = parser.parse_args()
    
    # 创建服务器
    server = DataServer(
        port=args.port,
        window_size=args.window_size
    )
    
    # 启动服务器
    logger.info("=" * 50)
    logger.info("主机端数据接收服务器")
    logger.info("=" * 50)
    logger.info(f"端口: {args.port}")
    logger.info(f"窗口大小: {args.window_size}")
    logger.info("按 Ctrl+C 停止服务器")
    logger.info("=" * 50)
    
    try:
        server.run(debug=args.debug)
    except KeyboardInterrupt:
        logger.info("\n服务器被用户中断")


if __name__ == '__main__':
    main()
