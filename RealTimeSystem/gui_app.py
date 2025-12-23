"""
GUI 可视化应用程序

该模块负责：
1. 使用 PyQt5 构建专业的图形用户界面
2. 使用 matplotlib 进行实时数据可视化
3. 显示历史数据、预测结果和对比信息
4. 提供用户交互控件（通道选择、预测步数等）

使用方式：
    python gui_app.py --port 5000 --model-path ../TimeSeries/Prac_train/checkpoints/best_model.pth
"""

import os
import sys

# [新增] 在导入 numpy/torch 之前设置环境变量
# 这对于防止多线程环境下的死锁至关重要
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# 允许某些库的重复加载（解决某些环境下的崩溃）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import logging
import argparse
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import deque

# 第三方库
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QComboBox, QPushButton, QGroupBox, QStatusBar, QFrame,
        QGridLayout, QSplitter, QSizePolicy, QSpacerItem
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt5.QtGui import QFont, QPalette, QColor
except ImportError:
    print("请安装 PyQt5: pip install PyQt5")
    sys.exit(1)

try:
    import matplotlib
    # 只在未设置后端时才设置 Qt5Agg 后端
    if matplotlib.get_backend() == 'agg' or not matplotlib.get_backend():
        matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    import matplotlib.pyplot as plt
except ImportError:
    print("请安装 matplotlib: pip install matplotlib")
    sys.exit(1)

import numpy as np

# 导入自定义模块
from server import DataServer, SlidingWindow
from inference_engine import create_inference_engine, InferenceEngine

# [新增] 配置 Times New Roman 字体
try:
    # Windows font path
    TIMES_FONT_PATH = r'C:\Windows\Fonts\times.ttf'
    T_16 = FontProperties(fname=TIMES_FONT_PATH, size=16)
    T_14 = FontProperties(fname=TIMES_FONT_PATH, size=14)
    T_12 = FontProperties(fname=TIMES_FONT_PATH, size=12)
    T_10 = FontProperties(fname=TIMES_FONT_PATH, size=10)
    T_9 = FontProperties(fname=TIMES_FONT_PATH, size=9)
    T_8 = FontProperties(fname=TIMES_FONT_PATH, size=8)
except:
    # Fallback to default if Times New Roman not available
    T_16 = FontProperties(size=16)
    T_14 = FontProperties(size=14)
    T_12 = FontProperties(size=12)
    T_10 = FontProperties(size=10)
    T_9 = FontProperties(size=9)
    T_8 = FontProperties(size=8)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 信号类，用于线程间通信
class SignalEmitter(QObject):
    """信号发射器，用于从后台线程更新 GUI"""
    data_updated = pyqtSignal()
    inference_completed = pyqtSignal(object)
    status_changed = pyqtSignal(str)


class PlotCanvas(FigureCanvas):
    """
    Matplotlib 绑定的 PyQt5 画布
    
    负责绑定 matplotlib 图表和 PyQt5 界面
    """
    
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        """
        初始化画布
        
        参数:
            parent: QWidget, 父控件
            width: int, 宽度（英寸）
            height: int, 高度（英寸）
            dpi: int, 分辨率
        """
        # 创建 Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor('#F0F0F0')
        
        super(PlotCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # 设置大小策略
        FigureCanvas.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)
        
        # 初始化子图
        self.axes = []
        self._init_subplots()
    
    def _init_subplots(self):
        """初始化子图布局"""
        # 清除现有图形
        self.fig.clear()
        self.axes = []
        
        # 创建一个主图（单通道或全通道会动态调整）
        ax = self.fig.add_subplot(111)
        self.axes.append(ax)
        
        self.fig.tight_layout(pad=3.0)
    
    def setup_multi_channel(self, num_channels: int = 8):
        """
        设置多通道布局（4行2列）
        
        参数:
            num_channels: int, 通道数量
        """
        self.fig.clear()
        self.axes = []
        
        rows, cols = 4, 2
        for i in range(num_channels):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            self.axes.append(ax)
        
        self.fig.tight_layout(pad=2.0)
        self.draw()
    
    def setup_single_channel(self):
        """设置单通道布局"""
        self.fig.clear()
        self.axes = []
        
        ax = self.fig.add_subplot(111)
        self.axes.append(ax)
        
        self.fig.tight_layout(pad=3.0)
        self.draw()
    
    def clear_all(self):
        """清除所有图表"""
        for ax in self.axes:
            ax.clear()
        self.draw()


class MainWindow(QMainWindow):
    """
    主窗口类
    
    实现完整的 GUI 界面和功能
    """
    
    # 通道名称和颜色
    CHANNEL_NAMES = [
        "Yellow", "Ultraviolet", "Infrared", "Red",
        "Green", "Blue", "Transparent", "Violet"
    ]
    
    CHANNEL_COLORS = [
        '#FFD700', '#9400D3', '#8B0000', '#FF0000',
        '#00FF00', '#0000FF', '#808080', '#EE82EE'
    ]
    
    def __init__(
        self,
        server: DataServer,
        inference_engine,
        parent=None
    ):
        """
        初始化主窗口
        
        参数:
            server: DataServer, 数据服务器实例
            inference_engine: InferenceEngine, 推理引擎实例
            parent: QWidget, 父控件
        """
        super(MainWindow, self).__init__(parent)
        
        self.server = server
        self.engine = inference_engine
        
        # 信号发射器
        self.signals = SignalEmitter()
        self.signals.data_updated.connect(self._on_data_updated)
        self.signals.inference_completed.connect(self._on_inference_completed)
        self.signals.status_changed.connect(self._on_status_changed)
        
        # 当前状态
        self.current_channel = -1  # -1 表示全通道
        self.predict_steps = 10
        self.last_prediction = None
        self.prediction_history = deque(maxlen=10)
        
        # CSV 文件读取配置
        self.csv_data_file = server.csv_file
        self.csv_prediction_file = server.prediction_file
        self.last_data_row_count = 0
        self.last_prediction_row_count = 0
        
        # 初始化界面
        self._init_ui()
        
        # 不再使用推理回调，改为定时读取CSV
        # self.server.set_inference_callback(self._run_inference)
        
        # 启动定时器更新界面（从CSV读取）
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_from_csv)
        self.update_timer.start(2000)  # 每2秒从CSV读取一次（用户要求：数据接收后过一会再可视化）
    
    def _init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Thermoelectric Chip Voltage Real-time Monitoring and Prediction System")
        self.setMinimumSize(1400, 900)  # Increased from 1200x800
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QGroupBox {
                font-family: 'Times New Roman';
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                font-family: 'Times New Roman';
                font-size: 12px;
            }
            QComboBox {
                font-family: 'Times New Roman';
                font-size: 11px;
                padding: 6px;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
            }
            QPushButton {
                font-family: 'Times New Roman';
                font-size: 12px;
                padding: 10px 18px;
                border: none;
                border-radius: 4px;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
        """)
        
        # 创建中心控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧：控制面板
        left_panel = self._create_control_panel()
        splitter.addWidget(left_panel)
        
        # 右侧：图表区域
        right_panel = self._create_plot_panel()
        splitter.addWidget(right_panel)
        
        # Set split ratio (more space for plot area)
        splitter.setSizes([350, 1050])  # Adjusted for larger window
        
        # 创建状态栏
        self._create_status_bar()
    
    def _create_control_panel(self) -> QWidget:
        """Create control panel"""
        panel = QWidget()
        panel.setMinimumWidth(350)  # Increased width
        layout = QVBoxLayout(panel)
        layout.setSpacing(18)  # Increased spacing
        
        # System Status Group
        status_group = QGroupBox("System Status")
        status_layout = QGridLayout()
        status_layout.setSpacing(10)
        
        # Status label
        self.status_label = QLabel("Waiting for data...")
        self.status_label.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 13px;")
        status_layout.addWidget(QLabel("Current Status:"), 0, 0)
        status_layout.addWidget(self.status_label, 0, 1)
        
        # Window size
        self.window_label = QLabel("0 / 60")
        status_layout.addWidget(QLabel("Window Data:"), 1, 0)
        status_layout.addWidget(self.window_label, 1, 1)
        
        # Receive count
        self.receive_label = QLabel("0")
        status_layout.addWidget(QLabel("Receive Count:"), 2, 0)
        status_layout.addWidget(self.receive_label, 2, 1)
        
        # Latest timestamp
        self.timestamp_label = QLabel("--")
        status_layout.addWidget(QLabel("Latest Data:"), 3, 0)
        status_layout.addWidget(self.timestamp_label, 3, 1)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Channel Selection Group
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout()
        channel_layout.setSpacing(8)
        
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("All Channels", -1)
        for i, name in enumerate(self.CHANNEL_NAMES):
            self.channel_combo.addItem(f"{i+1}. {name}", i)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        
        channel_layout.addWidget(QLabel("Select Display Channel:"))
        channel_layout.addWidget(self.channel_combo)
        
        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)
        
        # Prediction Settings Group
        predict_group = QGroupBox("Prediction Settings")
        predict_layout = QVBoxLayout()
        predict_layout.setSpacing(8)
        
        self.steps_combo = QComboBox()
        self.steps_combo.addItem("1 Step Prediction", 1)
        self.steps_combo.addItem("10 Steps Prediction", 10)
        self.steps_combo.setCurrentIndex(1)  # Default 10 steps
        self.steps_combo.currentIndexChanged.connect(self._on_steps_changed)
        
        predict_layout.addWidget(QLabel("Prediction Steps:"))
        predict_layout.addWidget(self.steps_combo)
        
        predict_group.setLayout(predict_layout)
        layout.addWidget(predict_group)
        
        # Current Voltage Values Group
        voltage_group = QGroupBox("Current Voltage Values")
        voltage_layout = QGridLayout()
        voltage_layout.setSpacing(8)
        
        self.voltage_labels = []
        for i, name in enumerate(self.CHANNEL_NAMES):
            row, col = i // 2, i % 2
            label_name = QLabel(f"{name}:")
            label_name.setStyleSheet(f"color: {self.CHANNEL_COLORS[i]}; font-weight: bold;")
            label_value = QLabel("--")
            label_value.setStyleSheet("font-family: 'Courier New'; font-size: 11px;")
            voltage_layout.addWidget(label_name, row, col * 2)
            voltage_layout.addWidget(label_value, row, col * 2 + 1)
            self.voltage_labels.append(label_value)
        
        voltage_group.setLayout(voltage_layout)
        layout.addWidget(voltage_group)
        
        # Operation Buttons Group
        button_group = QGroupBox("Operations")
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)
        
        # Clear data button
        clear_btn = QPushButton("Clear Data")
        clear_btn.clicked.connect(self._on_clear_clicked)
        clear_btn.setStyleSheet("background-color: #f44336;")
        button_layout.addWidget(clear_btn)
        
        # Refresh chart button
        refresh_btn = QPushButton("Refresh Chart")
        refresh_btn.clicked.connect(self._update_plot)
        button_layout.addWidget(refresh_btn)
        
        button_group.setLayout(button_layout)
        layout.addWidget(button_group)
        
        # Add stretch
        layout.addStretch()
        
        # Model Information
        model_group = QGroupBox("Model Information")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(6)
        
        model_info = self.engine.get_model_info()
        model_type = model_info.get('model_type', 'N/A').upper()
        window_size = model_info.get('window_size', 'N/A')
        device = str(model_info.get('device', 'N/A'))
        
        model_layout.addWidget(QLabel(f"Model Type: {model_type}"))
        model_layout.addWidget(QLabel(f"Window Size: {window_size}"))
        model_layout.addWidget(QLabel(f"Device: {device}"))
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        return panel
    
    def _create_plot_panel(self) -> QWidget:
        """Create plot panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        
        # Create title with Times New Roman font
        # Create title with Times New Roman font
        title_label = QLabel("Real-time Voltage Data and Prediction")
        title_label.setFont(QFont('Times New Roman', 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333333; padding: 8px;")
        layout.addWidget(title_label)
        
        # Create matplotlib canvas
        self.canvas = PlotCanvas(self, width=12, height=9, dpi=100)
        layout.addWidget(self.canvas)
        
        # Legend explanation with Times New Roman font
        legend_layout = QHBoxLayout()
        legend_layout.addStretch()
        
        legend_items = [
            ("Historical Data", "#2196F3"),
            ("Prediction", "#FF5722"),
            ("Actual Measurement", "#4CAF50")
        ]
        
        for text, color in legend_items:
            label = QLabel(f"● {text}")
            label.setStyleSheet(f"color: {color}; font-weight: bold; font-family: 'Times New Roman'; font-size: 13px;")
            legend_layout.addWidget(label)
            legend_layout.addSpacing(25)
        
        legend_layout.addStretch()
        layout.addLayout(legend_layout)
        
        return panel
    
    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Server status
        self.server_status = QLabel("Server: Running")
        self.server_status.setStyleSheet("color: green; font-family: 'Times New Roman'; font-size: 11px;")
        self.status_bar.addPermanentWidget(self.server_status)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        self.status_bar.addPermanentWidget(separator)
        
        # Port information
        port_label = QLabel(f"Port: {self.server.port}")
        port_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 11px;")
        self.status_bar.addPermanentWidget(port_label)
    
    def _on_channel_changed(self, index: int):
        """通道选择改变事件"""
        self.current_channel = self.channel_combo.currentData()
        logger.info(f"切换到通道: {self.current_channel}")
        
        # 重新设置图表布局
        if self.current_channel == -1:
            self.canvas.setup_multi_channel(8)
        else:
            self.canvas.setup_single_channel()
        
        self._update_plot()
    
    def _on_steps_changed(self, index: int):
        """预测步数改变事件"""
        self.predict_steps = self.steps_combo.currentData()
        logger.info(f"预测步数改变为: {self.predict_steps}")
    
    def _on_clear_clicked(self):
        """清除数据按钮点击事件"""
        self.server.window.clear()
        self.last_prediction = None
        self.prediction_history.clear()
        self.canvas.clear_all()
        self._update_status_labels()
        logger.info("数据已清除")
    
    def _on_data_updated(self):
        """数据更新信号处理"""
        self._update_status_labels()
        # 不在这里更新图表，避免频繁更新导致阻塞
        # 图表会通过定时器或推理完成后更新
    
    def _on_inference_completed(self, result):
        """推理完成信号处理"""
        self.last_prediction = result
        self.prediction_history.append(result)
        self._update_status("推理完成")
        # 使用 QTimer 确保图表更新不阻塞信号处理
        # 即使图表更新很快，延迟调用也能确保事件循环正常运行
        QTimer.singleShot(0, self._update_plot)
    
    def _on_status_changed(self, status: str):
        """状态改变信号处理"""
        self._update_status(status)
    
    def _update_status(self, status: str):
        """更新状态显示"""
        self.status_label.setText(status)
        
        # 根据状态设置颜色
        if "完成" in status:
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif "等待" in status:
            self.status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        elif "推理中" in status:
            self.status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: #666666; font-weight: bold;")
    
    def _update_status_labels(self):
        """更新状态标签"""
        window = self.server.window
        
        # 窗口大小
        current_size = window.get_current_size()
        self.window_label.setText(f"{current_size} / {window.window_size}")
        
        # 接收计数
        self.receive_label.setText(str(self.server.receive_count))
        
        # 最新时间戳
        timestamp = window.get_latest_timestamp()
        if timestamp:
            self.timestamp_label.setText(timestamp)
        
        # 更新电压值
        values = window.get_latest_values()
        if values:
            for i, (label, value) in enumerate(zip(self.voltage_labels, values)):
                label.setText(f"{value:.4f} V")
        
        # Update status
        if window.is_ready():
            self._update_status("Data Ready")
        else:
            self._update_status(f"Waiting for data... ({current_size}/{window.window_size})")
    
    def _update_display(self):
        """定时器更新显示（已废弃，改用_update_from_csv）"""
        self._update_status_labels()
    
    def _update_from_csv(self):
        """从CSV文件读取数据并更新界面"""
        import csv
        import os
        
        try:
            # 读取接收到的数据
            if os.path.exists(self.csv_data_file):
                with open(self.csv_data_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    
                    # 跳过表头，获取新数据
                    data_rows = rows[1:]  # Skip header
                    current_row_count = len(data_rows)
                    
                    if current_row_count > self.last_data_row_count:
                        # 有新数据
                        self.last_data_row_count = current_row_count
                        
                        # 更新状态标签
                        self.receive_label.setText(str(current_row_count))
                        
                        if data_rows:
                            # 获取最新数据
                            latest_row = data_rows[-1]
                            timestamp = latest_row[0]
                            values = [float(v) for v in latest_row[1:]]
                            
                            self.timestamp_label.setText(timestamp)
                            for i, (label, value) in enumerate(zip(self.voltage_labels, values)):
                                label.setText(f"{value:.4f} V")
                            
                            # 更新窗口大小显示
                            window_size = min(current_row_count, 60)
                            self.window_label.setText(f"{window_size} / 60")
                            
                            if window_size >= 60:
                                self._update_status("Data Ready")
                            else:
                                self._update_status(f"Waiting for data... ({window_size}/60)")
            
            # 读取预测结果
            if os.path.exists(self.csv_prediction_file):
                with open(self.csv_prediction_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    
                    # 跳过表头
                    pred_rows = rows[1:]  # Skip header
                    current_pred_count = len(pred_rows)
                    
                    if current_pred_count > self.last_prediction_row_count:
                        # 有新的预测
                        self.last_prediction_row_count = current_pred_count
                        
                        # 读取所有预测（按timestamp分组）
                        if pred_rows:
                            # 按timestamp分组所有预测
                            from collections import defaultdict
                            grouped_predictions = defaultdict(list)
                            
                            for row in pred_rows:
                                # row格式: [timestamp, step, ch1, ch2, ..., ch8]
                                timestamp = row[0]
                                pred_values = [float(v) for v in row[2:]]
                                grouped_predictions[timestamp].append(pred_values)
                            
                            # 将每个timestamp的预测转换为PredictionResult对象
                            from inference_engine import PredictionResult
                            new_predictions = []
                            
                            for timestamp in sorted(grouped_predictions.keys()):
                                predictions_list = grouped_predictions[timestamp]
                                predictions_array = np.array(predictions_list)
                                result = PredictionResult(
                                    predictions=predictions_array,
                                    steps=len(predictions_list),
                                    input_seq_len=60,
                                    timestamp=timestamp
                                )
                                new_predictions.append(result)
                            
                            # 更新prediction_history（保留所有预测）
                            # 注意：移除maxlen限制，保留所有历史预测
                            if not hasattr(self, 'all_predictions'):
                                self.all_predictions = []
                            
                            # 只添加新的预测（避免重复）
                            existing_timestamps = {p.timestamp for p in self.all_predictions}
                            for pred in new_predictions:
                                if pred.timestamp not in existing_timestamps:
                                    self.all_predictions.append(pred)
                            
                            if new_predictions:
                                self.last_prediction = new_predictions[-1]
                                self._update_status("Inference Completed")
                                
                                # 更新图表
                                self._update_plot()
        
        except Exception as e:
            logger.error(f"从CSV读取数据时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_inference(self):
        """执行推理（在后台线程中调用）"""
        try:
            self.signals.status_changed.emit("推理中...")
            
            # 获取窗口数据
            data = self.server.window.get_data()
            if data is None:
                return
            
            # 获取时间戳
            timestamp = self.server.window.get_latest_timestamp() or ""
            
            # 执行推理
            result = self.engine.predict(data, steps=self.predict_steps, timestamp=timestamp)
            
            # 保存预测结果到CSV（替代原来的GUI更新）
            self.server.save_prediction_to_csv(timestamp, result.predictions, self.predict_steps)
            
            logger.info(f"推理完成: {self.predict_steps} 步预测，已写入CSV")
            
        except Exception as e:
            logger.error(f"推理出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_plot(self):
        """更新图表（从CSV读取的数据）"""
        try:
            import csv
            import os
            
            # 从CSV读取所有数据点（包含timestamps）
            data = None
            timestamps = []
            all_data_rows = []
            
            if os.path.exists(self.csv_data_file):
                with open(self.csv_data_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    data_rows = rows[1:]  # Skip header
                    
                    if len(data_rows) > 0:
                        # 保存所有数据行用于创建timestamp映射
                        all_data_rows = data_rows
                        timestamps = [row[0] for row in data_rows]
                        
                        # 只显示最近60个数据点
                        if len(data_rows) >= 60:
                            recent_rows = data_rows[-60:]
                            data = np.array([[float(v) for v in row[1:]] for row in recent_rows])
                            # 调整timestamps也只取最近60个
                            display_timestamps = timestamps[-60:]
                        else:
                            data = np.array([[float(v) for v in row[1:]] for row in data_rows])
                            display_timestamps = timestamps
            
            # 创建timestamp到索引的映射（用于所有数据）
            timestamp_to_index = {ts: idx for idx, ts in enumerate(timestamps)}
            
            # 计算显示窗口的起始索引
            if len(timestamps) >= 60:
                display_start_index = len(timestamps) - 60
            else:
                display_start_index = 0
            
            if self.current_channel == -1:
                # 全通道显示
                self._plot_all_channels(data, timestamp_to_index, display_start_index)
            else:
                # 单通道显示
                self._plot_single_channel(data, self.current_channel, timestamp_to_index, display_start_index)
            
            # 使用 draw_idle 替代 draw，避免阻塞主线程
            self.canvas.draw_idle()
            
        except Exception as e:
            logger.error(f"更新图表出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_all_channels(self, data: Optional[np.ndarray], timestamp_to_index: dict = None, display_start_index: int = 0):
        """
        绘制所有通道
        
        参数:
            data: np.ndarray or None, 窗口数据
            timestamp_to_index: dict, timestamp到全局索引的映射
            display_start_index: int, 显示窗口在全局数据中的起始索引
        """
        # 确保有8个子图
        if len(self.canvas.axes) != 8:
            self.canvas.setup_multi_channel(8)
        
        for i, ax in enumerate(self.canvas.axes):
            ax.clear()
            
            if data is not None and len(data) > 0:
                # Plot historical data
                time_axis = np.arange(display_start_index, display_start_index + len(data))
                ax.plot(time_axis, data[:, i], 
                       color=self.CHANNEL_COLORS[i], 
                       linewidth=1.5, 
                       label='History')
                
                # Plot all predictions (not just the last one)
                if hasattr(self, 'all_predictions') and len(self.all_predictions) > 0 and timestamp_to_index is not None:
                    for pred_result in self.all_predictions:
                        # 根据预测的timestamp找到对应的数据索引
                        pred_timestamp = pred_result.timestamp
                        
                        if pred_timestamp in timestamp_to_index:
                            # 找到预测是基于哪个时间点做出的
                            pred_base_index = timestamp_to_index[pred_timestamp]
                            
                            # 预测从这个时间点之后开始
                            pred_data = pred_result.predictions
                            pred_time = np.arange(pred_base_index + 1, pred_base_index + 1 + len(pred_data))
                            
                            ax.plot(pred_time, pred_data[:, i],
                                   color='#FF5722',
                                   linewidth=1.5,
                                   linestyle='--',
                                   marker='o',
                                   markersize=2,
                                   alpha=0.6)  # 使用透明度以区分多个预测
                
                # 单独标记最新的预测（更明显）
                if self.last_prediction is not None and timestamp_to_index is not None:
                    pred_timestamp = self.last_prediction.timestamp
                    if pred_timestamp in timestamp_to_index:
                        pred_base_index = timestamp_to_index[pred_timestamp]
                        pred_data = self.last_prediction.predictions
                        pred_time = np.arange(pred_base_index + 1, pred_base_index + 1 + len(pred_data))
                        
                        ax.plot(pred_time, pred_data[:, i],
                               color='#FF5722',
                               linewidth=2,
                               linestyle='--',
                               marker='o',
                               markersize=3,
                               label='Prediction')
            
            # Set title and labels with Times New Roman font
            ax.set_title(f'{self.CHANNEL_NAMES[i]}', fontproperties=T_12, fontweight='bold')
            ax.set_xlabel('Time Step', fontproperties=T_10)
            ax.set_ylabel('Voltage (V)', fontproperties=T_10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', labelsize=9)
            
            # Set tick label font
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(T_9)
            
            if data is not None and len(data) > 0:
                ax.legend(loc='upper left', prop=T_8)
        
        # 不再调用 tight_layout，避免阻塞主线程
        # tight_layout 仅在初始化时（setup_multi_channel/setup_single_channel）调用一次
    
    def _plot_single_channel(self, data: Optional[np.ndarray], channel: int, timestamp_to_index: dict = None, display_start_index: int = 0):
        """
        绘制单个通道
        
        参数:
            data: np.ndarray or None, 窗口数据
            channel: int, 通道索引
            timestamp_to_index: dict, timestamp到全局索引的映射
            display_start_index: int, 显示窗口在全局数据中的起始索引
        """
        # 确保只有1个子图
        if len(self.canvas.axes) != 1:
            self.canvas.setup_single_channel()
        
        ax = self.canvas.axes[0]
        ax.clear()
        
        if data is not None and len(data) > 0:
            # Plot historical data
            time_axis = np.arange(display_start_index, display_start_index + len(data))
            ax.plot(time_axis, data[:, channel],
                   color=self.CHANNEL_COLORS[channel],
                   linewidth=2,
                   label='History')
            
            # Plot all predictions (not just the last one)
            if hasattr(self, 'all_predictions') and len(self.all_predictions) > 0 and timestamp_to_index is not None:
                for pred_result in self.all_predictions:
                    pred_timestamp = pred_result.timestamp
                    
                    if pred_timestamp in timestamp_to_index:
                        pred_base_index = timestamp_to_index[pred_timestamp]
                        pred_data = pred_result.predictions
                        pred_time = np.arange(pred_base_index + 1, pred_base_index + 1 + len(pred_data))
                        
                        ax.plot(pred_time, pred_data[:, channel],
                               color='#FF5722',
                               linewidth=2,
                               linestyle='--',
                               marker='o',
                               markersize=4,
                               alpha=0.5)  # 使用透明度以区分多个预测
            
            # Plot the latest prediction more prominently
            if self.last_prediction is not None and timestamp_to_index is not None:
                pred_timestamp = self.last_prediction.timestamp
                if pred_timestamp in timestamp_to_index:
                    pred_base_index = timestamp_to_index[pred_timestamp]
                    pred_data = self.last_prediction.predictions
                    pred_time = np.arange(pred_base_index + 1, pred_base_index + 1 + len(pred_data))
                    
                    ax.plot(pred_time, pred_data[:, channel],
                           color='#FF5722',
                           linewidth=2.5,
                           linestyle='--',
                           marker='o',
                           markersize=5,
                           label=f'Prediction ({self.predict_steps} steps)')
                    
                    # Connect historical data and prediction
                    if len(time_axis) > 0 and len(pred_time) > 0:
                        # Find correct index for connection point
                        if pred_base_index >= display_start_index and pred_base_index < display_start_index + len(data):
                            local_idx = pred_base_index - display_start_index
                            ax.plot([time_axis[local_idx], pred_time[0]],
                                   [data[local_idx, channel], pred_data[0, channel]],
                                   color='#FF5722',
                                   linewidth=1,
                                   linestyle=':',
                                   alpha=0.5)
        
        # Set title and labels with Times New Roman font
        ax.set_title(f'{self.CHANNEL_NAMES[channel]} Channel Voltage', fontproperties=T_16, fontweight='bold')
        ax.set_xlabel('Time Step (10s/step)', fontproperties=T_12)
        ax.set_ylabel('Voltage (V)', fontproperties=T_12)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Set tick label font
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(T_10)
        
        ax.legend(loc='upper left', prop=T_10)
        
        # Comment: tight_layout is not called here to avoid blocking main thread
        # tight_layout is only called once during initialization
    
    def closeEvent(self, event):
        """Window close event"""
        logger.info("Closing application...")
        self.update_timer.stop()
        event.accept()


def main():
    """主函数"""
    # [新增] 再次确保限制 PyTorch 线程数
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description='热电芯片电压实时监测与预测系统')
    
    parser.add_argument('--port', type=int, default=5000,
                       help='HTTP 服务端口 (默认: 5000)')
    parser.add_argument('--window-size', type=int, default=60,
                       help='滑动窗口大小 (默认: 60)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='模型检查点路径')
    parser.add_argument('--csv-dir', type=str, default='./realtime_data',
                       help='CSV数据输出目录 (默认: ./realtime_data)')
    
    args = parser.parse_args()
    
    # 创建推理引擎
    logger.info("=" * 60)
    logger.info("热电芯片电压实时监测与预测系统 - CSV解耦架构")
    logger.info("=" * 60)
    
    engine = create_inference_engine(args.model_path)
    
    # 创建数据服务器
    server = DataServer(
        port=args.port,
        window_size=args.window_size,
        csv_output_dir=args.csv_dir
    )
    
    # 创建 Qt 应用
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = MainWindow(server, engine)
    
    # 设置推理回调（在窗口创建后设置）
    server.set_inference_callback(window._run_inference)
    
    # 在后台线程启动服务器
    server.run_in_thread()
    
    window.show()
    
    logger.info("=" * 60)
    logger.info("GUI 已启动 - CSV解耦模式")
    logger.info(f"HTTP 服务: http://0.0.0.0:{args.port}")
    logger.info(f"数据文件: {server.csv_file}")
    logger.info(f"预测文件: {server.prediction_file}")
    logger.info("等待 Raspberry Pi 发送数据...")
    logger.info("=" * 60)
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
