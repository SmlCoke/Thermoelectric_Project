"""
训练脚本

该模块负责：
1. 加载数据
2. 创建和配置模型（LSTM或GRU）
3. 训练模型
4. 验证和评估
5. 保存模型和训练日志
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 导入自定义模块
from dataset import create_dataloaders
from model_lstm import LSTMModel
from model_gru import GRUModel


class Trainer:
    """训练器类，封装训练逻辑"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        """
        初始化训练器
        
        参数:
            model: nn.Module, LSTM或GRU模型
            train_loader: DataLoader, 训练数据加载器
            val_loader: DataLoader, 验证数据加载器
            device: torch.device, 设备
            config: dict, 配置参数
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 损失函数
        self.criterion = nn.MSELoss()  # 均方误差损失
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # TensorBoard日志
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        参数:
            epoch: int, 当前epoch编号
        
        返回:
            avg_loss: float, 平均训练损失
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            # 数据移到设备
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            predictions, _ = self.model(batch_x)
            
            # 计算损失
            loss = self.criterion(predictions, batch_y)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch}/{self.config["num_epochs"]}], '
                      f'Batch [{batch_idx}/{len(self.train_loader)}], '
                      f'Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """
        验证模型
        
        返回:
            avg_loss: float, 平均验证损失
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                # 数据移到设备
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向传播
                predictions, _ = self.model(batch_x)
                
                # 计算损失
                loss = self.criterion(predictions, batch_y)
                
                # 累计损失
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """
        完整的训练流程
        """
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            epoch_start = time.time()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # TensorBoard日志
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印进度
            epoch_time = time.time() - epoch_start
            print(f'\nEpoch [{epoch}/{self.config["num_epochs"]}] '
                  f'完成 (耗时: {epoch_time:.2f}s)')
            print(f'  训练损失: {train_loss:.6f}')
            print(f'  验证损失: {val_loss:.6f}')
            print(f'  学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint('best_model.pth', epoch, val_loss)
                print(f'  *** 新的最佳模型! 验证损失: {val_loss:.6f} ***')
            else:
                self.epochs_no_improve += 1
            
            # Early stopping
            if self.epochs_no_improve >= self.config['early_stopping_patience']:
                print(f'\nEarly stopping: {self.config["early_stopping_patience"]} '
                      f'个epoch验证损失未改善')
                break
            
            # 定期保存检查点
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, val_loss)
            
            print("-" * 60)
        
        # 训练完成
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"训练完成! 总耗时: {total_time / 60:.2f} 分钟")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print("=" * 60)
        
        # 保存最终模型
        self.save_checkpoint('final_model.pth', 
                           self.config['num_epochs'], 
                           self.history['val_loss'][-1])
        
        # 保存训练历史
        self.save_history()
        
        self.writer.close()
    
    def save_checkpoint(self, filename, epoch, val_loss):
        """
        保存模型检查点
        
        参数:
            filename: str, 文件名
            epoch: int, epoch编号
            val_loss: float, 验证损失
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'history': self.history
        }
        
        filepath = os.path.join(self.config['save_dir'], filename)
        torch.save(checkpoint, filepath)
        print(f'  检查点已保存: {filepath}')
    
    def save_history(self):
        """保存训练历史"""
        filepath = os.path.join(self.config['save_dir'], 'training_history.npy')
        np.save(filepath, self.history)
        print(f'训练历史已保存: {filepath}')


def main(args):
    """主函数"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 配置参数
    config = {
        # 数据参数
        'data_dir': args.data_dir,
        'window_size': args.window_size,
        'predict_steps': args.predict_steps,
        'batch_size': args.batch_size,
        'stride': args.stride,
        
        # 模型参数
        'model_type': args.model,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        
        # 训练参数
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        
        # 保存参数
        'save_dir': args.save_dir,
        'log_dir': args.log_dir,
        'save_interval': args.save_interval
    }
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, dataset = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        window_size=config['window_size'],
        predict_steps=config['predict_steps'],
        stride=config['stride'],
        normalize=True,
        train_ratio=0.8,
        num_workers=4
    )
    
    # 保存标准化器
    scaler_path = os.path.join(config['save_dir'], 'scaler.pkl')
    dataset.save_scaler(scaler_path)
    
    # 创建模型
    print(f"\n创建 {config['model_type'].upper()} 模型...")
    if config['model_type'] == 'lstm':
        model = LSTMModel(
            input_size=8,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=8,
            predict_steps=config['predict_steps'],
            dropout=config['dropout']
        )
    elif config['model_type'] == 'gru':
        model = GRUModel(
            input_size=8,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=8,
            predict_steps=config['predict_steps'],
            dropout=config['dropout']
        )
    else:
        raise ValueError(f"不支持的模型类型: {config['model_type']}")
    
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, device, config)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练LSTM/GRU时间序列预测模型')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, 
                       default='../TimeSeries',
                       help='CSV文件所在目录')
    parser.add_argument('--window_size', type=int, default=60,
                       help='输入序列长度')
    parser.add_argument('--predict_steps', type=int, default=10,
                       help='预测步数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--stride', type=int, default=5,
                       help='滑动窗口步长')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='gru', 
                       choices=['lstm', 'gru'],
                       help='模型类型: lstm 或 gru')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='RNN层数')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout比率')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping耐心值')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志保存目录')
    parser.add_argument('--save_interval', type=int, default=20,
                       help='保存检查点的间隔')
    
    args = parser.parse_args()
    main(args)
