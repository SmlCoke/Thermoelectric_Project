"""
训练脚本 - 完整版本
包含混合精度训练、多GPU支持、高级学习率调度等
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import time
from tqdm import tqdm

from config import Config
from model import AdvancedCNN_LSTM_Forecaster
from dataset import load_and_preprocess_data

def set_seed(seed):
    """设置随机种子以保证结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_scheduler(optimizer, config, num_training_steps):
    """
    创建学习率调度器
    """
    if not config.USE_SCHEDULER:
        return None
    
    if config.SCHEDULER_TYPE == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    elif config.SCHEDULER_TYPE == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=config.MIN_LR
        )
    elif config.SCHEDULER_TYPE == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.5
        )
    else:
        scheduler = None
    
    return scheduler

def train_epoch(model, train_loader, criterion, optimizer, device, config, scaler=None):
    """
    训练一个epoch（支持混合精度训练）
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='训练中')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # 移动数据到设备
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        if config.USE_MIXED_PRECISION and scaler is not None:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # 反向传播（混合精度）
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP)
            
            optimizer.step()
        
        # 统计
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, val_loader, criterion, device):
    """
    验证模型
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

def train(config):
    """
    完整训练流程（服务器版本）
    """
    # 设置随机种子
    set_seed(config.SEED)
    
    # 打印配置
    config.print_config()
    
    # 创建必要的目录
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # 加载数据
    print("\n" + "=" * 60)
    print("加载数据...")
    print("=" * 60)
    train_loader, val_loader, test_loader, scaler = load_and_preprocess_data(config)
    
    # 创建模型
    print("\n" + "=" * 60)
    print("创建模型...")
    print("=" * 60)
    model = AdvancedCNN_LSTM_Forecaster(config)
    
    # 多GPU支持
    if config.MULTI_GPU:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = nn.DataParallel(model)
    
    model = model.to(config.DEVICE)
    
    # 获取实际模型（处理DataParallel包装）
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    print(f"模型参数量: {actual_model.count_parameters():,}")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 学习率调度器
    num_training_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = get_scheduler(optimizer, config, num_training_steps)
    
    # 混合精度训练的Scaler
    scaler = GradScaler() if config.USE_MIXED_PRECISION else None
    
    # TensorBoard
    writer = SummaryWriter(config.LOG_DIR)
    
    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15  # 服务器版本有更多耐心
    
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 60)
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                config.DEVICE, config, scaler)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, config.DEVICE)
        
        # 更新学习率
        if scheduler is not None:
            if config.SCHEDULER_TYPE == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # 打印结果
        print(f"训练损失: {train_loss:.6f}")
        print(f"验证损失: {val_loss:.6f}")
        print(f"学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存模型（处理DataParallel）
            model_to_save = actual_model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, config.MODEL_SAVE_PATH)
            
            print(f"✓ 保存最佳模型 (验证损失: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"验证损失未改善 ({patience_counter}/{early_stop_patience})")
        
        # 定期保存检查点
        if (epoch + 1) % config.SAVE_FREQ == 0:
            checkpoint_path = config.MODEL_SAVE_PATH.replace('.pth', f'_epoch{epoch+1}.pth')
            model_to_save = actual_model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✓ 保存检查点: {checkpoint_path}")
        
        # 早停
        if patience_counter >= early_stop_patience:
            print(f"\n早停触发！连续 {early_stop_patience} 个epoch验证损失未改善")
            break
    
    # 训练结束
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"总训练时间: {total_time / 60:.2f} 分钟 ({total_time / 3600:.2f} 小时)")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型已保存到: {config.MODEL_SAVE_PATH}")
    
    writer.close()
    
    # 在测试集上评估
    print("\n" + "=" * 60)
    print("在测试集上评估...")
    print("=" * 60)
    
    # 加载最佳模型
    checkpoint = torch.load(config.MODEL_SAVE_PATH)
    actual_model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss = validate(model, test_loader, criterion, config.DEVICE)
    print(f"测试损失 (MSE): {test_loss:.6f}")
    print(f"测试损失 (RMSE): {np.sqrt(test_loss):.6f}")
    
    print("\n接下来可以运行:")
    print("  python test.py  # 查看预测结果可视化")
    print("  tensorboard --logdir=logs/  # 查看训练曲线")

if __name__ == "__main__":
    config = Config()
    train(config)
