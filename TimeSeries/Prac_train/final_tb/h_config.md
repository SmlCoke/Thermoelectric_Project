## 步骤
请严格按照如下步骤执行指令
### 1. 拉取仓库最新更新
```bash
git pull origin main
```

### 2. 训练

- sub6：降采样因子为6数据集训练命令
```bash
python train.py --model lstm --hidden_size 256 --num_layers 2 --num_epochs 100 --data_dir ../Prac_data/TrainSet/final_tb/sub6 --window_size 60 --predict_steps 10 --batch_size 32 --stride 1 --save_dir ../Prac_train/final_tb/sub6/checkpoints --log_dir  ../Prac_train/final_tb/sub6/logs 
```
训练结束之后，在 `TimeSeries/Prac_train/final_tb/sub6` 中存放着模型权重文件，将这个文件夹发给我。

- d1_sub6：降噪阈值1.75然后执行降采样因子为6数据集训练命令
```bash
python train.py --model lstm --hidden_size 256 --num_layers 2 --num_epochs 100 --data_dir ../Prac_data/TrainSet/final_tb/d1_sub6 --window_size 60 --predict_steps 10 --batch_size 32 --stride 1 --save_dir ../Prac_train/final_tb/d1_sub6/checkpoints --log_dir  ../Prac_train/final_tb/d1_sub6/logs 
```
训练结束之后，在 `TimeSeries/Prac_train/final_tb/d1_sub6` 中存放着模型权重文件，将这个文件夹发给我。
