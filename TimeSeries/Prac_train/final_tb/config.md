## config

- d1: 降噪阈值1.75
- sub6: 降采样因子为6
- raw: 使用原始数据
- d1_sub6: 降噪阈值1.75然后执行降采样因子为6的下采样

### 命令
- raw：原始数据集训练命令
```bash
python train.py --model lstm --hidden_size 256 --num_layers 2 --num_epochs 100 --data_dir ../Prac_data/TrainSet/final_tb/raw --window_size 60 --predict_steps 10 --batch_size 32 --stride 1 --save_dir ../Prac_train/final_tb/raw/checkpoints --log_dir  ../Prac_train/final_tb/raw/logs 
```

- d1：降噪阈值1.75数据集训练命令
```bash
python train.py --model lstm --hidden_size 256 --num_layers 2 --num_epochs 100 --data_dir ../Prac_data/TrainSet/final_tb/d1 --window_size 60 --predict_steps 10 --batch_size 32 --stride 1 --save_dir ../Prac_train/final_tb/d1/checkpoints --log_dir  ../Prac_train/final_tb/d1/logs 
```

- sub6：降采样因子为6数据集训练命令
```bash
python train.py --model lstm --hidden_size 256 --num_layers 2 --num_epochs 100 --data_dir ../Prac_data/TrainSet/final_tb/sub6 --window_size 60 --predict_steps 10 --batch_size 32 --stride 1 --save_dir ../Prac_train/final_tb/sub6/checkpoints --log_dir  ../Prac_train/final_tb/sub6/logs 
```

- d1_sub6：降噪阈值1.75然后执行降采样因子为6数据集训练命令
```bash
python train.py --model lstm --hidden_size 256 --num_layers 2 --num_epochs 100 --data_dir ../Prac_data/TrainSet/final_tb/d1_sub6 --window_size 60 --predict_steps 10 --batch_size 32 --stride 1 --save_dir ../Prac_train/final_tb/d1_sub6/checkpoints --log_dir  ../Prac_train/final_tb/d1_sub6/logs 
```