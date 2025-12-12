#!/bin/bash
# 数据增强快速开始脚本
# 使用推荐配置 (N=1,2,3,5) 处理Prac_data目录中的所有CSV文件

echo "======================================================================"
echo "时序数据降采样 - 快速开始"
echo "======================================================================"
echo ""

# 检查Prac_data目录是否存在
if [ ! -d "../Prac_data" ]; then
    echo "错误: ../Prac_data 目录不存在"
    echo "请确保您的实验数据放在 TimeSeries/Prac_data/ 目录中"
    exit 1
fi

# 统计CSV文件数量
csv_count=$(find ../Prac_data -name "*.csv" -type f | wc -l)
if [ $csv_count -eq 0 ]; then
    echo "错误: ../Prac_data 目录中没有CSV文件"
    echo "请将您的实验数据CSV文件放入该目录"
    exit 1
fi

echo "找到 $csv_count 个CSV文件"
echo ""

# 设置输出目录
OUTPUT_DIR="./augmented_data"

echo "配置:"
echo "  输入目录: ../Prac_data"
echo "  输出目录: $OUTPUT_DIR"
echo "  降采样率: 1, 2, 3, 5 (推荐)"
echo "  最小样本数: 100"
echo ""

read -p "按回车键开始处理，或按 Ctrl+C 取消..." 

echo ""
echo "开始处理..."
echo ""

# 运行降采样脚本
python3 subsample_data.py \
    -d ../Prac_data \
    -o "$OUTPUT_DIR" \
    -r 1 2 3 5 \
    -m 100

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ 处理完成！"
    echo "======================================================================"
    echo ""
    echo "生成的数据文件位于: $OUTPUT_DIR"
    echo ""
    echo "下一步:"
    echo "  1. 查看生成的文件:"
    echo "     ls -lh $OUTPUT_DIR"
    echo ""
    echo "  2. 开始训练模型:"
    echo "     cd ../src"
    echo "     python train.py --model gru --hidden_size 128 --num_epochs 100"
    echo ""
    echo "  3. 查看训练日志:"
    echo "     tensorboard --logdir=../logs"
    echo ""
else
    echo ""
    echo "处理失败，请检查错误信息"
    exit 1
fi
