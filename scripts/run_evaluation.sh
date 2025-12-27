#!/bin/bash
################################################################################
# 模型评估脚本
# 评估所有训练好的模型（RNN、Transformer、T5）并生成结果
################################################################################

set -e

# 激活conda环境
echo "激活conda环境..."
source /mnt/afs/250010036/miniconda3/bin/activate nlp

# 进入项目目录
cd /mnt/afs/250010036/course/NLP

echo ""
echo "========================================================================"
echo "开始评估所有模型"
echo "========================================================================"
echo ""

# 运行评估
python src/evaluation.py \
    --data_dir /mnt/afs/250010036/course/NLP/data \
    --experiments_dir /mnt/afs/250010036/course/NLP/experiments \
    --results_dir /mnt/afs/250010036/course/NLP/results \
    --device cuda

echo ""
echo "========================================================================"
echo "评估完成！"
echo "========================================================================"
echo ""
echo "结果文件保存在: /mnt/afs/250010036/course/NLP/results/"
echo ""
echo "可以查看:"
echo "  - results/rnn_en2zh_results.json"
echo "  - results/rnn_zh2en_results.json"
echo "  - results/transformer_en2zh_results.json"
echo "  - results/transformer_zh2en_results.json"
echo "  - results/t5_en2zh_results.json"
echo "  - results/t5_zh2en_results.json"
echo "  - results/evaluation_summary.json (汇总)"
echo ""

