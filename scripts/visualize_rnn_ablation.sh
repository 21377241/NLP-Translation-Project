#!/bin/bash
# ============================================================
# RNN消融实验可视化脚本
# ============================================================

set -e

PROJECT_DIR="/mnt/afs/250010036/course/NLP"
EXPERIMENT_DIR="${PROJECT_DIR}/experiments/rnn_ablation"
OUTPUT_DIR="${PROJECT_DIR}/results/rnn_ablation_visualizations"

# 激活环境
source /mnt/afs/250010036/miniconda3/bin/activate nlp
cd ${PROJECT_DIR}

echo "============================================================"
echo "RNN消融实验可视化"
echo "============================================================"

# 可视化所有实验
python src/visualize_rnn_ablation.py \
    --mode all \
    --experiment_dir "${EXPERIMENT_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "✓ 可视化完成！"
echo "============================================================"
echo "可视化结果位置: ${OUTPUT_DIR}"

