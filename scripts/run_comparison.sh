#!/bin/bash
# ============================================================
# 模型对比和可视化
# ============================================================

set -e

# 配置
PROJECT_DIR="/mnt/afs/250010036/course/NLP"

# 激活环境
echo "============================================================"
echo "生成模型对比报告和可视化图表"
echo "============================================================"
source /mnt/afs/250010036/miniconda3/bin/activate nlp
cd ${PROJECT_DIR}

# 创建结果目录
mkdir -p results/figures

# 模型对比
echo ""
echo "--- 生成模型对比报告 ---"
if [ -d "results" ] && [ "$(ls -A results/*.json 2>/dev/null)" ]; then
    python scripts/compare_models.py \
        --results_dir results/ \
        --output_file results/model_comparison.json
    echo "✓ 对比报告生成完成: results/model_comparison.json"
else
    echo "⚠ 评估结果文件不存在，请先运行 run_evaluation.sh"
fi

# 生成可视化图表
echo ""
echo "--- 生成可视化图表 ---"
if [ -d "experiments" ]; then
    python src/visualize.py \
        --experiments_dir experiments/ \
        --results_dir results/ \
        --output_dir results/figures/
    echo "✓ 可视化图表生成完成: results/figures/"
else
    echo "⚠ 实验目录不存在，跳过可视化"
fi

echo ""
echo "============================================================"
echo "✓ 对比和可视化完成！"
echo "============================================================"
echo ""
echo "结果文件:"
echo "  - 对比报告: results/model_comparison.json"
echo "  - 可视化图表: results/figures/"
echo ""


