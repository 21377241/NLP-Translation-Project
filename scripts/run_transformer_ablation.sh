#!/bin/bash
# Transformer 消融实验与超参数敏感性分析运行脚本
# 
# 作业要求：
# 1. 架构消融研究：位置编码方案对比、归一化方法对比
# 2. 超参数敏感性分析：批次大小、学习率、模型规模

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
EPOCHS=10
USE_SMALL="--use_small"
DATA_DIR="AP0004_Midterm&Final_translation_dataset_zh_en"
OUTPUT_BASE="experiments/transformer_ablation"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --large)
            USE_SMALL=""
            shift
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项] <实验类型>"
            echo ""
            echo "实验类型:"
            echo "  position_encoding  - 位置编码消融实验（绝对/可学习/相对）"
            echo "  normalization      - 归一化方法消融实验（LayerNorm/RMSNorm）"
            echo "  architecture       - 完整架构消融实验"
            echo "  batch_size         - 批次大小敏感性分析"
            echo "  learning_rate      - 学习率敏感性分析"
            echo "  model_scale        - 模型规模敏感性分析"
            echo "  hyperparameter     - 完整超参数敏感性分析"
            echo "  full               - 运行所有实验"
            echo ""
            echo "选项:"
            echo "  --epochs N         - 训练轮数（默认: 10）"
            echo "  --large            - 使用大数据集（100k）"
            echo "  --output DIR       - 输出目录"
            echo "  --help             - 显示此帮助信息"
            exit 0
            ;;
        *)
            EXPERIMENT_TYPE="$1"
            shift
            ;;
    esac
done

# 检查实验类型
if [ -z "$EXPERIMENT_TYPE" ]; then
    echo -e "${YELLOW}请选择实验类型:${NC}"
    echo ""
    echo "  1) position_encoding  - 位置编码消融实验"
    echo "  2) normalization      - 归一化方法消融实验"
    echo "  3) architecture       - 完整架构消融实验"
    echo "  4) batch_size         - 批次大小敏感性分析"
    echo "  5) learning_rate      - 学习率敏感性分析"
    echo "  6) model_scale        - 模型规模敏感性分析"
    echo "  7) hyperparameter     - 完整超参数敏感性分析"
    echo "  8) full               - 运行所有实验"
    echo ""
    read -p "请输入选择 (1-8): " choice
    
    case $choice in
        1) EXPERIMENT_TYPE="position_encoding" ;;
        2) EXPERIMENT_TYPE="normalization" ;;
        3) EXPERIMENT_TYPE="architecture" ;;
        4) EXPERIMENT_TYPE="batch_size" ;;
        5) EXPERIMENT_TYPE="learning_rate" ;;
        6) EXPERIMENT_TYPE="model_scale" ;;
        7) EXPERIMENT_TYPE="hyperparameter" ;;
        8) EXPERIMENT_TYPE="full" ;;
        *) echo -e "${RED}无效选择${NC}"; exit 1 ;;
    esac
fi

# 显示实验信息
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Transformer 消融实验与超参数敏感性分析${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${GREEN}实验类型:${NC} $EXPERIMENT_TYPE"
echo -e "${GREEN}训练轮数:${NC} $EPOCHS"
echo -e "${GREEN}数据集:${NC} $([ -z "$USE_SMALL" ] && echo "100k" || echo "10k")"
echo -e "${GREEN}输出目录:${NC} $OUTPUT_BASE"
echo ""

# 确认运行
read -p "是否开始运行? (y/n): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "已取消"
    exit 0
fi

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

# 运行实验
echo ""
echo -e "${GREEN}开始运行实验...${NC}"
echo ""

cd "$(dirname "$0")"

python -m src.train_transformer_ablation \
    --experiment_type "$EXPERIMENT_TYPE" \
    --epochs "$EPOCHS" \
    $USE_SMALL \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_BASE" \
    2>&1 | tee "$OUTPUT_BASE/${EXPERIMENT_TYPE}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo -e "${GREEN}实验完成!${NC}"
echo -e "${GREEN}结果保存在: $OUTPUT_BASE${NC}"

