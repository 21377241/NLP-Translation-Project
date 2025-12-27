#!/bin/bash
# ============================================================
# RNN Seq2Seq 消融实验运行脚本
# 
# 实验内容：
# 1. 注意力机制对比：点积 vs 乘性 vs 加性
# 2. 训练策略对比：Teacher Forcing vs Scheduled Sampling vs Free Running
# 3. 解码策略对比：贪婪解码 vs 束搜索
#
# 作者: NLP课程项目
# ============================================================

set -e

# 配置
PROJECT_DIR="/mnt/afs/250010036/course/NLP"
DATA_DIR="${PROJECT_DIR}/AP0004_Midterm&Final_translation_dataset_zh_en"
VOCAB_DIR="${PROJECT_DIR}/data/vocab"
OUTPUT_DIR="${PROJECT_DIR}/experiments/rnn_ablation"

# 默认参数
EXPERIMENT_TYPE="attention"    # attention, training_strategy, full
SRC_LANG="en"
TGT_LANG="zh"
EPOCHS=15
BATCH_SIZE=64
HIDDEN_DIM=512
EMBED_DIM=256
LEARNING_RATE=0.001
USE_SMALL="--use_small"        # 使用10k数据集

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_type)
            EXPERIMENT_TYPE="$2"
            shift 2
            ;;
        --src_lang)
            SRC_LANG="$2"
            shift 2
            ;;
        --tgt_lang)
            TGT_LANG="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --use_large)
            USE_SMALL=""
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --experiment_type TYPE    实验类型: attention, training_strategy, full (默认: attention)"
            echo "  --src_lang LANG          源语言 (默认: en)"
            echo "  --tgt_lang LANG          目标语言 (默认: zh)"
            echo "  --epochs N               训练轮数 (默认: 15)"
            echo "  --batch_size N           批次大小 (默认: 64)"
            echo "  --use_large              使用大数据集 (100k)"
            echo "  --help, -h               显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 翻译方向设置
DIRECTION="${SRC_LANG}2${TGT_LANG}"
OUTPUT_SUBDIR="${OUTPUT_DIR}/${DIRECTION}"

# 激活环境
echo "============================================================"
echo "RNN Seq2Seq 消融实验"
echo "============================================================"
echo "实验类型: ${EXPERIMENT_TYPE}"
echo "翻译方向: ${SRC_LANG} -> ${TGT_LANG}"
echo "训练轮数: ${EPOCHS}"
echo "批次大小: ${BATCH_SIZE}"
echo "输出目录: ${OUTPUT_SUBDIR}"
echo "============================================================"

source /mnt/afs/250010036/miniconda3/bin/activate nlp
cd ${PROJECT_DIR}

# 检查GPU
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 运行消融实验训练
echo ""
echo "============================================================"
echo "开始消融实验训练..."
echo "============================================================"

python src/train_rnn_ablation.py \
    --experiment_type "${EXPERIMENT_TYPE}" \
    --data_dir "${DATA_DIR}" \
    --vocab_dir "${VOCAB_DIR}" \
    --src_lang "${SRC_LANG}" \
    --tgt_lang "${TGT_LANG}" \
    ${USE_SMALL} \
    --embed_dim ${EMBED_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --n_layers 2 \
    --dropout 0.3 \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --repetition_penalty 1.5 \
    --output_dir "${OUTPUT_SUBDIR}"

echo ""
echo "============================================================"
echo "✓ 消融实验训练完成！"
echo "============================================================"

# 根据实验类型确定评估目录
if [ "${EXPERIMENT_TYPE}" == "attention" ]; then
    EVAL_DIR="${OUTPUT_SUBDIR}/attention_ablation"
elif [ "${EXPERIMENT_TYPE}" == "training_strategy" ]; then
    EVAL_DIR="${OUTPUT_SUBDIR}/training_strategy_ablation"
elif [ "${EXPERIMENT_TYPE}" == "full" ]; then
    EVAL_DIR="${OUTPUT_SUBDIR}/full_ablation"
else
    EVAL_DIR="${OUTPUT_SUBDIR}"
fi

# 运行评估
echo ""
echo "============================================================"
echo "开始评估所有模型（包含多种解码策略）..."
echo "============================================================"

python src/evaluate_rnn_ablation.py \
    --mode ablation \
    --experiment_dir "${EVAL_DIR}" \
    --data_dir "${DATA_DIR}" \
    --vocab_dir "${VOCAB_DIR}" \
    --src_lang "${SRC_LANG}" \
    --tgt_lang "${TGT_LANG}" \
    --batch_size 32 \
    --max_len 100 \
    --num_samples 20 \
    --repetition_penalty 1.5 \
    --output_path "${EVAL_DIR}/evaluation_report.json"

echo ""
echo "============================================================"
echo "✓ 所有实验完成！"
echo "============================================================"
echo ""
echo "结果位置:"
echo "  训练日志: ${EVAL_DIR}/*/logs/train_log.json"
echo "  模型检查点: ${EVAL_DIR}/*/checkpoints/model_best.pt"
echo "  评估报告: ${EVAL_DIR}/evaluation_report.json"
echo "  对比汇总: ${EVAL_DIR}/comparison_summary.json"

