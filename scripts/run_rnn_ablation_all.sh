#!/bin/bash
# ============================================================
# RNN Seq2Seq 完整消融实验运行脚本
# 
# 运行所有消融实验：
# 1. 注意力机制消融实验（英译中）
# 2. 训练策略消融实验（英译中）
# 3. 注意力机制消融实验（中译英）
# 4. 训练策略消融实验（中译英）
#
# 作者: NLP课程项目
# ============================================================

set -e

SCRIPT_DIR="/mnt/afs/250010036/course/NLP/scripts"
PROJECT_DIR="/mnt/afs/250010036/course/NLP"

# 激活环境
source /mnt/afs/250010036/miniconda3/bin/activate nlp
cd ${PROJECT_DIR}

echo "============================================================"
echo "开始运行RNN完整消融实验"
echo "============================================================"

# ============================================================
# 1. 注意力机制消融实验 - 英译中
# ============================================================
echo ""
echo ">>> [1/4] 注意力机制消融实验 - 英译中"
echo ""
bash ${SCRIPT_DIR}/run_rnn_ablation.sh \
    --experiment_type attention \
    --src_lang en \
    --tgt_lang zh \
    --epochs 15 \
    --batch_size 64

# ============================================================
# 2. 训练策略消融实验 - 英译中
# ============================================================
echo ""
echo ">>> [2/4] 训练策略消融实验 - 英译中"
echo ""
bash ${SCRIPT_DIR}/run_rnn_ablation.sh \
    --experiment_type training_strategy \
    --src_lang en \
    --tgt_lang zh \
    --epochs 15 \
    --batch_size 64

# ============================================================
# 3. 注意力机制消融实验 - 中译英
# ============================================================
echo ""
echo ">>> [3/4] 注意力机制消融实验 - 中译英"
echo ""
bash ${SCRIPT_DIR}/run_rnn_ablation.sh \
    --experiment_type attention \
    --src_lang zh \
    --tgt_lang en \
    --epochs 15 \
    --batch_size 64

# ============================================================
# 4. 训练策略消融实验 - 中译英
# ============================================================
echo ""
echo ">>> [4/4] 训练策略消融实验 - 中译英"
echo ""
bash ${SCRIPT_DIR}/run_rnn_ablation.sh \
    --experiment_type training_strategy \
    --src_lang zh \
    --tgt_lang en \
    --epochs 15 \
    --batch_size 64

echo ""
echo "============================================================"
echo "✓ 所有RNN消融实验完成！"
echo "============================================================"
echo ""
echo "实验结果位置:"
echo "  英译中注意力消融: experiments/rnn_ablation/en2zh/attention_ablation/"
echo "  英译中训练策略消融: experiments/rnn_ablation/en2zh/training_strategy_ablation/"
echo "  中译英注意力消融: experiments/rnn_ablation/zh2en/attention_ablation/"
echo "  中译英训练策略消融: experiments/rnn_ablation/zh2en/training_strategy_ablation/"

