#!/bin/bash
# ============================================================
# Transformer 模型训练 - 英译中
# ============================================================

set -e

# 配置
PROJECT_DIR="/mnt/afs/250010036/course/NLP"
DATA_DIR="${PROJECT_DIR}/AP0004_Midterm&Final_translation_dataset_zh_en"

# 激活环境
echo "============================================================"
echo "Transformer 训练 - 英译中"
echo "============================================================"
source /mnt/afs/250010036/miniconda3/bin/activate nlp
cd ${PROJECT_DIR}

# 检查GPU
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 训练
echo ""
echo "开始训练..."
python src/train_transformer.py \
    --data_dir "${DATA_DIR}" \
    --src_lang en \
    --tgt_lang zh \
    --d_model 256 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --dim_feedforward 512 \
    --dropout 0.1 \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --repetition_penalty 1.5 \
    --checkpoint_dir experiments/transformer_en2zh/checkpoints \
    --log_dir experiments/transformer_en2zh/logs

echo ""
echo "✓ Transformer 英译中模型训练完成！"
echo "检查点位置: experiments/transformer_en2zh/checkpoints/"


