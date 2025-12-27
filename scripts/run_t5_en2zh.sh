#!/bin/bash
# ============================================================
# mT5 模型微调 - 英译中
# ============================================================

set -e

# 配置
PROJECT_DIR="/mnt/afs/250010036/course/NLP"
DATA_DIR="${PROJECT_DIR}/AP0004_Midterm&Final_translation_dataset_zh_en"
TRAIN_FILE="${DATA_DIR}/train_10k.jsonl"
VALID_FILE="${DATA_DIR}/valid.jsonl"

# 激活环境
echo "============================================================"
echo "mT5 模型微调 - 英译中"
echo "============================================================"
source /mnt/afs/250010036/miniconda3/bin/activate nlp
cd ${PROJECT_DIR}

# 检查GPU
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 训练
echo ""
echo "开始微调..."
python src/train_t5.py \
    --train_file "${TRAIN_FILE}" \
    --valid_file "${VALID_FILE}" \
    --model_name ./models/mt5-small \
    --src_lang en \
    --tgt_lang zh \
    --epochs 15 \
    --batch_size 4 \
    --eval_batch_size 8 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --max_src_len 256 \
    --max_tgt_len 256 \
    --num_beams 4 \
    --gradient_accumulation_steps 2 \
    --max_grad_norm 1.0 \
    --early_stopping \
    --early_stopping_patience 5 \
    --output_dir experiments/t5_en2zh

echo ""
echo "✓ mT5 英译中模型微调完成！"
echo "模型位置: experiments/t5_en2zh/best_model/"


