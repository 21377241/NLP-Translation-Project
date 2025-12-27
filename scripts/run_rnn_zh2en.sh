#!/bin/bash
# ============================================================
# RNN Seq2Seq 模型训练 - 中译英
# ============================================================

set -e

# 配置
PROJECT_DIR="/mnt/afs/250010036/course/NLP"
DATA_DIR="${PROJECT_DIR}/AP0004_Midterm&Final_translation_dataset_zh_en"

# 激活环境
echo "============================================================"
echo "RNN Seq2Seq 训练 - 中译英"
echo "============================================================"
source /mnt/afs/250010036/miniconda3/bin/activate nlp
cd ${PROJECT_DIR}

# 检查GPU
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 训练
echo ""
echo "开始训练..."
python src/train_rnn.py \
    --data_dir "${DATA_DIR}" \
    --src_lang zh \
    --tgt_lang en \
    --embed_dim 256 \
    --hidden_dim 512 \
    --n_layers 2 \
    --attn_type multiplicative \
    --dropout 0.3 \
    --epochs 30 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --teacher_forcing_ratio 0.3 \
    --repetition_penalty 1.5 \
    --checkpoint_dir experiments/rnn_zh2en/checkpoints \
    --log_dir experiments/rnn_zh2en/logs

echo ""
echo "✓ RNN 中译英模型训练完成！"
echo "检查点位置: experiments/rnn_zh2en/checkpoints/"


