# 模型文件说明

## 为什么模型文件不在仓库中？

由于 GitHub 限制单个文件不能超过 100MB，以下大型文件无法直接上传：

1. **mT5 预训练模型**: `models/mt5-small/pytorch_model.bin` (1.2GB)
2. **训练好的模型**: `experiments/*/checkpoints/*.pt` (每个 100MB-230MB)

## 如何获取模型文件

### 方法 1: 下载 mT5 预训练模型

```bash
# 使用 Hugging Face CLI
pip install huggingface_hub
huggingface-cli download google/mt5-small --local-dir models/mt5-small

# 或者使用 Python
python -c "
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
model.save_pretrained('models/mt5-small')
tokenizer.save_pretrained('models/mt5-small')
"
```

### 方法 2: 重新训练模型

如果你有 GPU 资源，可以重新训练模型：

```bash
# 训练 RNN 模型（约 2-4 小时，需要 GPU）
bash scripts/run_rnn_en2zh.sh
bash scripts/run_rnn_zh2en.sh

# 训练 Transformer 模型（约 2-4 小时，需要 GPU）
bash scripts/run_transformer_en2zh.sh
bash scripts/run_transformer_zh2en.sh

# 微调 T5 模型（约 4-8 小时，需要 GPU）
bash scripts/run_t5_en2zh.sh
bash scripts/run_t5_zh2en.sh
```

### 方法 3: 使用 Git LFS（推荐用于大文件）

如果你想将模型文件也纳入版本控制，可以使用 Git LFS：

```bash
# 安装 Git LFS
git lfs install

# 跟踪大文件
git lfs track "*.bin"
git lfs track "*.pt"
git lfs track "*.pth"

# 添加并提交
git add .gitattributes
git add models/mt5-small/pytorch_model.bin
git add experiments/*/checkpoints/model_best.pt
git commit -m "Add model files with Git LFS"
git push
```

## 模型文件清单

### 必需的模型文件（用于 inference.py）

#### 1. mT5 预训练模型（用于 T5 翻译）
```
models/mt5-small/
├── config.json              ✅ 已上传
├── pytorch_model.bin        ❌ 需要下载 (1.2GB)
├── special_tokens_map.json  ✅ 已上传
├── spiece.model             ✅ 已上传
└── tokenizer_config.json    ✅ 已上传
```

#### 2. RNN 模型（用于 RNN 翻译）
```
experiments/rnn_en2zh/checkpoints/model_best.pt    ❌ 需要训练 (220MB)
experiments/rnn_zh2en/checkpoints/model_best.pt    ❌ 需要训练 (232MB)
```

#### 3. Transformer 模型（用于 Transformer 翻译）
```
experiments/transformer_en2zh/checkpoints/model_best.pt  ❌ 需要训练 (142MB)
experiments/transformer_zh2en/checkpoints/model_best.pt  ❌ 需要训练 (149MB)
```

### 可选的模型文件（用于消融实验）

```
experiments/rnn_ablation/en2zh/attention_ablation/*/checkpoints/model_best.pt
experiments/rnn_ablation/zh2en/attention_ablation/*/checkpoints/model_best.pt
experiments/rnn_ablation/en2zh/training_strategy_ablation/*/checkpoints/model_best.pt
experiments/rnn_ablation/zh2en/training_strategy_ablation/*/checkpoints/model_best.pt
```

## 快速测试（无需模型文件）

你可以在没有模型文件的情况下测试代码：

```bash
# 查看模型架构
python -c "from src.models.rnn_seq2seq import create_rnn_model; print(create_rnn_model(10000, 10000))"

# 查看数据处理
python -c "from src.data_utils import tokenize_en, tokenize_zh; print(tokenize_en('Hello world'))"

# 运行训练（在小数据集上）
python src/train_rnn.py --train_file data/train_mini.jsonl --valid_file data/valid.jsonl --epochs 1
```

## 云存储链接（如果可用）

如果你有云存储空间（如 Google Drive、百度网盘等），可以将模型文件上传到云端，并在此处添加下载链接：

```
# 示例
mT5 模型: [下载链接]
RNN 模型: [下载链接]
Transformer 模型: [下载链接]
```

## 使用 Git LFS 的注意事项

- GitHub 免费账户的 Git LFS 配额：1GB 存储 + 1GB/月带宽
- 如果模型文件总大小超过配额，考虑使用外部云存储
- Git LFS 适合团队协作和版本管理大文件

## 联系方式

如需获取训练好的模型文件，请联系：
- 学号: 21377241
- 邮箱: [你的邮箱]

