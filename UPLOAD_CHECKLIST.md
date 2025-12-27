# GitHub 上传检查清单

## ✅ 已上传的文件

### 核心代码文件
- ✅ `inference.py` - 一键推理脚本（作业必需）
- ✅ `requirements.txt` - Python 依赖

### 源代码 (src/)
- ✅ `src/data_utils.py` - 数据处理工具
- ✅ `src/models/rnn_seq2seq.py` - RNN 模型
- ✅ `src/models/transformer.py` - Transformer 模型
- ✅ `src/models/t5_finetune.py` - T5 模型
- ✅ `src/train_*.py` - 训练脚本
- ✅ `src/evaluate*.py` - 评估脚本

### 数据文件 (data/)
- ✅ `data/vocab/vocab_en.json` - 英文词表
- ✅ `data/vocab/vocab_zh.json` - 中文词表
- ✅ `data/*.jsonl` - 数据集文件

### 脚本文件 (scripts/)
- ✅ `scripts/run_*.sh` - 所有运行脚本

### 模型配置 (models/)
- ✅ `models/mt5-small/config.json`
- ✅ `models/mt5-small/special_tokens_map.json`
- ✅ `models/mt5-small/spiece.model`
- ✅ `models/mt5-small/tokenizer_config.json`

### 文档
- ✅ `README.md` - 项目说明
- ✅ `MODELS_README.md` - 模型文件说明
- ✅ `项目报告.md` - 中文项目报告
- ✅ `PROJECT_REPORT_EN.md` - 英文项目报告
- ✅ `docs/` - 其他文档

## ❌ 未上传的文件（由于 GitHub 限制）

### 大型模型文件
- ❌ `models/mt5-small/pytorch_model.bin` (1.2GB)
  - 原因：超过 GitHub 100MB 限制
  - 解决方案：从 Hugging Face 下载（见 MODELS_README.md）

### 训练好的模型 Checkpoint
- ❌ `experiments/*/checkpoints/*.pt` (每个 100MB-230MB)
  - 原因：超过 GitHub 100MB 限制
  - 解决方案：重新训练或使用 Git LFS（见 MODELS_README.md）

### 实验结果
- ❌ `results/` (1.6MB，已在 .gitignore 中排除)
  - 原因：可重新生成
  - 解决方案：运行评估脚本重新生成

## 🔍 inference.py 依赖检查

### inference.py 需要的文件

#### ✅ 已上传且可用
1. `src/data_utils.py` - 数据处理 ✅
2. `src/models/rnn_seq2seq.py` - RNN 模型定义 ✅
3. `src/models/transformer.py` - Transformer 模型定义 ✅
4. `src/models/t5_finetune.py` - T5 模型定义 ✅
5. `data/vocab/vocab_en.json` - 英文词表 ✅
6. `data/vocab/vocab_zh.json` - 中文词表 ✅

#### ❌ 需要额外准备
1. `models/mt5-small/pytorch_model.bin` - T5 模型权重 ❌
   - **影响**: 无法使用 `--model t5` 选项
   - **解决**: 运行 `huggingface-cli download google/mt5-small --local-dir models/mt5-small`

2. `experiments/rnn_en2zh/checkpoints/model_best.pt` - RNN 英译中模型 ❌
   - **影响**: 无法使用 `--model rnn --direction en2zh`
   - **解决**: 运行 `bash scripts/run_rnn_en2zh.sh` 训练模型

3. `experiments/rnn_zh2en/checkpoints/model_best.pt` - RNN 中译英模型 ❌
   - **影响**: 无法使用 `--model rnn --direction zh2en`
   - **解决**: 运行 `bash scripts/run_rnn_zh2en.sh` 训练模型

4. `experiments/transformer_en2zh/checkpoints/model_best.pt` - Transformer 英译中 ❌
   - **影响**: 无法使用 `--model transformer --direction en2zh`
   - **解决**: 运行 `bash scripts/run_transformer_en2zh.sh` 训练模型

5. `experiments/transformer_zh2en/checkpoints/model_best.pt` - Transformer 中译英 ❌
   - **影响**: 无法使用 `--model transformer --direction zh2en`
   - **解决**: 运行 `bash scripts/run_transformer_zh2en.sh` 训练模型

## 📋 使用说明

### 场景 1: 只查看代码（无需模型文件）
```bash
git clone https://github.com/21377241/NLP-Translation-Project.git
cd NLP-Translation-Project
# 查看代码实现
cat inference.py
cat src/models/rnn_seq2seq.py
```

### 场景 2: 运行 inference.py（需要模型文件）
```bash
git clone https://github.com/21377241/NLP-Translation-Project.git
cd NLP-Translation-Project
pip install -r requirements.txt

# 下载 mT5 模型（用于 T5 翻译）
huggingface-cli download google/mt5-small --local-dir models/mt5-small

# 训练 RNN 模型（用于 RNN 翻译）
bash scripts/run_rnn_en2zh.sh

# 运行推理
python inference.py --model rnn --input "Hello world" --direction en2zh
```

### 场景 3: 使用 Git LFS 管理大文件
```bash
# 在本地仓库启用 Git LFS
git lfs install
git lfs track "*.bin"
git lfs track "*.pt"

# 添加大文件
git add models/mt5-small/pytorch_model.bin
git add experiments/*/checkpoints/model_best.pt
git commit -m "Add model files with Git LFS"
git push
```

## 🎯 推荐方案

### 方案 A: 最小化仓库（当前方案）
- ✅ 优点：仓库小，克隆快速
- ❌ 缺点：需要额外步骤获取模型
- 适用：代码审查、协作开发

### 方案 B: 使用 Git LFS
- ✅ 优点：所有文件统一管理
- ❌ 缺点：需要 Git LFS，可能超出配额
- 适用：完整项目交付

### 方案 C: 外部云存储
- ✅ 优点：不受 GitHub 限制
- ❌ 缺点：需要维护云存储链接
- 适用：大型模型分享

## 📝 总结

当前仓库包含了 **所有必需的代码和配置文件**，可以：
1. ✅ 查看和理解代码实现
2. ✅ 重新训练所有模型
3. ✅ 运行评估和实验

但是 **不包含训练好的模型权重**，如需直接运行 `inference.py`，需要：
1. 下载 mT5 预训练模型（1.2GB）
2. 训练或下载 RNN/Transformer 模型（每个 100MB-230MB）

详细说明请查看 `README.md` 和 `MODELS_README.md`。
