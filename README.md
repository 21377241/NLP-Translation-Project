# NLP Translation Project

中英文机器翻译项目，实现了 RNN、Transformer 和 T5 三种翻译模型。

## 项目结构

```
.
├── inference.py              # 一键推理脚本（作业必需）
├── requirements.txt          # Python 依赖
├── data/                     # 数据集和词表
│   ├── vocab/               # 词表文件
│   └── *.jsonl              # 训练/验证/测试数据
├── src/                      # 源代码
│   ├── data_utils.py        # 数据处理工具
│   ├── models/              # 模型实现
│   │   ├── rnn_seq2seq.py
│   │   ├── transformer.py
│   │   └── t5_finetune.py
│   └── train_*.py           # 训练脚本
├── scripts/                  # 运行脚本
├── models/                   # 预训练模型
│   └── mt5-small/           # mT5-small 模型（需要下载权重）
├── experiments/              # 训练输出（模型 checkpoint，需要单独下载）
└── results/                  # 评估结果和可视化

```

## ⚠️ 重要说明：模型文件

由于 GitHub 文件大小限制（单文件 < 100MB），以下大型文件**未包含在仓库中**：

### 1. mT5 预训练模型权重 (1.2GB)
- **文件**: `models/mt5-small/pytorch_model.bin`
- **下载方式**:
  ```bash
  # 方法 1: 从 Hugging Face 下载
  pip install huggingface_hub
  huggingface-cli download google/mt5-small --local-dir models/mt5-small
  
  # 方法 2: 使用 Python 脚本
  from transformers import MT5ForConditionalGeneration
  model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
  model.save_pretrained('models/mt5-small')
  ```

### 2. 训练好的模型 Checkpoint (约 45GB)
- **目录**: `experiments/`
- **说明**: 包含所有训练好的 RNN 和 Transformer 模型
- **需要的核心模型**:
  - `experiments/rnn_en2zh/checkpoints/model_best.pt` (220MB)
  - `experiments/rnn_zh2en/checkpoints/model_best.pt` (232MB)
  - `experiments/transformer_en2zh/checkpoints/model_best.pt` (142MB)
  - `experiments/transformer_zh2en/checkpoints/model_best.pt` (149MB)

**获取方式**:
- 选项 1: 从云盘下载（如果提供）
- 选项 2: 重新训练模型（参见下方训练说明）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型文件

根据上述说明下载必要的模型文件。

### 3. 使用 inference.py 进行翻译

```bash
# RNN 模型：英译中（贪婪解码）
python inference.py --model rnn --input "Hello world" --direction en2zh

# RNN 模型：英译中（束搜索）
python inference.py --model rnn --input "Hello world" --direction en2zh --beam_size 5

# Transformer 模型：中译英
python inference.py --model transformer --input "你好世界" --direction zh2en

# T5 模型：英译中
python inference.py --model t5 --input "Hello world" --direction en2zh

# 指定自定义 checkpoint
python inference.py --model rnn --input "Hello" --checkpoint path/to/model.pt
```

### 4. 训练模型

如果需要重新训练模型：

```bash
# 训练 RNN 模型
bash scripts/run_rnn_en2zh.sh
bash scripts/run_rnn_zh2en.sh

# 训练 Transformer 模型
bash scripts/run_transformer_en2zh.sh
bash scripts/run_transformer_zh2en.sh

# 微调 T5 模型
bash scripts/run_t5_en2zh.sh
bash scripts/run_t5_zh2en.sh
```

## 模型架构

### 1. RNN Seq2Seq
- 编码器-解码器架构
- 支持 LSTM/GRU
- 注意力机制（Dot/Additive/Multiplicative）
- 训练策略：Teacher Forcing / Scheduled Sampling / Free Running

### 2. Transformer
- 标准 Transformer 架构
- 多头注意力机制
- 位置编码

### 3. T5 (mT5-small)
- 预训练的多语言 T5 模型
- 微调用于中英翻译任务

## 评估

```bash
# 评估所有模型
bash scripts/run_evaluation.sh

# RNN 消融实验
bash scripts/run_rnn_ablation_all.sh
bash scripts/visualize_rnn_ablation.sh
```

## 项目报告

- 中文版: `项目报告.md`
- 英文版: `PROJECT_REPORT_EN.md`

## 依赖环境

- Python 3.8+
- PyTorch 1.12+
- transformers (用于 T5 模型)
- jieba (中文分词)
- sacrebleu (BLEU 评估)

详见 `requirements.txt`

## 文件说明

### inference.py 依赖的文件

`inference.py` 需要以下文件才能正常运行：

**必需文件**:
- `src/data_utils.py` - 数据处理工具 ✅
- `src/models/rnn_seq2seq.py` - RNN 模型 ✅
- `src/models/transformer.py` - Transformer 模型 ✅
- `src/models/t5_finetune.py` - T5 模型 ✅
- `data/vocab/vocab_en.json` - 英文词表 ✅
- `data/vocab/vocab_zh.json` - 中文词表 ✅

**模型文件（需要单独下载）**:
- `models/mt5-small/pytorch_model.bin` - mT5 权重 ❌ (1.2GB)
- `experiments/*/checkpoints/model_best.pt` - 训练好的模型 ❌ (每个 100MB+)

## 替代方案

如果无法下载大型模型文件，可以：

1. **仅使用代码**: 查看模型实现和训练逻辑
2. **重新训练**: 使用提供的训练脚本从头训练（需要 GPU）
3. **使用小型测试**: 在小数据集上快速验证代码功能

## License

本项目仅用于学术目的。

## 作者

学号: 21377241

