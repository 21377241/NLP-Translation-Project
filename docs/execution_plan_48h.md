# 中英机器翻译项目 - 48小时紧急执行方案（修订版）

**制定时间**：2025年12月26日 20:00  
**最后更新**：2025年12月26日 21:00  
**截止时间**：2025年12月28日 23:59  
**剩余时间**：约48小时  
**策略定位**：保证核心功能完成，确保可提交

---

## ⚠️ 重要修正（v2.0更新）

### 修正1：RNN编码器方向问题
**原问题**：初版计划使用了双向LSTM，但作业要求是单向层  
**解决方案**：
- ✅ 默认使用**单向LSTM/GRU**（严格符合作业要求）
- ✅ 可选：实现双向版本作为对比实验，并在报告中说明
- ✅ 更新了代码示例和配置文件

### 修正2：Transformer实现复杂度
**原问题**：从零实现完整Transformer在3小时内难以保证无bug  
**解决方案**：
- ✅ **推荐使用PyTorch内置`nn.Transformer`**（稳定可靠）
- ✅ 提供了完整的封装代码（包含位置编码）
- ✅ 可选：有时间再实现自定义版本进行对比
- ⏱️ 预计节省2-3小时开发和调试时间

### 新增3：实验管理系统
**新增内容**：独立的实验管理框架  
**核心功能**：
- ✅ 每个实验有独立的目录、配置、结果文件
- ✅ 使用YAML配置文件管理超参数
- ✅ 统一的实验执行接口
- ✅ 自动化的实验对比工具
- 🎯 **确保实验互不干扰，结果可复现**

### 关键优势
1. **更符合要求**：严格遵循作业规定（单向层）
2. **更快实现**：使用成熟组件，减少调试时间
3. **更好管理**：实验系统化，便于报告撰写
4. **更易复现**：配置文件记录所有细节

---

## 📊 数据集分析报告

### 数据集概况
| 文件名 | 数量 | 用途 | 优先级 |
|--------|------|------|--------|
| `train_10k.jsonl` | 10,000条 | 快速训练迭代 | **P0 必用** |
| `train_100k.jsonl` | 100,000条 | 完整训练（时间允许） | P1 可选 |
| `valid.jsonl` | 500条 | 验证集 | P0 必用 |
| `test.jsonl` | 200条 | 测试集（最终评估） | P0 必用 |

### 数据格式
```json
{
  "en": "1929 or 1989?",
  "zh": "1929年还是1989年?",
  "index": 0
}
```

### 数据特点分析
- ✅ **格式统一**：标准JSON Lines格式，易于解析
- ✅ **双语对齐**：英中句对一一对应
- ✅ **质量较高**：来自新闻语料，句子结构完整
- ⚠️ **领域特定**：主要是新闻/政治经济类文本
- ⚠️ **可能问题**：长句较多、专业术语、标点差异

### 推荐策略
**第一阶段**：仅使用 `train_10k.jsonl`（保证快速迭代）  
**第二阶段**：如果时间充裕，使用 `train_100k.jsonl` 提升性能

---

## ⏰ 48小时时间轴（精确到小时）

### 🔥 Day 1 - 12月26日（今天剩余时间 + 晚上）
**目标：完成环境搭建 + 数据处理 + RNN基础实现**

| 时间 | 任务 | 具体内容 | 检查点 |
|------|------|----------|--------|
| 20:00-21:00 | 环境搭建 | 安装依赖、创建项目结构 | `requirements.txt`、目录创建完成 |
| 21:00-22:30 | 数据预处理 | 分词、构建词表、数据集类 | 可以加载batch数据 |
| 22:30-01:00 | RNN模型实现 | Encoder+Decoder+Attention | 模型可以forward |
| 01:00-02:00 | 训练脚本 | 训练循环、保存checkpoint | RNN开始训练，loss下降 |

### ⚡ Day 2 - 12月27日（全天）
**目标：完成Transformer + 实验对比 + 报告初稿**

| 时间 | 任务 | 具体内容 | 检查点 |
|------|------|----------|--------|
| 08:00-09:00 | 继续RNN训练 | 监控训练、调整超参数 | RNN模型训练完成 |
| 09:00-10:00 | 评估系统 | BLEU计算、inference.py | 能够评估RNN模型 |
| 10:00-13:00 | Transformer实现 | 完整Transformer架构 | Transformer可以训练 |
| 13:00-14:00 | 午餐休息 | 检查GPU/训练进度 | - |
| 14:00-16:00 | Transformer训练 | 训练Transformer模型 | Transformer训练完成 |
| 16:00-17:00 | 模型对比实验 | 两模型BLEU对比、生成样例 | 对比数据收集完成 |
| 17:00-19:00 | 报告撰写（第一部分） | 模型架构、实现过程 | 报告50%完成 |
| 19:00-20:00 | 晚餐休息 | 代码提交Git | - |
| 20:00-22:00 | 报告撰写（第二部分） | 实验结果、分析讨论 | 报告90%完成 |
| 22:00-23:00 | 对比实验（如有时间） | 注意力机制对比等 | 额外实验数据 |
| 23:00-00:00 | 代码整理 | README、注释完善 | GitHub仓库完善 |

### 🎯 Day 3 - 12月28日（截止日）
**目标：最后调整 + 报告完善 + 提交**

| 时间 | 任务 | 具体内容 | 检查点 |
|------|------|----------|--------|
| 08:00-10:00 | 报告完善 | 添加图表、修改格式 | 报告100%完成 |
| 10:00-11:00 | inference.py测试 | 确保推理脚本完美运行 | 可一键推理 |
| 11:00-12:00 | 最后检查 | 所有文件、Git、PDF | - |
| 12:00-14:00 | 缓冲时间 | 处理突发问题 | - |
| 14:00-20:00 | **安全边界** | 准备展示PPT（如需要） | - |
| **23:59** | **截止** | 提交报告到Piazza | ✅ 完成 |

---

## 🏗️ 项目结构（实际执行版）

```
NLP/
├── AP0004_Midterm&Final_translation_dataset_zh_en/  # 数据集（已有）
│   ├── train_10k.jsonl         # 10k训练集
│   ├── train_100k.jsonl        # 100k训练集（备用）
│   ├── valid.jsonl             # 验证集
│   └── test.jsonl              # 测试集
│
├── data/                        # 处理后的数据
│   ├── vocab/                   # 词表
│   │   ├── vocab_en.json       # 英文词表
│   │   └── vocab_zh.json       # 中文词表
│   └── processed/               # 预处理后的数据（可选）
│
├── experiments/                 # 🔥 实验管理目录（新增）
│   ├── exp_001_rnn_baseline/   # 实验1：RNN基线
│   │   ├── config.yaml         # 实验配置
│   │   ├── checkpoints/        # 模型检查点
│   │   ├── logs/               # 训练日志
│   │   └── results.json        # 实验结果
│   ├── exp_002_transformer_baseline/  # 实验2：Transformer基线
│   ├── exp_003_rnn_attention_comparison/  # 实验3：注意力对比
│   ├── exp_004_transformer_ablation/      # 实验4：Transformer消融
│   └── summary.md              # 所有实验总结
│
├── src/                         # 源代码目录
│   ├── __init__.py
│   │
│   ├── data_utils.py           # 数据处理工具
│   │   # - load_data()         # 加载JSONL
│   │   # - Tokenizer类         # 中英文分词
│   │   # - build_vocab()       # 构建词表
│   │   # - TranslationDataset  # PyTorch Dataset
│   │
│   ├── models/                 # 模型定义
│   │   ├── __init__.py
│   │   ├── rnn_seq2seq.py     # RNN翻译模型
│   │   │   # - Encoder（BiLSTM，2层）
│   │   │   # - Attention（Dot-product/Additive）
│   │   │   # - Decoder（LSTM，2层）
│   │   │   # - Seq2Seq（完整模型）
│   │   │
│   │   └── transformer.py      # Transformer模型
│   │       # - PositionalEncoding
│   │       # - MultiHeadAttention
│   │       # - TransformerEncoder
│   │       # - TransformerDecoder
│   │       # - Transformer（完整模型）
│   │
│   ├── train_rnn.py            # RNN训练脚本
│   ├── train_transformer.py    # Transformer训练脚本
│   │
│   ├── evaluate.py             # 评估工具
│   │   # - calculate_bleu()
│   │   # - translate_batch()
│   │
│   └── config.py               # 配置文件（超参数）
│
├── checkpoints/                # 模型检查点
│   ├── rnn_best.pt
│   └── transformer_best.pt
│
├── results/                    # 实验结果
│   ├── rnn_results.json       # RNN实验结果
│   ├── transformer_results.json
│   ├── translations/          # 翻译样例
│   └── figures/               # 可视化图表
│
├── docs/                       # 文档
│   ├── project_require.md     # 作业要求
│   ├── implementation_plan.md # 原始计划
│   └── execution_plan_48h.md  # 本文档（执行方案）
│
├── inference.py                # 🔥 一键推理脚本（必需！）
├── requirements.txt            # 依赖包
├── README.md                   # 项目说明
└── .gitignore                  # Git忽略文件
```

---

## 💻 技术实现细节

### 1️⃣ 数据处理模块（data_utils.py）

#### 关键功能
```python
# 1. 加载数据
def load_data(file_path):
    """加载JSONL格式数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 2. 中文分词（使用jieba）
def tokenize_zh(text):
    return list(jieba.cut(text))

# 3. 英文分词（简单split）
def tokenize_en(text):
    return text.lower().split()

# 4. 构建词表
def build_vocab(tokens_list, min_freq=2, max_size=30000):
    """
    构建词表：<PAD>=0, <UNK>=1, <SOS>=2, <EOS>=3
    """
    counter = Counter([token for tokens in tokens_list for token in tokens])
    vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

# 5. Dataset类
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, max_len=100):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __getitem__(self, idx):
        # 返回：src_ids, tgt_ids, src_len, tgt_len
        pass
```

#### 预处理流程
```
原始数据 → 分词 → 构建词表 → 数字化 → DataLoader
```

---

### 2️⃣ RNN模型实现（rnn_seq2seq.py）

#### ⚠️ 重要修正：编码器方向问题

**作业要求**：编码器和解码器各包含两个**单向层**（unidirectional layers）

**方案调整**：
1. **标准实现**（推荐）：使用单向LSTM/GRU，符合作业要求
2. **增强实现**（可选）：使用双向编码器，但需在报告中说明并对比单向/双向效果

#### 模型架构（单向版本 - 符合要求）
```python
class Encoder(nn.Module):
    """
    单向编码器（符合作业要求）
    输入：源语言序列 [batch, src_len]
    输出：隐藏状态 [batch, src_len, hidden], 最终状态
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, 
                           bidirectional=False,  # 单向！
                           batch_first=True, 
                           dropout=0.3 if n_layers > 1 else 0)

class Attention(nn.Module):
    """
    注意力机制（可切换不同类型）
    支持：点积（dot-product）、加性（additive）、乘性（multiplicative）
    """
    def __init__(self, hidden_dim, attn_type='dot'):
        super().__init__()
        self.attn_type = attn_type
        if attn_type == 'additive':
            # Bahdanau注意力
            self.W1 = nn.Linear(hidden_dim, hidden_dim)
            self.W2 = nn.Linear(hidden_dim, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1)
        elif attn_type == 'multiplicative':
            self.W = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, query, keys, values, mask=None):
        # query: [batch, hidden] (decoder hidden state)
        # keys: [batch, src_len, hidden] (encoder outputs)
        # values: [batch, src_len, hidden] (same as keys)
        
        if self.attn_type == 'dot':
            # 点积注意力
            scores = torch.matmul(query.unsqueeze(1), keys.transpose(1, 2))
            # scores: [batch, 1, src_len]
        elif self.attn_type == 'additive':
            # 加性注意力（Bahdanau）
            q = self.W1(query).unsqueeze(1)  # [batch, 1, hidden]
            k = self.W2(keys)  # [batch, src_len, hidden]
            scores = self.v(torch.tanh(q + k))  # [batch, src_len, 1]
            scores = scores.transpose(1, 2)  # [batch, 1, src_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        
        attn_weights = F.softmax(scores, dim=-1)  # [batch, 1, src_len]
        context = torch.matmul(attn_weights, values)  # [batch, 1, hidden]
        
        return context.squeeze(1), attn_weights.squeeze(1)
        
class Decoder(nn.Module):
    """
    带注意力的单向解码器（符合作业要求）
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=2, attn_type='dot'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(hidden_dim, attn_type)
        # 输入是embedding + context vector
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, n_layers,
                           batch_first=True,
                           dropout=0.3 if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
```

#### 训练策略
- **Teacher Forcing**：解码器输入使用真实目标序列（训练初期）
- **Free Running**：解码器输入使用自己的预测（可选对比实验）

#### 超参数设置（快速版 - 符合作业要求）
```python
CONFIG_RNN = {
    # 模型架构（严格符合作业要求）
    'embed_dim': 256,
    'hidden_dim': 256,
    'n_layers': 2,              # 编码器和解码器各2层
    'bidirectional': False,     # ⚠️ 单向（unidirectional）
    'dropout': 0.3,
    'attention_type': 'dot',    # dot, additive, multiplicative
    
    # 训练参数
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 15,
    'max_len': 100,
    'grad_clip': 1.0,
    'teacher_forcing_ratio': 1.0,  # Teacher Forcing vs Free Running
    
    # 词表
    'vocab_size': 30000,
    'min_freq': 2,
}

# 可选：双向编码器版本（用于对比实验）
CONFIG_RNN_BIDIRECTIONAL = {
    **CONFIG_RNN,
    'bidirectional': True,  # 使用双向编码器
    # 注意：使用双向时，需要在报告中说明并对比单向/双向的差异
}
```

---

### 3️⃣ Transformer模型实现（transformer.py）

#### ⚠️ 重要简化：使用PyTorch内置Transformer

**问题**：从零实现完整Transformer在3小时内很难保证无bug运行

**解决方案**：使用PyTorch内置的`nn.Transformer`，大幅简化实现

#### 推荐实现（使用内置模块）

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码（必须自己实现，因为内置Transformer不包含）"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerNMT(nn.Module):
    """
    使用PyTorch内置Transformer的翻译模型
    优势：稳定可靠、实现简单、3小时内可完成
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 d_model=256, nhead=4, num_encoder_layers=3, 
                 num_decoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # 🔥 使用PyTorch内置Transformer（核心简化）
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 使用batch_first格式
        )
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        """生成decoder的causal mask（下三角）"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        """
        src: [batch, src_len]
        tgt: [batch, tgt_len]
        """
        # 生成tgt mask（防止看到未来信息）
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Embedding + Positional Encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Transformer forward
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # 输出投影
        logits = self.fc_out(output)
        return logits
    
    def translate(self, src, max_len=100, sos_idx=2, eos_idx=3):
        """贪婪解码（推理时使用）"""
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # 编码源序列
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb)
        
        # 初始化目标序列（<SOS>）
        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
            
            output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(output[:, -1, :])  # 只取最后一个token
            
            next_token = logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 检查是否所有序列都生成了<EOS>
            if (next_token == eos_idx).all():
                break
        
        return tgt[:, 1:]  # 去掉<SOS>
```

#### 可选：从零实现（仅当有充足时间）

如果时间允许且想展示更多技术细节，可以实现以下组件：

```python
class MultiHeadAttention(nn.Module):
    """自定义多头注意力（用于消融实验）"""
    pass

class TransformerEncoderLayer(nn.Module):
    """自定义编码器层（用于修改归一化方式等）"""
    pass
```

**建议**：先用内置版本跑通，有时间再实现自定义版本进行对比

#### 超参数设置（快速版）
```python
CONFIG_TRANSFORMER = {
    'd_model': 256,
    'n_heads': 4,
    'n_layers': 3,  # Encoder和Decoder各3层
    'd_ff': 1024,
    'dropout': 0.1,
    'batch_size': 64,
    'learning_rate': 0.0001,
    'epochs': 15,
    'max_len': 100,
}
```

---

### 4️⃣ 训练脚本（train_rnn.py / train_transformer.py）

#### 通用训练流程
```python
def train(model, train_loader, valid_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD
    
    best_bleu = 0
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            src, tgt = batch
            output = model(src, tgt[:, :-1])  # Teacher forcing
            loss = criterion(output.view(-1, vocab_size), tgt[:, 1:].view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        bleu = evaluate(model, valid_loader)
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, BLEU={bleu:.2f}")
        
        # 保存最佳模型
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), 'checkpoints/model_best.pt')
```

---

### 5️⃣ 评估系统（evaluate.py）

#### BLEU-4计算
```python
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

def calculate_bleu(model, test_loader, vocab_tgt):
    """计算BLEU分数"""
    references = []
    hypotheses = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            src, tgt = batch
            pred = model.translate(src)  # 贪婪解码或束搜索
            
            # 转换为文本
            for i in range(len(pred)):
                ref = ids_to_tokens(tgt[i], vocab_tgt)
                hyp = ids_to_tokens(pred[i], vocab_tgt)
                references.append([ref])
                hypotheses.append(hyp)
    
    bleu = corpus_bleu(references, hypotheses)
    return bleu * 100
```

---

### 6️⃣ 推理脚本（inference.py）🔥

```python
#!/usr/bin/env python3
"""
一键推理脚本 - 必需文件
用法：python inference.py --model rnn --input "Hello world"
"""
import argparse
import torch
from src.models.rnn_seq2seq import Seq2Seq
from src.models.transformer import Transformer
from src.data_utils import Tokenizer, load_vocab

def translate(text, model, src_vocab, tgt_vocab, device='cpu'):
    """翻译单个句子"""
    model.eval()
    tokens = tokenize(text)
    ids = [src_vocab.get(t, 1) for t in tokens]  # 1=<UNK>
    src_tensor = torch.LongTensor([ids]).to(device)
    
    with torch.no_grad():
        output_ids = model.translate(src_tensor, max_len=100)
    
    output_tokens = [tgt_vocab[id] for id in output_ids]
    return ' '.join(output_tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['rnn', 'transformer'], required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--direction', choices=['en2zh', 'zh2en'], default='en2zh')
    args = parser.parse_args()
    
    # 加载模型
    model = load_model(args.model)
    vocab_src, vocab_tgt = load_vocabs(args.direction)
    
    # 翻译
    result = translate(args.input, model, vocab_src, vocab_tgt)
    print(f"输入：{args.input}")
    print(f"翻译：{result}")
```

---

## 📁 实验管理系统（独立互不干扰）

### 为什么需要实验管理？

在多个实验之间切换时，容易出现：
- ❌ 配置混乱（不知道哪个模型用了什么超参数）
- ❌ 结果覆盖（新实验覆盖旧实验的checkpoint）
- ❌ 无法复现（忘记某个实验的具体设置）

**解决方案**：为每个实验创建独立的目录和配置文件

---

### 实验目录结构

```
experiments/
├── exp_001_rnn_baseline/              # 实验1：RNN基线模型
│   ├── config.yaml                    # 实验配置（超参数、数据路径等）
│   ├── checkpoints/                   # 模型检查点
│   │   ├── model_epoch_5.pt
│   │   ├── model_epoch_10.pt
│   │   └── model_best.pt             # 最佳模型
│   ├── logs/                          # 训练日志
│   │   └── train.log
│   ├── results/                       # 实验结果
│   │   ├── metrics.json              # BLEU、Loss等指标
│   │   ├── translations.txt          # 翻译样例
│   │   └── figures/                  # 图表
│   │       ├── loss_curve.png
│   │       └── bleu_curve.png
│   └── README.md                      # 实验说明
│
├── exp_002_transformer_baseline/      # 实验2：Transformer基线
│   └── (同上结构)
│
├── exp_003_rnn_dot_vs_additive/       # 实验3：RNN注意力机制对比
│   ├── dot_attention/                 # 子实验：点积注意力
│   ├── additive_attention/            # 子实验：加性注意力
│   └── comparison.md                  # 对比结果
│
├── exp_004_transformer_ablation/      # 实验4：Transformer消融实验
│   ├── with_pos_encoding/
│   ├── without_pos_encoding/
│   └── comparison.md
│
└── summary.md                         # 🔥 所有实验总结（用于报告）
```

---

### 配置文件模板（config.yaml）

```yaml
# exp_001_rnn_baseline/config.yaml
experiment:
  name: "RNN Baseline"
  id: "exp_001"
  description: "RNN with dot-product attention (unidirectional)"
  date: "2025-12-26"

model:
  type: "rnn_seq2seq"
  embed_dim: 256
  hidden_dim: 256
  n_layers: 2
  dropout: 0.3
  attention_type: "dot"  # dot, additive, multiplicative
  bidirectional: false   # 单向（符合要求）

data:
  train_file: "AP0004_Midterm&Final_translation_dataset_zh_en/train_10k.jsonl"
  valid_file: "AP0004_Midterm&Final_translation_dataset_zh_en/valid.jsonl"
  test_file: "AP0004_Midterm&Final_translation_dataset_zh_en/test.jsonl"
  max_len: 100
  min_freq: 2
  vocab_size: 30000

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 15
  optimizer: "adam"
  grad_clip: 1.0
  teacher_forcing_ratio: 1.0
  early_stopping_patience: 3

evaluation:
  beam_size: 1  # 1=greedy, >1=beam search
  metrics: ["bleu-4", "loss"]

output:
  checkpoint_dir: "experiments/exp_001_rnn_baseline/checkpoints"
  log_dir: "experiments/exp_001_rnn_baseline/logs"
  results_dir: "experiments/exp_001_rnn_baseline/results"
```

---

### 实验管理脚本

#### 1. 创建新实验
```bash
# scripts/create_experiment.sh
#!/bin/bash
EXP_ID=$1
EXP_NAME=$2

mkdir -p experiments/${EXP_ID}_{$EXP_NAME}/{checkpoints,logs,results/figures}
echo "Created experiment: ${EXP_ID}"
```

#### 2. 运行实验（统一接口）
```python
# scripts/run_experiment.py
import yaml
import argparse

def run_experiment(config_path):
    """根据配置文件运行实验"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    exp_id = config['experiment']['id']
    model_type = config['model']['type']
    
    print(f"Running experiment: {exp_id}")
    print(f"Model type: {model_type}")
    
    if model_type == 'rnn_seq2seq':
        from src.train_rnn import train_rnn
        train_rnn(config)
    elif model_type == 'transformer':
        from src.train_transformer import train_transformer
        train_transformer(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    args = parser.parse_args()
    
    run_experiment(args.config)
```

**使用方法**：
```bash
python scripts/run_experiment.py --config experiments/exp_001_rnn_baseline/config.yaml
```

---

### 实验结果记录（results/metrics.json）

```json
{
  "experiment_id": "exp_001",
  "experiment_name": "RNN Baseline",
  "date": "2025-12-26",
  "status": "completed",
  
  "training": {
    "total_epochs": 15,
    "best_epoch": 12,
    "training_time_hours": 1.5,
    "convergence": true
  },
  
  "metrics": {
    "train_loss_final": 2.34,
    "valid_loss_best": 2.89,
    "test_bleu_4": 15.67,
    "test_loss": 2.92
  },
  
  "model_info": {
    "total_parameters": 12500000,
    "trainable_parameters": 12500000,
    "model_size_mb": 47.6
  },
  
  "inference": {
    "avg_inference_time_ms": 45,
    "sentences_per_second": 22
  },
  
  "samples": [
    {
      "source": "Hello world",
      "reference": "你好世界",
      "prediction": "你好 世界",
      "bleu": 85.6
    }
  ]
}
```

---

### 实验对比工具

```python
# scripts/compare_experiments.py
import json
import pandas as pd
import matplotlib.pyplot as plt

def compare_experiments(exp_ids):
    """对比多个实验的结果"""
    results = []
    
    for exp_id in exp_ids:
        metrics_path = f"experiments/{exp_id}/results/metrics.json"
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            results.append({
                'Experiment': data['experiment_name'],
                'BLEU-4': data['metrics']['test_bleu_4'],
                'Training Time (h)': data['training']['training_time_hours'],
                'Parameters (M)': data['model_info']['total_parameters'] / 1e6,
                'Inference (ms)': data['inference']['avg_inference_time_ms']
            })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # 生成对比图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].bar(df['Experiment'], df['BLEU-4'])
    axes[0, 0].set_title('BLEU-4 Score')
    axes[0, 0].set_ylabel('BLEU')
    
    axes[0, 1].bar(df['Experiment'], df['Training Time (h)'])
    axes[0, 1].set_title('Training Time')
    axes[0, 1].set_ylabel('Hours')
    
    axes[1, 0].bar(df['Experiment'], df['Parameters (M)'])
    axes[1, 0].set_title('Model Size')
    axes[1, 0].set_ylabel('Parameters (M)')
    
    axes[1, 1].bar(df['Experiment'], df['Inference (ms)'])
    axes[1, 1].set_title('Inference Speed')
    axes[1, 1].set_ylabel('ms/sentence')
    
    plt.tight_layout()
    plt.savefig('experiments/comparison.png', dpi=300)
    print("Comparison chart saved to: experiments/comparison.png")

if __name__ == '__main__':
    # 对比所有完成的实验
    exp_ids = [
        'exp_001_rnn_baseline',
        'exp_002_transformer_baseline',
        'exp_003_rnn_dot_vs_additive'
    ]
    compare_experiments(exp_ids)
```

---

### 实验总结文档（experiments/summary.md）

```markdown
# 实验总结报告

## 实验概览

| ID | 实验名称 | 状态 | BLEU-4 | 训练时间 |
|----|---------|------|--------|---------|
| exp_001 | RNN Baseline | ✅ 完成 | 15.67 | 1.5h |
| exp_002 | Transformer Baseline | ✅ 完成 | 18.23 | 2.1h |
| exp_003 | RNN Attention Comparison | ✅ 完成 | - | 1.8h |
| exp_004 | Transformer Ablation | ⏳ 进行中 | - | - |

## 核心发现

### 1. RNN vs Transformer
- Transformer的BLEU分数比RNN高2.56分
- 但训练时间多40%
- RNN推理速度更快（45ms vs 67ms）

### 2. 注意力机制对比
- 点积注意力：BLEU 15.67
- 加性注意力：BLEU 15.82（略优）
- 差异不大，可能因为模型规模较小

### 3. 位置编码消融
- 有位置编码：BLEU 18.23
- 无位置编码：BLEU 12.45
- 位置编码对Transformer至关重要

## 翻译样例对比

[插入对比表格]

## 结论与建议

[总结关键发现]
```

---

## 🔬 实验对比计划

### P0 - 必须完成的实验（确保提交）

#### 实验1：RNN基线模型
**实验ID**：`exp_001_rnn_baseline`  
**目标**：实现并训练基础RNN翻译模型  
**配置**：
- 模型：单向LSTM（2层）+ 点积注意力
- 数据：train_10k.jsonl
- 训练：15 epochs，batch_size=64
- 解码：贪婪解码

**产出**：
- 训练好的模型checkpoint
- BLEU-4分数
- 10个翻译样例
- 训练曲线图

**执行命令**：
```bash
python scripts/run_experiment.py --config experiments/exp_001_rnn_baseline/config.yaml
```

---

#### 实验2：Transformer基线模型
**实验ID**：`exp_002_transformer_baseline`  
**目标**：使用PyTorch内置Transformer实现翻译  
**配置**：
- 模型：Transformer（3层encoder + 3层decoder）
- 位置编码：sin/cos绝对位置编码
- 数据：train_10k.jsonl
- 训练：15 epochs，batch_size=64

**产出**：
- 训练好的模型checkpoint
- BLEU-4分数
- 10个翻译样例
- 训练曲线图

**执行命令**：
```bash
python scripts/run_experiment.py --config experiments/exp_002_transformer_baseline/config.yaml
```

---

#### 实验3：RNN vs Transformer对比
**实验ID**：在实验1和2的基础上进行对比  
**对比维度**：

| 指标 | RNN | Transformer | 对比分析 |
|------|-----|-------------|----------|
| 训练时间 | ? 小时 | ? 小时 | 哪个更快？ |
| BLEU-4 | ? | ? | 翻译质量差异 |
| 模型参数量 | ? M | ? M | 模型复杂度 |
| 推理速度 | ? ms/句 | ? ms/句 | 实际应用效率 |
| 长句表现 | ? | ? | >50词的句子BLEU |

**产出**：
- 对比表格（用于报告）
- 可视化图表（4个子图）
- 错误案例分析（各5个）

**执行命令**：
```bash
python scripts/compare_experiments.py exp_001_rnn_baseline exp_002_transformer_baseline
```

---

### P1 - 尽量完成的实验（提升报告质量）

#### 实验4：RNN注意力机制对比
**实验ID**：`exp_003_rnn_attention_comparison`  
**目标**：对比不同注意力机制的效果  
**子实验**：
- 4a. 点积注意力（dot-product）- 已在exp_001完成
- 4b. 加性注意力（additive/Bahdanau）
- 4c. 乘性注意力（multiplicative）- 可选

**配置差异**：仅修改`attention_type`参数

**产出**：
- 3种注意力的BLEU对比
- 注意力权重可视化（heatmap）
- 分析：哪种注意力更适合中英翻译

**预期结果**：差异可能不大（<1 BLEU），但展示了理解

---

#### 实验5：Teacher Forcing比例对比
**实验ID**：`exp_004_rnn_teacher_forcing`  
**目标**：研究Teacher Forcing对训练的影响  
**子实验**：
- 5a. TF ratio = 1.0（始终使用真实标签）
- 5b. TF ratio = 0.5（50%概率使用真实标签）
- 5c. TF ratio = 0.0（Free Running，完全使用预测）

**产出**：
- 训练稳定性对比（loss曲线）
- 最终BLEU对比
- 分析：TF对收敛速度和泛化的影响

---

### P2 - 时间允许再做（锦上添花）

#### 实验6：Transformer位置编码消融
**实验ID**：`exp_005_transformer_ablation`  
**目标**：验证位置编码的重要性  
**子实验**：
- 6a. 标准sin/cos位置编码（已在exp_002完成）
- 6b. 不使用位置编码

**产出**：
- BLEU对比（预期差距显著）
- 证明位置编码对Transformer的必要性

---

#### 实验7：束搜索 vs 贪婪解码
**实验ID**：`exp_006_decoding_strategy`  
**目标**：对比不同解码策略  
**子实验**：
- 7a. 贪婪解码（beam_size=1）
- 7b. 束搜索（beam_size=3）
- 7c. 束搜索（beam_size=5）

**产出**：
- BLEU提升幅度
- 推理时间增加
- 分析：性能与效率的权衡

---

#### 实验8：大数据集训练（可选）
**实验ID**：`exp_007_large_data`  
**目标**：如果时间允许，使用100k数据集重新训练  
**预期**：BLEU可能提升3-5分

---

### 实验优先级总结

**必做（确保报告完整）**：
1. ✅ exp_001: RNN基线
2. ✅ exp_002: Transformer基线  
3. ✅ 两模型对比分析

**推荐做（提升报告质量）**：
4. ✅ exp_003: 注意力机制对比（RNN要求）
5. ⭕ exp_004: Teacher Forcing对比（如果时间允许）

**可选做（时间充裕才考虑）**：
6. ⭕ exp_005: 位置编码消融
7. ⭕ exp_006: 解码策略对比
8. ⭕ exp_007: 大数据集训练

---

### 实验时间规划

| 实验 | 预计时间 | 何时执行 |
|-----|---------|---------|
| exp_001 | 2小时 | Day 1晚上 |
| exp_002 | 2.5小时 | Day 2上午 |
| 两模型对比 | 1小时 | Day 2下午 |
| exp_003 | 2小时 | Day 2下午（如有时间）|
| 其他实验 | - | 视情况而定 |

---

## 📝 报告结构（10-15页）

### 报告大纲
```
第1页：封面
  - 课程名称、项目标题
  - 学号、姓名
  - GitHub仓库URL ⚠️ 必须

第2页：摘要（0.5页）
  - 项目目标
  - 使用的模型和方法
  - 主要结论（3-5条）

第3-4页：数据集与预处理（1.5页）
  - 数据集统计
  - 预处理流程图
  - 词表大小、平均句长等统计

第5-7页：模型架构（3页）
  - RNN架构图 + 说明（1页）
  - Transformer架构图 + 说明（1页）
  - 关键代码片段（0.5页）
  - 超参数表格（0.5页）

第8-9页：实现过程（1.5页）
  - 技术选型
  - 遇到的主要问题和解决方案
  - 训练策略

第10-12页：实验结果（3页）
  - 训练曲线对比图（Loss/BLEU随epoch变化）
  - BLEU对比表格
  - 翻译样例展示（中译英、英译中各5个）
  - 注意力机制对比（如完成）
  - 错误案例分析

第13页：分析与讨论（1页）
  - RNN vs Transformer优劣对比表
  - 训练效率分析
  - 翻译质量分析
  - 长句处理能力

第14页：个人反思（1页）
  - 学到的知识和技能
  - 遇到的最大挑战
  - 可改进的方向
  - 心得体会

第15页：参考文献
```

### 报告制作建议
1. **使用LaTeX或Word模板**：确保格式专业
2. **图表清晰**：使用matplotlib生成高质量图表
3. **代码片段**：选择最关键的代码，添加注释
4. **诚实汇报**：即使BLEU不高，也要分析原因
5. **突出思考**：体现对模型原理的理解

---

## 📦 依赖包清单（requirements.txt）

```txt
# 深度学习框架
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# Hugging Face（用于预训练模型，可选）
transformers>=4.30.0

# 分词工具
jieba>=0.42.1
nltk>=3.8

# 评估工具
sacrebleu>=2.3.0

# 数据处理
numpy>=1.24.0
pandas>=2.0.0

# 可视化
matplotlib>=3.7.0
seaborn>=0.12.0

# 工具库
tqdm>=4.65.0
tensorboard>=2.13.0

# 其他
scikit-learn>=1.3.0
```

**安装命令**：
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

---

## ✅ 检查清单（确保完成）

### 代码部分
- [ ] `src/data_utils.py` - 数据处理工具
- [ ] `src/models/rnn_seq2seq.py` - RNN模型
- [ ] `src/models/transformer.py` - Transformer模型
- [ ] `src/train_rnn.py` - RNN训练脚本
- [ ] `src/train_transformer.py` - Transformer训练脚本
- [ ] `src/evaluate.py` - 评估脚本
- [ ] `inference.py` - 🔥 一键推理脚本（必需）
- [ ] `requirements.txt` - 依赖包
- [ ] `README.md` - 项目说明

### 模型部分
- [ ] RNN模型训练完成（至少10个epoch）
- [ ] Transformer模型训练完成（至少10个epoch）
- [ ] 保存最佳checkpoint（`checkpoints/`目录）
- [ ] 在test.jsonl上评估BLEU

### 实验部分
- [ ] RNN vs Transformer对比数据
- [ ] 至少一个对比实验（注意力机制或其他）
- [ ] 翻译样例（10个以上）
- [ ] 训练曲线图

### 报告部分
- [ ] 报告PDF完成（10页以上）
- [ ] 文件命名正确：学号_姓名.pdf
- [ ] 首页包含GitHub仓库URL
- [ ] 包含所有必需章节

### Git仓库
- [ ] 代码已上传GitHub
- [ ] README写清楚运行方法
- [ ] .gitignore配置正确（排除数据集、模型权重）
- [ ] 仓库设置为public（或提供访问权限）

---

## ⚠️ 关键风险与应对

### 风险1：训练时间过长
**应对**：
- 使用更小的模型（hidden_dim=128）
- 减少层数（n_layers=1）
- 减少epoch（5-10个epoch足够）
- 只使用train_10k数据集

### 风险2：显存不足（GPU OOM）
**应对**：
- 减小batch_size（从64降到32或16）
- 减小max_len（从100降到50）
- 使用梯度累积（accumulate gradients）
- CPU训练（慢但可行）

### 风险3：BLEU分数很低
**应对**：
- **不要慌！** 作业要求明确说评分不依赖BLEU高低
- 在报告中诚实汇报，分析可能原因：
  - 训练数据少、训练时间短
  - 模型规模小、超参数未充分调优
  - 中文分词质量、词表大小限制
- 展示模型确实在学习（loss下降曲线）
- 提出改进方向

### 风险4：代码有bug，无法运行
**应对**：
- 每完成一个模块立即测试
- 使用小数据（3-5个样本）先过拟合
- 打印中间tensor的shape
- 参考PyTorch官方教程代码

### 风险5：时间不够
**应对**：
- **立即启动最小可行方案（MVP）**
- 放弃所有P2实验，只做P0
- 放弃预训练模型微调
- 报告简化，确保结构完整

---

## 🚀 立即行动计划（今晚必做）

### 第一步：环境搭建（20:00-21:00）

```bash
# 1. 创建项目结构
cd /mnt/afs/250010036/course/NLP
mkdir -p src/models checkpoints results/figures data/vocab docs

# 2. 创建requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
jieba>=0.42.1
nltk>=3.8
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
sacrebleu>=2.3.0
EOF

# 3. 安装依赖（如果环境未配置）
# pip install -r requirements.txt
# python -c "import nltk; nltk.download('punkt')"

# 4. 初始化Git（如果还没有）
git init
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pth
*.pt
checkpoints/
data/processed/
*.log
.ipynb_checkpoints/
EOF

git add .
git commit -m "Initial project structure"
```

### 第二步：数据预处理（21:00-22:30）

创建 `src/data_utils.py`，实现：
1. 加载JSONL数据
2. 中英文分词
3. 构建词表
4. TranslationDataset类

**验证**：能够成功加载一个batch数据并打印shape

### 第三步：RNN模型（22:30-01:00）

创建 `src/models/rnn_seq2seq.py`，实现：
1. Encoder
2. Attention
3. Decoder
4. Seq2Seq完整模型

**验证**：随机输入可以forward，输出shape正确

### 第四步：开始训练（01:00-02:00）

创建 `src/train_rnn.py`，开始训练
- 设置后台训练（nohup或tmux）
- 睡觉前启动训练，明早检查

---

## 🎯 成功标准

### 最低标准（及格线）
- ✅ RNN和Transformer两个模型都能训练并产出翻译
- ✅ inference.py能够运行
- ✅ 有BLEU评估数据（即使不高）
- ✅ 报告结构完整，10页以上
- ✅ GitHub仓库可访问

### 目标标准（良好）
- ✅ 至少完成1-2个对比实验
- ✅ 翻译质量尚可（BLEU > 5）
- ✅ 报告分析有深度，有可视化图表
- ✅ 代码规范，有注释

### 理想标准（优秀）
- ✅ 完成3个以上对比实验
- ✅ 翻译质量较好（BLEU > 10）
- ✅ 报告有深度见解和创新思考
- ✅ 代码质量高，模块化好

---

## 📞 紧急求助资源

1. **PyTorch官方教程**：https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
2. **Annotated Transformer**：http://nlp.seas.harvard.edu/annotated-transformer/
3. **BLEU计算**：nltk.translate.bleu_score
4. **ChatGPT/Claude**：遇到bug时快速调试
5. **GitHub搜索**：搜索"pytorch seq2seq translation"找参考代码

---

## 💪 最后的话

**时间很紧，但完全可行！**

关键原则：
1. **不追求完美**：能跑通最重要
2. **优先核心功能**：保证P0任务完成
3. **及时提交Git**：避免代码丢失
4. **诚实汇报**：报告重过程轻结果
5. **保持冷静**：遇到问题快速寻求解决方案

**立即开始执行！祝你顺利完成作业！🚀**

---

---

## 📋 附录：快速启动清单

### A. 立即创建实验目录结构

```bash
# 在项目根目录执行
mkdir -p experiments/{exp_001_rnn_baseline,exp_002_transformer_baseline,exp_003_rnn_attention_comparison}/{checkpoints,logs,results/figures}

# 创建scripts目录
mkdir -p scripts

echo "✅ 实验目录结构创建完成"
```

### B. 实验配置文件快速生成

```bash
# 生成exp_001配置文件
cat > experiments/exp_001_rnn_baseline/config.yaml << 'EOF'
experiment:
  name: "RNN Baseline (Unidirectional)"
  id: "exp_001"
  description: "单向LSTM + 点积注意力"
  date: "2025-12-26"

model:
  type: "rnn_seq2seq"
  embed_dim: 256
  hidden_dim: 256
  n_layers: 2
  dropout: 0.3
  attention_type: "dot"
  bidirectional: false

data:
  train_file: "AP0004_Midterm&Final_translation_dataset_zh_en/train_10k.jsonl"
  valid_file: "AP0004_Midterm&Final_translation_dataset_zh_en/valid.jsonl"
  test_file: "AP0004_Midterm&Final_translation_dataset_zh_en/test.jsonl"
  max_len: 100
  min_freq: 2

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 15
  grad_clip: 1.0
  teacher_forcing_ratio: 1.0

output:
  checkpoint_dir: "experiments/exp_001_rnn_baseline/checkpoints"
  log_dir: "experiments/exp_001_rnn_baseline/logs"
  results_dir: "experiments/exp_001_rnn_baseline/results"
EOF

# 生成exp_002配置文件
cat > experiments/exp_002_transformer_baseline/config.yaml << 'EOF'
experiment:
  name: "Transformer Baseline"
  id: "exp_002"
  description: "PyTorch内置Transformer + 位置编码"
  date: "2025-12-26"

model:
  type: "transformer"
  d_model: 256
  nhead: 4
  num_encoder_layers: 3
  num_decoder_layers: 3
  dim_feedforward: 1024
  dropout: 0.1

data:
  train_file: "AP0004_Midterm&Final_translation_dataset_zh_en/train_10k.jsonl"
  valid_file: "AP0004_Midterm&Final_translation_dataset_zh_en/valid.jsonl"
  test_file: "AP0004_Midterm&Final_translation_dataset_zh_en/test.jsonl"
  max_len: 100
  min_freq: 2

training:
  batch_size: 64
  learning_rate: 0.0001
  epochs: 15
  grad_clip: 1.0

output:
  checkpoint_dir: "experiments/exp_002_transformer_baseline/checkpoints"
  log_dir: "experiments/exp_002_transformer_baseline/logs"
  results_dir: "experiments/exp_002_transformer_baseline/results"
EOF

echo "✅ 配置文件生成完成"
```

### C. 核心代码文件清单

**必须实现的文件**（按优先级）：

```
1. src/data_utils.py          # P0 - 数据处理（2小时）
   ├── load_data()
   ├── Tokenizer类
   ├── build_vocab()
   └── TranslationDataset

2. src/models/rnn_seq2seq.py  # P0 - RNN模型（3小时）
   ├── Encoder（单向LSTM）
   ├── Attention（dot/additive）
   ├── Decoder
   └── Seq2Seq

3. src/models/transformer.py  # P0 - Transformer（2小时）
   ├── PositionalEncoding
   └── TransformerNMT（使用nn.Transformer）

4. src/train_rnn.py           # P0 - RNN训练（1小时）
5. src/train_transformer.py   # P0 - Transformer训练（1小时）
6. src/evaluate.py            # P0 - 评估系统（1小时）
7. inference.py               # P0 - 一键推理（1小时）

总计核心开发时间：约11小时
```

### D. 每日任务检查表

#### Day 1（今晚）完成度检查
- [ ] requirements.txt创建并安装完成
- [ ] 项目目录结构创建完成
- [ ] 实验配置文件生成完成
- [ ] data_utils.py实现并测试通过
- [ ] rnn_seq2seq.py实现并可以forward
- [ ] train_rnn.py开始训练（后台运行）
- [ ] Git初始化并首次提交

#### Day 2（明天）完成度检查
- [ ] RNN模型训练完成，BLEU评估完成
- [ ] evaluate.py和inference.py实现完成
- [ ] transformer.py实现并测试通过
- [ ] Transformer训练完成，BLEU评估完成
- [ ] 两模型对比数据收集完成
- [ ] 至少1个额外对比实验完成（注意力机制）
- [ ] 报告草稿完成80%
- [ ] 代码上传GitHub

#### Day 3（截止日）完成度检查
- [ ] 报告完成100%（10页以上）
- [ ] 报告命名正确：学号_姓名.pdf
- [ ] 首页包含GitHub URL
- [ ] inference.py测试通过
- [ ] README.md完善
- [ ] 所有代码提交Git
- [ ] **提交到Piazza** ✅

### E. 关键技术决策总结

| 决策点 | 选择 | 原因 |
|--------|------|------|
| RNN编码器方向 | **单向** | 符合作业要求 |
| RNN单元类型 | LSTM或GRU | 都可以，LSTM更常见 |
| 注意力机制 | 先dot，再additive | 循序渐进 |
| Transformer实现 | **nn.Transformer** | 节省时间，稳定可靠 |
| 位置编码 | sin/cos | 标准实现 |
| 训练数据 | **train_10k** | 快速迭代 |
| 评估指标 | BLEU-4 | 作业要求 |
| 解码策略 | 先greedy，再beam | 先保证能跑 |
| 实验管理 | YAML配置 | 结构化、可复现 |

### F. 应急联系资源

1. **PyTorch文档**：https://pytorch.org/docs/stable/index.html
2. **Seq2Seq教程**：https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
3. **Transformer教程**：http://nlp.seas.harvard.edu/annotated-transformer/
4. **BLEU计算**：https://www.nltk.org/api/nltk.translate.bleu_score.html
5. **Jieba分词**：https://github.com/fxsjy/jieba

---

## 🎯 最后的行动呼吁

**现在就开始执行！**

1. ✅ 复制上述bash命令，创建目录结构
2. ✅ 生成实验配置文件
3. ✅ 安装依赖包（pip install -r requirements.txt）
4. ✅ 开始实现data_utils.py（今晚的核心任务）

**记住**：
- ⏰ 时间紧迫，不要追求完美
- 🎯 优先完成P0任务
- 📝 及时提交Git（每完成一个模块）
- 💪 保持冷静，问题总有解决方案

**祝你顺利完成作业！🚀**

---

**文档版本**：v2.0（修订版）  
**最后更新**：2025-12-26 21:00  
**修订内容**：修正RNN方向、简化Transformer实现、新增实验管理系统

