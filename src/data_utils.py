"""
数据处理模块
包含数据加载、分词、词表构建、Dataset类等功能
"""

import json
import os
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

import jieba
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# 特殊token定义
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


def load_data(file_path: str) -> List[Dict]:
    """
    加载JSONL格式数据
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        数据列表，每个元素是一个字典 {"en": "...", "zh": "...", "index": N}
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def clean_text(text: str) -> str:
    """清洗文本：去除多余空白、统一标点等"""
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_zh(text: str) -> List[str]:
    """
    中文分词（使用jieba）
    
    Args:
        text: 中文文本
        
    Returns:
        分词结果列表
    """
    text = clean_text(text)
    tokens = list(jieba.cut(text))
    # 过滤空白token
    tokens = [t for t in tokens if t.strip()]
    return tokens


def tokenize_en(text: str) -> List[str]:
    """
    英文分词（小写 + 按空格分割 + 简单标点处理）
    
    Args:
        text: 英文文本
        
    Returns:
        分词结果列表
    """
    text = clean_text(text).lower()
    # 在标点符号前后添加空格
    text = re.sub(r'([.,!?;:\'\"\(\)\[\]])', r' \1 ', text)
    tokens = text.split()
    # 过滤空白token
    tokens = [t for t in tokens if t.strip()]
    return tokens


def build_vocab(tokens_list: List[List[str]], 
                min_freq: int = 2, 
                max_size: int = 30000) -> Dict[str, int]:
    """
    构建词表
    
    Args:
        tokens_list: 分词结果列表的列表
        min_freq: 最小词频
        max_size: 词表最大大小
        
    Returns:
        词表字典 {token: index}
    """
    # 统计词频
    counter = Counter()
    for tokens in tokens_list:
        counter.update(tokens)
    
    # 初始化词表（特殊token）
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    
    # 按词频排序，添加词汇
    for word, freq in counter.most_common(max_size - len(SPECIAL_TOKENS)):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab


def get_reverse_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    """获取反向词表（index -> token）"""
    return {idx: token for token, idx in vocab.items()}


def tokens_to_ids(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    """将token列表转换为id列表"""
    return [vocab.get(token, UNK_IDX) for token in tokens]


def ids_to_tokens(ids: List[int], reverse_vocab: Dict[int, str], 
                  remove_special: bool = True) -> List[str]:
    """将id列表转换为token列表"""
    tokens = [reverse_vocab.get(idx, UNK_TOKEN) for idx in ids]
    if remove_special:
        # 移除特殊token
        tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
    return tokens


def save_vocab(vocab: Dict[str, int], file_path: str):
    """保存词表到JSON文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(file_path: str) -> Dict[str, int]:
    """从JSON文件加载词表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class TranslationDataset(Dataset):
    """
    翻译数据集
    
    支持双向翻译：en2zh 和 zh2en
    """
    
    def __init__(self, 
                 data: List[Dict],
                 src_vocab: Dict[str, int],
                 tgt_vocab: Dict[str, int],
                 src_lang: str = 'en',
                 tgt_lang: str = 'zh',
                 max_len: int = 100):
        """
        Args:
            data: 数据列表
            src_vocab: 源语言词表
            tgt_vocab: 目标语言词表
            src_lang: 源语言 ('en' 或 'zh')
            tgt_lang: 目标语言 ('en' 或 'zh')
            max_len: 最大序列长度
        """
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        
        # 选择分词器
        self.src_tokenizer = tokenize_en if src_lang == 'en' else tokenize_zh
        self.tgt_tokenizer = tokenize_zh if tgt_lang == 'zh' else tokenize_en
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        
        # 获取源语言和目标语言文本
        src_text = item[self.src_lang]
        tgt_text = item[self.tgt_lang]
        
        # 分词
        src_tokens = self.src_tokenizer(src_text)
        tgt_tokens = self.tgt_tokenizer(tgt_text)
        
        # 截断
        src_tokens = src_tokens[:self.max_len - 2]  # 留出SOS和EOS的位置
        tgt_tokens = tgt_tokens[:self.max_len - 2]
        
        # 转换为id（添加SOS和EOS）
        src_ids = [SOS_IDX] + tokens_to_ids(src_tokens, self.src_vocab) + [EOS_IDX]
        tgt_ids = [SOS_IDX] + tokens_to_ids(tgt_tokens, self.tgt_vocab) + [EOS_IDX]
        
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批处理函数：填充序列到相同长度
    
    Args:
        batch: [(src_ids, tgt_ids), ...]
        
    Returns:
        src_padded: [batch_size, src_max_len]
        tgt_padded: [batch_size, tgt_max_len]
        src_lengths: [batch_size]
        tgt_lengths: [batch_size]
    """
    src_seqs, tgt_seqs = zip(*batch)
    
    # 记录原始长度
    src_lengths = torch.tensor([len(s) for s in src_seqs])
    tgt_lengths = torch.tensor([len(t) for t in tgt_seqs])
    
    # 填充
    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_IDX)
    
    return src_padded, tgt_padded, src_lengths, tgt_lengths


def create_dataloaders(train_data: List[Dict],
                       valid_data: List[Dict],
                       test_data: List[Dict],
                       src_vocab: Dict[str, int],
                       tgt_vocab: Dict[str, int],
                       src_lang: str = 'en',
                       tgt_lang: str = 'zh',
                       batch_size: int = 64,
                       max_len: int = 100,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试DataLoader
    """
    train_dataset = TranslationDataset(train_data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_len)
    valid_dataset = TranslationDataset(valid_data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_len)
    test_dataset = TranslationDataset(test_data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                              collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=collate_fn, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader


def prepare_data(data_dir: str,
                 vocab_dir: str,
                 src_lang: str = 'en',
                 tgt_lang: str = 'zh',
                 train_file: str = 'train_10k.jsonl',
                 valid_file: str = 'valid.jsonl',
                 test_file: str = 'test.jsonl',
                 min_freq: int = 2,
                 max_vocab_size: int = 30000) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, int], Dict[str, int]]:
    """
    准备所有数据：加载数据、构建词表
    
    Returns:
        train_data, valid_data, test_data, src_vocab, tgt_vocab
    """
    # 加载数据
    print(f"Loading data from {data_dir}...")
    train_data = load_data(os.path.join(data_dir, train_file))
    valid_data = load_data(os.path.join(data_dir, valid_file))
    test_data = load_data(os.path.join(data_dir, test_file))
    
    print(f"  Train: {len(train_data)} samples")
    print(f"  Valid: {len(valid_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # 分词
    print("Tokenizing...")
    src_tokenizer = tokenize_en if src_lang == 'en' else tokenize_zh
    tgt_tokenizer = tokenize_zh if tgt_lang == 'zh' else tokenize_en
    
    src_tokens_list = [src_tokenizer(item[src_lang]) for item in train_data]
    tgt_tokens_list = [tgt_tokenizer(item[tgt_lang]) for item in train_data]
    
    # 构建词表
    print("Building vocabulary...")
    src_vocab_path = os.path.join(vocab_dir, f'vocab_{src_lang}.json')
    tgt_vocab_path = os.path.join(vocab_dir, f'vocab_{tgt_lang}.json')
    
    # 检查是否已有词表
    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        print(f"  Loading existing vocabularies...")
        src_vocab = load_vocab(src_vocab_path)
        tgt_vocab = load_vocab(tgt_vocab_path)
    else:
        print(f"  Building new vocabularies...")
        src_vocab = build_vocab(src_tokens_list, min_freq, max_vocab_size)
        tgt_vocab = build_vocab(tgt_tokens_list, min_freq, max_vocab_size)
        
        # 保存词表
        save_vocab(src_vocab, src_vocab_path)
        save_vocab(tgt_vocab, tgt_vocab_path)
    
    print(f"  Source vocab size: {len(src_vocab)}")
    print(f"  Target vocab size: {len(tgt_vocab)}")
    
    return train_data, valid_data, test_data, src_vocab, tgt_vocab


def create_padding_mask(seq: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """
    创建padding mask
    
    Args:
        seq: [batch_size, seq_len]
        
    Returns:
        mask: [batch_size, seq_len], True表示padding位置
    """
    return seq == pad_idx


if __name__ == '__main__':
    # 测试代码
    print("=" * 50)
    print("Testing data_utils module")
    print("=" * 50)
    
    # 测试数据加载
    data_dir = 'AP0004_Midterm&Final_translation_dataset_zh_en'
    vocab_dir = 'data/vocab'
    
    print("\n1. Testing data loading...")
    train_data = load_data(os.path.join(data_dir, 'train_10k.jsonl'))
    print(f"   Loaded {len(train_data)} training samples")
    print(f"   Sample: {train_data[0]}")
    
    # 测试分词
    print("\n2. Testing tokenization...")
    sample_en = train_data[0]['en']
    sample_zh = train_data[0]['zh']
    print(f"   EN: {sample_en}")
    print(f"   EN tokens: {tokenize_en(sample_en)}")
    print(f"   ZH: {sample_zh}")
    print(f"   ZH tokens: {tokenize_zh(sample_zh)}")
    
    # 测试词表构建
    print("\n3. Testing vocabulary building...")
    en_tokens_list = [tokenize_en(item['en']) for item in train_data[:1000]]
    zh_tokens_list = [tokenize_zh(item['zh']) for item in train_data[:1000]]
    
    en_vocab = build_vocab(en_tokens_list, min_freq=2, max_size=10000)
    zh_vocab = build_vocab(zh_tokens_list, min_freq=2, max_size=10000)
    print(f"   EN vocab size: {len(en_vocab)}")
    print(f"   ZH vocab size: {len(zh_vocab)}")
    print(f"   Special tokens: {list(en_vocab.items())[:4]}")
    
    # 测试Dataset
    print("\n4. Testing TranslationDataset...")
    dataset = TranslationDataset(train_data[:100], en_vocab, zh_vocab, 'en', 'zh', max_len=50)
    src, tgt = dataset[0]
    print(f"   Sample src shape: {src.shape}")
    print(f"   Sample tgt shape: {tgt.shape}")
    print(f"   Sample src ids: {src[:10].tolist()}")
    print(f"   Sample tgt ids: {tgt[:10].tolist()}")
    
    # 测试DataLoader
    print("\n5. Testing DataLoader...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    src_batch, tgt_batch, src_lens, tgt_lens = next(iter(loader))
    print(f"   Batch src shape: {src_batch.shape}")
    print(f"   Batch tgt shape: {tgt_batch.shape}")
    print(f"   Src lengths: {src_lens.tolist()}")
    print(f"   Tgt lengths: {tgt_lens.tolist()}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
