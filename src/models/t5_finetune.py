"""
T5 预训练模型微调模块
使用 Hugging Face transformers 库

支持的模型:
- t5-small, t5-base: 英语 T5 模型
- google/mt5-small, google/mt5-base: 多语言 mT5 模型（推荐用于中英翻译）

注意: 对于中英翻译任务，推荐使用 mT5 模型，因为原始 T5 主要针对英语训练。
"""

import os
import sys
import json
import re
from typing import List, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import (
        T5ForConditionalGeneration, 
        T5Tokenizer,
        MT5ForConditionalGeneration,
        MT5Tokenizer,
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        T5Config
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers library not installed. T5 model will not be available.")

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 中文分词（用于 BLEU 计算）
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False


def tokenize_chinese(text: str) -> List[str]:
    """中文分词"""
    if HAS_JIEBA:
        return list(jieba.cut(text))
    else:
        # 按字符分割
        return list(text)


def clean_text(text: str) -> str:
    """清洗文本"""
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class T5TranslationDataset(Dataset):
    """
    T5 翻译数据集
    将翻译任务格式化为 T5 的输入格式
    """
    
    def __init__(self,
                 data: List[Dict],
                 tokenizer,
                 src_lang: str = 'en',
                 tgt_lang: str = 'zh',
                 max_src_len: int = 128,
                 max_tgt_len: int = 128,
                 use_prefix: bool = True):
        """
        Args:
            data: 数据列表 [{"en": "...", "zh": "...", "index": N}, ...]
            tokenizer: T5/MT5 Tokenizer
            src_lang: 源语言
            tgt_lang: 目标语言
            max_src_len: 源序列最大长度
            max_tgt_len: 目标序列最大长度
            use_prefix: 是否使用任务前缀
        """
        self.data = data
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.use_prefix = use_prefix
        
        # 设置任务前缀
        if use_prefix:
            if src_lang == 'en' and tgt_lang == 'zh':
                self.prefix = "translate English to Chinese: "
            elif src_lang == 'zh' and tgt_lang == 'en':
                self.prefix = "translate Chinese to English: "
            else:
                self.prefix = f"translate {src_lang} to {tgt_lang}: "
        else:
            self.prefix = ""
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        src_text = clean_text(str(item[self.src_lang]))
        tgt_text = clean_text(str(item[self.tgt_lang]))
        
        # 添加任务前缀
        input_text = self.prefix + src_text
        
        # 编码输入
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 编码目标
        # 注意：mT5 不需要 as_target_tokenizer()，因为它使用统一的多语言tokenizer
        target_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_tgt_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 处理标签：将 padding token 设为 -100（忽略计算损失）
        # 注意：T5模型会自动处理decoder input的右移，所以直接使用target_ids作为labels
        labels = target_encoding['input_ids'].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # 重要：确保EOS token不被mask掉
        # T5需要学习何时结束生成
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels,
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze()
        }


class DynamicBatchDataset(Dataset):
    """
    支持动态批次的数据集
    按长度分组以减少 padding
    """
    
    def __init__(self,
                 data: List[Dict],
                 tokenizer,
                 src_lang: str = 'en',
                 tgt_lang: str = 'zh',
                 max_src_len: int = 128,
                 max_tgt_len: int = 128,
                 use_prefix: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.use_prefix = use_prefix
        
        # 设置任务前缀
        if use_prefix:
            if src_lang == 'en' and tgt_lang == 'zh':
                self.prefix = "translate English to Chinese: "
            elif src_lang == 'zh' and tgt_lang == 'en':
                self.prefix = "translate Chinese to English: "
            else:
                self.prefix = f"translate {src_lang} to {tgt_lang}: "
        else:
            self.prefix = ""
        
        # 预计算长度并排序
        self._precompute_lengths()
    
    def _precompute_lengths(self):
        """预计算每个样本的长度"""
        self.lengths = []
        for item in self.data:
            src_text = self.prefix + clean_text(str(item[self.src_lang]))
            # 估计 token 数量（简化版本）
            src_len = len(self.tokenizer.encode(src_text, add_special_tokens=False))
            self.lengths.append(min(src_len, self.max_src_len))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        src_text = clean_text(str(item[self.src_lang]))
        tgt_text = clean_text(str(item[self.tgt_lang]))
        
        input_text = self.prefix + src_text
        
        # 编码时不 padding，后续在 collate_fn 中处理
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_src_len,
            truncation=True,
            return_tensors='pt'
        )
        
        # 编码目标
        # 注意：mT5 不需要 as_target_tokenizer()
        target_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_tgt_len,
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids'].squeeze().clone()
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels,
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze()
        }


def dynamic_collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    动态 padding 的 collate 函数
    
    Args:
        batch: 批次数据列表
        pad_token_id: padding token ID
        
    Returns:
        批次字典
    """
    # 获取最大长度
    max_input_len = max(item['input_ids'].size(0) for item in batch)
    max_label_len = max(item['labels'].size(0) for item in batch)
    
    batch_size = len(batch)
    
    # 初始化张量
    input_ids = torch.full((batch_size, max_input_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    labels = torch.full((batch_size, max_label_len), -100, dtype=torch.long)
    decoder_attention_mask = torch.zeros(batch_size, max_label_len, dtype=torch.long)
    
    # 填充数据
    for i, item in enumerate(batch):
        input_len = item['input_ids'].size(0)
        label_len = item['labels'].size(0)
        
        input_ids[i, :input_len] = item['input_ids']
        attention_mask[i, :input_len] = item['attention_mask']
        labels[i, :label_len] = item['labels']
        decoder_attention_mask[i, :label_len] = item['decoder_attention_mask']
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'decoder_attention_mask': decoder_attention_mask
    }


class T5Translator(nn.Module):
    """
    T5/mT5 翻译模型封装
    
    支持:
    - T5 系列模型 (t5-small, t5-base, t5-large)
    - mT5 系列模型 (google/mt5-small, google/mt5-base) - 推荐用于中英翻译
    """
    
    # 支持的模型列表
    SUPPORTED_MODELS = {
        't5-small': ('T5', 60_000_000),
        't5-base': ('T5', 220_000_000),
        't5-large': ('T5', 770_000_000),
        'google/mt5-small': ('mT5', 300_000_000),
        'google/mt5-base': ('mT5', 580_000_000),
    }
    
    def __init__(self,
                 model_name: str = './models/mt5-small',
                 src_lang: str = 'en',
                 tgt_lang: str = 'zh',
                 device: torch.device = None,
                 use_prefix: bool = True,
                 load_in_8bit: bool = False):
        """
        Args:
            model_name: 模型名称
            src_lang: 源语言
            tgt_lang: 目标语言
            device: 设备
            use_prefix: 是否使用任务前缀
            load_in_8bit: 是否使用 8-bit 量化加载（节省显存）
        """
        super().__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required for T5 model. "
                            "Install with: pip install transformers sentencepiece")
        
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_prefix = use_prefix
        
        # 判断模型类型
        self.is_mt5 = 'mt5' in model_name.lower()
        
        # 加载模型和分词器
        print(f"Loading {'mT5' if self.is_mt5 else 'T5'} model: {model_name}")
        print(f"  This may take a moment...")
        
        try:
            # 使用 Auto 类自动选择正确的模型类型
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model with AutoModel, trying specific class: {e}")
            if self.is_mt5:
                self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
                self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
            else:
                self.tokenizer = T5Tokenizer.from_pretrained(model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        self.model.to(self.device)
        
        # 设置任务前缀
        if use_prefix:
            if src_lang == 'en' and tgt_lang == 'zh':
                self.prefix = "translate English to Chinese: "
            elif src_lang == 'zh' and tgt_lang == 'en':
                self.prefix = "translate Chinese to English: "
            else:
                self.prefix = f"translate {src_lang} to {tgt_lang}: "
        else:
            self.prefix = ""
        
        print(f"  ✓ Model loaded successfully!")
        print(f"  Total parameters: {self.count_parameters():,}")
        print(f"  Device: {self.device}")
    
    def count_parameters(self) -> int:
        """统计模型参数数量"""
        return sum(p.numel() for p in self.model.parameters())
    
    def count_trainable_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def freeze_encoder(self):
        """冻结编码器参数（用于参数高效微调）"""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        print(f"Encoder frozen. Trainable parameters: {self.count_trainable_parameters():,}")
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.model.parameters():
            param.requires_grad = True
        print(f"All parameters unfrozen. Trainable parameters: {self.count_trainable_parameters():,}")
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None) -> Dict:
        """
        前向传播
        
        Args:
            input_ids: [batch, src_len]
            attention_mask: [batch, src_len]
            labels: [batch, tgt_len] (训练时使用)
            decoder_attention_mask: [batch, tgt_len]
            
        Returns:
            模型输出（包含 loss 和 logits）
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return outputs
    
    def translate(self,
                  text: str,
                  max_length: int = 128,
                  num_beams: int = 4,
                  length_penalty: float = 1.0,
                  no_repeat_ngram_size: int = 2,
                  **kwargs) -> str:
        """
        翻译单个句子
        
        Args:
            text: 输入文本
            max_length: 最大生成长度
            num_beams: 束搜索大小
            length_penalty: 长度惩罚因子
            no_repeat_ngram_size: 禁止重复的 n-gram 大小
            **kwargs: 其他生成参数
            
        Returns:
            翻译结果
        """
        self.model.eval()
        
        # 清洗并添加任务前缀
        text = clean_text(text)
        input_text = self.prefix + text
        
        # 编码
        inputs = self.tokenizer(
            input_text, 
            return_tensors='pt',
            max_length=max_length,
            truncation=True
        ).to(self.device)
        
        # 生成
        # 设置生成参数，避免生成 <extra_id_X> 等特殊token
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
                # 重要：设置强制开始token为pad token（对于mT5）
                forced_bos_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        # 解码
        translation = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return translation
    
    def translate_batch(self,
                        texts: List[str],
                        max_length: int = 128,
                        num_beams: int = 4,
                        batch_size: int = 8,
                        length_penalty: float = 1.0,
                        no_repeat_ngram_size: int = 2,
                        show_progress: bool = False,
                        **kwargs) -> List[str]:
        """
        批量翻译
        
        Args:
            texts: 输入文本列表
            max_length: 最大生成长度
            num_beams: 束搜索大小
            batch_size: 批次大小
            length_penalty: 长度惩罚因子
            no_repeat_ngram_size: 禁止重复的 n-gram 大小
            show_progress: 是否显示进度
            **kwargs: 其他生成参数
            
        Returns:
            翻译结果列表
        """
        self.model.eval()
        translations = []
        
        # 准备迭代器
        n_batches = (len(texts) + batch_size - 1) // batch_size
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(texts), batch_size), total=n_batches, desc="Translating")
            except ImportError:
                iterator = range(0, len(texts), batch_size)
        else:
            iterator = range(0, len(texts), batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # 清洗并添加任务前缀
            input_texts = [self.prefix + clean_text(text) for text in batch_texts]
            
            # 编码
            inputs = self.tokenizer(
                input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # 生成
            # 设置生成参数，避免生成 <extra_id_X> 等特殊token
            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=True,
                    # 重要：设置强制开始token为pad token（对于mT5）
                    forced_bos_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )
            
            # 解码
            batch_translations = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            translations.extend(batch_translations)
        
        return translations
    
    def save_model(self, save_path: str, save_config: bool = True):
        """
        保存模型
        
        Args:
            save_path: 保存路径
            save_config: 是否保存配置信息
        """
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型和分词器
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # 保存配置信息
        if save_config:
            config = {
                'model_name': self.model_name,
                'src_lang': self.src_lang,
                'tgt_lang': self.tgt_lang,
                'use_prefix': self.use_prefix,
                'prefix': self.prefix,
                'is_mt5': self.is_mt5
            }
            config_path = os.path.join(save_path, 'translator_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        加载模型
        
        Args:
            load_path: 模型路径
        """
        # 加载配置
        config_path = os.path.join(load_path, 'translator_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.src_lang = config.get('src_lang', self.src_lang)
            self.tgt_lang = config.get('tgt_lang', self.tgt_lang)
            self.use_prefix = config.get('use_prefix', self.use_prefix)
            self.prefix = config.get('prefix', self.prefix)
            self.is_mt5 = config.get('is_mt5', self.is_mt5)
        
        # 加载模型和分词器
        self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        
        print(f"Model loaded from {load_path}")
    
    @classmethod
    def from_pretrained(cls, 
                        load_path: str, 
                        device: torch.device = None) -> 'T5Translator':
        """
        从预训练目录加载模型
        
        Args:
            load_path: 模型路径
            device: 设备
            
        Returns:
            T5Translator 实例
        """
        # 读取配置
        config_path = os.path.join(load_path, 'translator_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            model_name = config.get('model_name', './models/mt5-small')
            src_lang = config.get('src_lang', 'en')
            tgt_lang = config.get('tgt_lang', 'zh')
            use_prefix = config.get('use_prefix', True)
        else:
            model_name = './models/mt5-small'
            src_lang = 'en'
            tgt_lang = 'zh'
            use_prefix = True
        
        # 创建实例（使用本地路径作为模型名）
        translator = cls(
            model_name=load_path,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=device,
            use_prefix=use_prefix
        )
        
        return translator


def create_t5_model(model_name: str = './models/mt5-small',
                    src_lang: str = 'en',
                    tgt_lang: str = 'zh',
                    device: torch.device = None,
                    use_prefix: bool = True) -> T5Translator:
    """
    创建 T5 翻译模型
    
    推荐:
    - 中英翻译使用 'google/mt5-small' 或 'google/mt5-base'
    - 纯英语任务可使用 't5-small' 或 't5-base'
    
    Args:
        model_name: T5 模型名称
        src_lang: 源语言
        tgt_lang: 目标语言
        device: 设备
        use_prefix: 是否使用任务前缀
        
    Returns:
        T5Translator 模型
    """
    return T5Translator(
        model_name=model_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=device,
        use_prefix=use_prefix
    )


def create_t5_dataloader(data: List[Dict],
                         tokenizer,
                         src_lang: str = 'en',
                         tgt_lang: str = 'zh',
                         batch_size: int = 8,
                         max_src_len: int = 128,
                         max_tgt_len: int = 128,
                         shuffle: bool = True,
                         use_prefix: bool = True,
                         num_workers: int = 0,
                         dynamic_padding: bool = False) -> DataLoader:
    """
    创建 T5 数据加载器
    
    Args:
        data: 数据列表
        tokenizer: T5/MT5 Tokenizer
        src_lang: 源语言
        tgt_lang: 目标语言
        batch_size: 批次大小
        max_src_len: 源序列最大长度
        max_tgt_len: 目标序列最大长度
        shuffle: 是否打乱数据
        use_prefix: 是否使用任务前缀
        num_workers: 数据加载线程数
        dynamic_padding: 是否使用动态 padding
        
    Returns:
        DataLoader
    """
    if dynamic_padding:
        dataset = DynamicBatchDataset(
            data, tokenizer, src_lang, tgt_lang, max_src_len, max_tgt_len, use_prefix
        )
        collate_fn = lambda batch: dynamic_collate_fn(batch, tokenizer.pad_token_id)
    else:
        dataset = T5TranslationDataset(
            data, tokenizer, src_lang, tgt_lang, max_src_len, max_tgt_len, use_prefix
        )
        collate_fn = None
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )


def calculate_bleu_chinese(predictions: List[str], 
                           references: List[str],
                           tokenize: bool = True) -> float:
    """
    计算中文 BLEU 分数（使用分词）
    
    SacreBLEU 格式说明:
    - predictions: [hyp1, hyp2, ...] 假设翻译列表
    - references: [ref1, ref2, ...] 对应的参考翻译列表
    - SacreBLEU 期望 references 格式为 [[ref1, ref2, ...]] (外层是参考数量，内层对应假设)
    
    Args:
        predictions: 预测翻译列表
        references: 参考翻译列表（与预测一一对应）
        tokenize: 是否进行中文分词
        
    Returns:
        BLEU-4 分数
    """
    try:
        import sacrebleu
    except ImportError:
        print("sacrebleu not installed")
        return 0.0
    
    if len(predictions) != len(references):
        print(f"Warning: predictions ({len(predictions)}) and references ({len(references)}) have different lengths")
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
    
    if tokenize:
        # 对中文进行分词，使用空格连接
        tokenized_preds = [' '.join(tokenize_chinese(p)) for p in predictions]
        tokenized_refs = [' '.join(tokenize_chinese(r)) for r in references]
    else:
        tokenized_preds = predictions
        tokenized_refs = references
    
    # SacreBLEU 格式: references 是 [[ref1, ref2, ...]]
    # 这里只有一个参考，所以是 [tokenized_refs]
    bleu = sacrebleu.corpus_bleu(tokenized_preds, [tokenized_refs])
    return bleu.score


if __name__ == '__main__':
    # 测试代码
    if not HAS_TRANSFORMERS:
        print("transformers library not installed. Skipping tests.")
        exit(0)
    
    print("="*60)
    print("Testing T5 Translation Model")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 测试使用 mT5-small（推荐用于中英翻译）
    print("\n1. Testing mT5-small model...")
    try:
        model = create_t5_model(
            model_name='google/mt5-small',
            src_lang='en',
            tgt_lang='zh',
            device=device
        )
        
        print(f"\n  Total parameters: {model.count_parameters():,}")
        print(f"  Trainable parameters: {model.count_trainable_parameters():,}")
        
        # 测试翻译
        print("\n2. Testing translation...")
        test_texts = [
            "Hello, how are you?",
            "The weather is nice today.",
            "I love machine learning."
        ]
        
        for text in test_texts:
            translation = model.translate(text)
            print(f"  EN: {text}")
            print(f"  ZH: {translation}")
            print()
        
        # 测试批量翻译
        print("3. Testing batch translation...")
        translations = model.translate_batch(test_texts, batch_size=2)
        print(f"  Translated {len(translations)} sentences")
        
        # 测试数据集
        print("\n4. Testing T5TranslationDataset...")
        sample_data = [
            {"en": "Hello world", "zh": "你好世界", "index": 0},
            {"en": "Good morning", "zh": "早上好", "index": 1}
        ]
        dataset = T5TranslationDataset(sample_data, model.tokenizer, 'en', 'zh')
        sample = dataset[0]
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
        
        # 测试数据加载器
        print("\n5. Testing DataLoader...")
        loader = create_t5_dataloader(sample_data, model.tokenizer, batch_size=2)
        batch = next(iter(loader))
        print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"  Batch labels shape: {batch['labels'].shape}")
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
