"""
T5/mT5 模型微调训练脚本
用于中英翻译任务

主要特性:
- 支持 T5 和 mT5 系列模型
- 混合精度训练 (FP16) 加速
- 梯度累积支持更大有效批次
- 早停机制防止过拟合
- 详细的训练日志和进度条
- 保存训练配置便于复现
- 支持中文分词的 BLEU 计算

使用示例:
    python src/train_t5.py \\
        --train_file AP0004_Midterm_Final_translation_dataset_zh_en/train_10k.jsonl \\
        --valid_file AP0004_Midterm_Final_translation_dataset_zh_en/valid.jsonl \\
        --model_name ./models/mt5-small \\
        --epochs 3 \\
        --batch_size 8 \\
        --output_dir experiments/t5_finetune
"""

import os
import sys
import json
import argparse
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

try:
    from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Error: transformers library not installed.")
    print("Install with: pip install transformers sentencepiece")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.t5_finetune import (
    T5Translator, 
    create_t5_dataloader,
    calculate_bleu_chinese,
    clean_text
)


def set_seed(seed: int):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(filepath: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def evaluate_model(model: T5Translator, 
                   data: List[Dict], 
                   src_lang: str = 'en',
                   tgt_lang: str = 'zh',
                   batch_size: int = 16,
                   max_samples: int = 500,
                   num_beams: int = 4,
                   show_examples: int = 3) -> Tuple[float, List[Dict]]:
    """
    评估模型性能
    
    Args:
        model: T5Translator 模型
        data: 验证数据
        src_lang: 源语言
        tgt_lang: 目标语言
        batch_size: 批次大小
        max_samples: 最大评估样本数
        num_beams: 束搜索大小
        show_examples: 显示的翻译示例数量
        
    Returns:
        (BLEU分数, 翻译示例列表)
    """
    # 限制评估样本数
    eval_data = data[:max_samples]
    
    # 收集源文本和参考翻译
    src_texts = [clean_text(item[src_lang]) for item in eval_data]
    references = [clean_text(item[tgt_lang]) for item in eval_data]
    
    # 生成翻译
    model.model.eval()
    translations = model.translate_batch(
        src_texts, 
        batch_size=batch_size,
        num_beams=num_beams,
        max_length=256,
        min_length=5,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        show_progress=HAS_TQDM
    )
    
    # 计算 BLEU
    # 根据目标语言选择是否进行中文分词
    tokenize_chinese = (tgt_lang == 'zh')
    bleu_score = calculate_bleu_chinese(translations, references, tokenize=tokenize_chinese)
    
    # 收集翻译示例
    examples = []
    for i in range(min(show_examples, len(eval_data))):
        examples.append({
            'source': src_texts[i],
            'reference': references[i],
            'prediction': translations[i]
        })
    
    return bleu_score, examples


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.1, mode: str = 'max'):
        """
        Args:
            patience: 容忍的 epoch 数量
            min_delta: 最小改进量
            mode: 'max' (越大越好) 或 'min' (越小越好)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前分数
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_epoch(model: T5Translator,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler,
                device: torch.device,
                gradient_accumulation_steps: int = 1,
                max_grad_norm: float = 1.0,
                use_amp: bool = False,
                scaler=None,
                log_interval: int = 50) -> Dict:
    """
    训练一个 epoch
    
    Args:
        model: T5Translator 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        gradient_accumulation_steps: 梯度累积步数
        max_grad_norm: 最大梯度范数
        use_amp: 是否使用混合精度
        scaler: GradScaler (用于 AMP)
        log_interval: 日志间隔
        
    Returns:
        训练统计信息字典
    """
    model.model.train()
    
    total_loss = 0
    num_steps = 0
    
    # 进度条
    if HAS_TQDM:
        pbar = tqdm(train_loader, desc="Training", leave=False)
    else:
        pbar = train_loader
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        # 移动数据到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        decoder_attention_mask = batch.get('decoder_attention_mask')
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(device)
        
        # 前向传播（可选混合精度）
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    decoder_attention_mask=decoder_attention_mask
                )
                loss = outputs.loss / gradient_accumulation_steps
            
            # 反向传播
            scaler.scale(loss).backward()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_steps += 1
        
        # 梯度累积
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
            
            nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=max_grad_norm)
            
            # 更新参数
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        # 更新进度条
        if HAS_TQDM:
            avg_loss = total_loss / num_steps
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})
        elif (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_steps
            lr = scheduler.get_last_lr()[0]
            print(f"  Step {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
    
    return {
        'loss': total_loss / num_steps,
        'learning_rate': scheduler.get_last_lr()[0]
    }


def train_t5(args):
    """T5 微调训练主函数"""
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"T5 Fine-tuning for Machine Translation")
    print(f"{'='*60}")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存训练配置
    config = vars(args)
    config['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config['device'] = str(device)
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nTraining config saved to {config_path}")
    
    # 加载数据
    print("\n--- Loading Data ---")
    train_data = load_jsonl(args.train_file)
    valid_data = load_jsonl(args.valid_file)
    
    # 限制训练数据量
    if args.max_train_samples:
        train_data = train_data[:args.max_train_samples]
    
    print(f"  Training samples: {len(train_data):,}")
    print(f"  Validation samples: {len(valid_data):,}")
    
    # 创建模型
    print(f"\n--- Creating Model ---")
    print(f"  Model: {args.model_name}")
    model = T5Translator(
        model_name=args.model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        device=device,
        use_prefix=True
    )
    
    # 可选：冻结编码器
    if args.freeze_encoder:
        print("  Freezing encoder parameters...")
        model.freeze_encoder()
    
    # 创建数据加载器
    print(f"\n--- Creating DataLoaders ---")
    train_loader = create_t5_dataloader(
        train_data,
        model.tokenizer,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        batch_size=args.batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        shuffle=True,
        num_workers=args.num_workers,
        dynamic_padding=args.dynamic_padding
    )
    print(f"  Training batches: {len(train_loader):,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # 设置优化器
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    
    # 设置学习率调度器
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    if args.scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:  # cosine
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    print(f"\n--- Training Configuration ---")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Scheduler: {args.scheduler_type}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Weight decay: {args.weight_decay}")
    
    # 混合精度训练
    use_amp = args.fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print(f"  Mixed precision: FP16")
    
    # 早停
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=0.1,
        mode='max'
    ) if args.early_stopping else None
    
    # 训练历史
    history = {
        'train_loss': [],
        'valid_bleu': [],
        'learning_rate': [],
        'epoch_time': []
    }
    
    # 最佳模型
    best_bleu = 0.0
    best_epoch = 0
    
    print(f"\n{'='*60}")
    print("Starting Training...")
    print(f"{'='*60}")
    
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # 训练
        train_stats = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            use_amp=use_amp,
            scaler=scaler,
            log_interval=args.log_interval
        )
        
        # 评估
        print("\nEvaluating...")
        valid_bleu, examples = evaluate_model(
            model, valid_data,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            batch_size=args.eval_batch_size,
            max_samples=args.max_eval_samples,
            num_beams=args.num_beams,
            show_examples=3
        )
        
        # 记录历史
        epoch_time = time.time() - epoch_start
        history['train_loss'].append(train_stats['loss'])
        history['valid_bleu'].append(valid_bleu)
        history['learning_rate'].append(train_stats['learning_rate'])
        history['epoch_time'].append(epoch_time)
        
        # 打印结果
        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"    Train Loss: {train_stats['loss']:.4f}")
        print(f"    Valid BLEU: {valid_bleu:.2f}")
        print(f"    Time: {epoch_time/60:.1f} min")
        
        # 显示翻译示例
        if examples:
            print(f"\n  Translation Examples:")
            for i, ex in enumerate(examples[:2]):
                print(f"    [{i+1}] Source: {ex['source'][:50]}...")
                print(f"        Reference: {ex['reference'][:50]}...")
                print(f"        Prediction: {ex['prediction'][:50]}...")
        
        # 保存最佳模型
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            best_epoch = epoch + 1
            best_path = os.path.join(args.output_dir, 'best_model')
            model.save_model(best_path)
            print(f"\n  ✓ New best model! BLEU: {best_bleu:.2f}")
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}')
            model.save_model(checkpoint_path)
        
        # 保存训练历史
        history_path = os.path.join(args.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # 早停检查
        if early_stopping is not None:
            if early_stopping(valid_bleu):
                print(f"\n  Early stopping triggered at epoch {epoch+1}")
                break
    
    # 训练完成
    total_time = time.time() - training_start_time
    
    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"{'='*60}")
    print(f"\n  Total time: {total_time/60:.1f} minutes")
    print(f"  Best BLEU: {best_bleu:.2f} (Epoch {best_epoch})")
    print(f"  Best model: {os.path.join(args.output_dir, 'best_model')}")
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, 'final_model')
    model.save_model(final_path)
    print(f"  Final model: {final_path}")
    
    # 保存最终训练历史
    history['total_time'] = total_time
    history['best_bleu'] = best_bleu
    history['best_epoch'] = best_epoch
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Training history: {history_path}")
    
    return model, history


def test_translation(model: T5Translator, test_file: str, 
                     src_lang: str, tgt_lang: str,
                     num_samples: int = 10, num_beams: int = 4):
    """测试翻译效果"""
    test_data = load_jsonl(test_file)[:num_samples]
    
    print("\n" + "="*60)
    print("Translation Examples:")
    print("="*60)
    
    for i, item in enumerate(test_data):
        src_text = clean_text(item[src_lang])
        ref_text = clean_text(item[tgt_lang])
        pred_text = model.translate(src_text, num_beams=num_beams)
        
        print(f"\n[{i+1}]")
        print(f"  Source ({src_lang}): {src_text}")
        print(f"  Reference ({tgt_lang}): {ref_text}")
        print(f"  Prediction: {pred_text}")


def main():
    parser = argparse.ArgumentParser(
        description='T5/mT5 Fine-tuning for Machine Translation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 使用本地 mT5-small 模型进行英译中（推荐）
    python src/train_t5.py \\
        --train_file data/train_10k.jsonl \\
        --valid_file data/valid.jsonl \\
        --model_name ./models/mt5-small \\
        --epochs 3

    # 使用混合精度和梯度累积
    python src/train_t5.py \\
        --train_file data/train_10k.jsonl \\
        --valid_file data/valid.jsonl \\
        --model_name ./models/mt5-small \\
        --fp16 \\
        --gradient_accumulation_steps 4
        """
    )
    
    # 数据参数
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--train_file', type=str, required=True,
                           help='Path to training data (JSONL)')
    data_group.add_argument('--valid_file', type=str, required=True,
                           help='Path to validation data (JSONL)')
    data_group.add_argument('--test_file', type=str, default=None,
                           help='Path to test data (JSONL)')
    data_group.add_argument('--max_train_samples', type=int, default=None,
                           help='Maximum training samples (for debugging)')
    data_group.add_argument('--max_eval_samples', type=int, default=500,
                           help='Maximum evaluation samples')
    
    # 模型参数
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--model_name', type=str, default='./models/mt5-small',
                            help='Model name or local path (e.g., ./models/mt5-small, google/mt5-small)')
    model_group.add_argument('--src_lang', type=str, default='en',
                            choices=['en', 'zh'],
                            help='Source language')
    model_group.add_argument('--tgt_lang', type=str, default='zh',
                            choices=['en', 'zh'],
                            help='Target language')
    model_group.add_argument('--max_src_len', type=int, default=128,
                            help='Maximum source sequence length')
    model_group.add_argument('--max_tgt_len', type=int, default=128,
                            help='Maximum target sequence length')
    model_group.add_argument('--freeze_encoder', action='store_true',
                            help='Freeze encoder parameters')
    
    # 训练参数
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=3,
                            help='Number of training epochs')
    train_group.add_argument('--batch_size', type=int, default=8,
                            help='Training batch size')
    train_group.add_argument('--eval_batch_size', type=int, default=16,
                            help='Evaluation batch size')
    train_group.add_argument('--learning_rate', type=float, default=3e-5,
                            help='Learning rate')
    train_group.add_argument('--weight_decay', type=float, default=0.01,
                            help='Weight decay')
    train_group.add_argument('--warmup_ratio', type=float, default=0.1,
                            help='Warmup ratio')
    train_group.add_argument('--scheduler_type', type=str, default='linear',
                            choices=['linear', 'cosine'],
                            help='Learning rate scheduler type')
    train_group.add_argument('--max_grad_norm', type=float, default=1.0,
                            help='Maximum gradient norm')
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help='Gradient accumulation steps')
    train_group.add_argument('--fp16', action='store_true',
                            help='Use mixed precision training')
    train_group.add_argument('--dynamic_padding', action='store_true',
                            help='Use dynamic padding')
    train_group.add_argument('--num_workers', type=int, default=0,
                            help='DataLoader workers')
    train_group.add_argument('--seed', type=int, default=42,
                            help='Random seed')
    
    # 早停参数
    early_group = parser.add_argument_group('Early Stopping')
    early_group.add_argument('--early_stopping', action='store_true',
                            help='Enable early stopping')
    early_group.add_argument('--early_stopping_patience', type=int, default=3,
                            help='Early stopping patience')
    
    # 解码参数
    decode_group = parser.add_argument_group('Decoding')
    decode_group.add_argument('--num_beams', type=int, default=4,
                             help='Beam search size for evaluation')
    
    # 输出参数
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output_dir', type=str, default='experiments/t5_finetune',
                             help='Output directory')
    output_group.add_argument('--log_interval', type=int, default=50,
                             help='Log interval (batches)')
    output_group.add_argument('--save_interval', type=int, default=1,
                             help='Checkpoint save interval (epochs)')
    
    args = parser.parse_args()
    
    # 训练模型
    model, history = train_t5(args)
    
    # 测试翻译
    if args.test_file:
        test_translation(
            model, args.test_file,
            args.src_lang, args.tgt_lang,
            num_samples=10, num_beams=args.num_beams
        )


if __name__ == '__main__':
    main()
