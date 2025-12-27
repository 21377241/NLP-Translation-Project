"""
RNN Seq2Seq 模型训练脚本
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import (
    prepare_data, create_dataloaders,
    PAD_IDX, SOS_IDX, EOS_IDX,
    get_reverse_vocab, ids_to_tokens
)
from src.models.rnn_seq2seq import create_rnn_model, count_parameters


def train_epoch(model, train_loader, optimizer, criterion, clip, device, teacher_forcing_ratio=1.0):
    """
    训练一个epoch
    
    Returns:
        平均loss
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        src, tgt, src_lens, tgt_lens = batch
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt, src_lens, teacher_forcing_ratio)
        
        # 计算loss (忽略第一个token <SOS>)
        # output: [batch, tgt_len-1, vocab_size]
        # tgt: [batch, tgt_len]
        output = output.reshape(-1, output.size(-1))  # [batch*(tgt_len-1), vocab_size]
        tgt = tgt[:, 1:].reshape(-1)  # [batch*(tgt_len-1)]
        
        loss = criterion(output, tgt)
        
        # Backward pass
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def evaluate(model, valid_loader, criterion, device):
    """
    验证
    
    Returns:
        平均loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in valid_loader:
            src, tgt, src_lens, tgt_lens = batch
            src, tgt = src.to(device), tgt.to(device)
            
            # Forward pass (不使用teacher forcing)
            output = model(src, tgt, src_lens, teacher_forcing_ratio=0.0)
            
            output = output.reshape(-1, output.size(-1))
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            total_loss += loss.item()
    
    return total_loss / len(valid_loader)


def translate_sample(model, src, src_vocab_rev, tgt_vocab_rev, device, max_len=50, repetition_penalty=1.5):
    """翻译单个样本（带重复惩罚）"""
    model.eval()
    src_tensor = src.unsqueeze(0).to(device)
    
    with torch.no_grad():
        translation = model.translate(src_tensor, max_len=max_len, repetition_penalty=repetition_penalty)
    
    translation = translation[0].cpu().tolist()
    translated_tokens = ids_to_tokens(translation, tgt_vocab_rev)
    
    return translated_tokens


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 选择训练文件
    train_file = 'train_10k.jsonl' if args.use_small else 'train_100k.jsonl'
    
    # 准备数据 - 使用正确的返回顺序
    print("\n=== Loading Data ===")
    train_data, valid_data, test_data, src_vocab, tgt_vocab = prepare_data(
        data_dir=args.data_dir,
        vocab_dir=args.vocab_dir,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        train_file=train_file,
        valid_file='valid.jsonl',
        test_file='test.jsonl',
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size
    )
    
    # 创建DataLoader - 使用正确的函数
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        batch_size=args.batch_size,
        max_len=args.max_len
    )
    
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
    
    # 创建模型
    print("\n=== Creating Model ===")
    model = create_rnn_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        attn_type=args.attn_type,
        rnn_type=args.rnn_type,
        device=device
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Attention type: {args.attn_type}")
    print(f"RNN type: {args.rnn_type}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 创建输出目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 训练日志
    train_log = {
        'args': vars(args),
        'train_losses': [],
        'valid_losses': [],
        'best_epoch': 0,
        'best_valid_loss': float('inf')
    }
    
    # 获取反向词表
    tgt_vocab_rev = get_reverse_vocab(tgt_vocab)
    src_vocab_rev = get_reverse_vocab(src_vocab)
    
    # 训练循环
    print("\n=== Training ===")
    best_valid_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            args.grad_clip, device, args.teacher_forcing_ratio
        )
        
        # 验证
        valid_loss = evaluate(model, valid_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(valid_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 记录日志
        train_log['train_losses'].append(train_loss)
        train_log['valid_losses'].append(valid_loss)
        
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            train_log['best_epoch'] = epoch
            train_log['best_valid_loss'] = valid_loss
            
            checkpoint_path = os.path.join(args.checkpoint_dir, 'model_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'src_vocab_size': len(src_vocab),
                'tgt_vocab_size': len(tgt_vocab),
                'args': vars(args)
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")
        
        # 生成翻译样例（选择第一个样例）
        if epoch % 5 == 0 or epoch == 1:
            print("\n  Sample translations:")
            sample_batch = next(iter(valid_loader))
            
            # 选择批次中的第一个样本
            src_lens = sample_batch[2]  # 源文本长度
            
            src_sample = sample_batch[0][0]  # 第一个样本
            tgt_sample = sample_batch[1][0]
            
            # 源文本
            src_tokens = ids_to_tokens(src_sample.tolist(), src_vocab_rev)
            print(f"    Source (len={src_lens[0]}): {' '.join(src_tokens)}")
            
            # 参考翻译
            tgt_tokens = ids_to_tokens(tgt_sample.tolist(), tgt_vocab_rev)
            if args.tgt_lang == 'zh':
                print(f"    Reference: {''.join(tgt_tokens)}")
            else:
                print(f"    Reference: {' '.join(tgt_tokens)}")
            
            # 模型翻译（使用重复惩罚）
            pred_tokens = translate_sample(model, src_sample, src_vocab_rev, tgt_vocab_rev, device, 
                                          repetition_penalty=args.repetition_penalty)
            if args.tgt_lang == 'zh':
                print(f"    Prediction: {''.join(pred_tokens)}")
            else:
                print(f"    Prediction: {' '.join(pred_tokens)}")
    
    # 训练完成
    total_time = time.time() - start_time
    print(f"\n=== Training Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best epoch: {train_log['best_epoch']}")
    print(f"Best valid loss: {train_log['best_valid_loss']:.4f}")
    
    # 保存训练日志
    log_path = os.path.join(args.log_dir, 'train_log.json')
    train_log['total_time_minutes'] = total_time / 60
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(train_log, f, indent=2)
    print(f"Training log saved to {log_path}")
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'model_final.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss': valid_loss,
        'src_vocab_size': len(src_vocab),
        'tgt_vocab_size': len(tgt_vocab),
        'args': vars(args)
    }, final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RNN Seq2Seq model')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, 
                        default='AP0004_Midterm&Final_translation_dataset_zh_en',
                        help='数据目录')
    parser.add_argument('--vocab_dir', type=str, default='data/vocab',
                        help='词表目录')
    parser.add_argument('--src_lang', type=str, default='en',
                        help='源语言')
    parser.add_argument('--tgt_lang', type=str, default='zh',
                        help='目标语言')
    parser.add_argument('--use_small', action='store_true', default=True,
                        help='使用小数据集 (10k)')
    parser.add_argument('--max_len', type=int, default=100,
                        help='最大序列长度')
    parser.add_argument('--min_freq', type=int, default=2,
                        help='最小词频')
    parser.add_argument('--max_vocab_size', type=int, default=30000,
                        help='最大词表大小')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='词嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='隐藏层维度')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='RNN层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout比率')
    parser.add_argument('--attn_type', type=str, default='dot',
                        choices=['dot', 'additive', 'multiplicative'],
                        help='注意力类型')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        choices=['lstm', 'gru'],
                        help='RNN类型')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=15,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0,
                        help='Teacher Forcing比率')
    parser.add_argument('--repetition_penalty', type=float, default=1.5,
                        help='重复惩罚系数（推理时使用，1.0=无惩罚，推荐1.2-2.0）')
    
    # 输出参数
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='experiments/exp_001_rnn_baseline/checkpoints',
                        help='检查点目录')
    parser.add_argument('--log_dir', type=str,
                        default='experiments/exp_001_rnn_baseline/logs',
                        help='日志目录')
    
    args = parser.parse_args()
    main(args)
