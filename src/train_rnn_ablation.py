"""
RNN Seq2Seq 模型消融实验训练脚本

实验内容：
1. 注意力机制对比：点积(dot)、乘性(multiplicative)、加性(additive)
2. 训练策略对比：Teacher Forcing vs Scheduled Sampling vs Free Running
3. 支持多种实验配置的批量训练

作者: NLP课程项目
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import copy

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


# ============================================================
# 实验配置定义
# ============================================================

# 注意力机制实验配置
ATTENTION_EXPERIMENTS = {
    'dot': {
        'name': '点积注意力 (Dot Product)',
        'attn_type': 'dot',
        'description': '最简单的注意力机制，直接计算query和key的点积: score = Q · K'
    },
    'multiplicative': {
        'name': '乘性注意力 (Multiplicative/Luong)',
        'attn_type': 'multiplicative',
        'description': 'Luong风格注意力，使用可学习权重矩阵: score = Q^T · W · K'
    },
    'additive': {
        'name': '加性注意力 (Additive/Bahdanau)',
        'attn_type': 'additive',
        'description': 'Bahdanau风格注意力: score = v^T · tanh(W1·Q + W2·K)'
    }
}

# 训练策略实验配置
TRAINING_STRATEGY_EXPERIMENTS = {
    'teacher_forcing': {
        'name': 'Teacher Forcing (TF=1.0)',
        'teacher_forcing_ratio': 1.0,
        'description': '每个解码步骤都使用真实目标token作为输入，训练稳定但可能导致exposure bias'
    },
    'scheduled_sampling': {
        'name': 'Scheduled Sampling (TF=0.5)',
        'teacher_forcing_ratio': 0.5,
        'description': '混合使用真实token和模型预测token，平衡训练效率和泛化能力'
    },
    'free_running': {
        'name': 'Free Running (TF=0.0)',
        'teacher_forcing_ratio': 0.0,
        'description': '完全使用模型自身的预测作为下一步输入，更接近推理时的行为'
    }
}


# ============================================================
# 训练函数
# ============================================================

def train_epoch(model, train_loader, optimizer, criterion, clip, device, 
                teacher_forcing_ratio=1.0, scheduled_sampling=False, epoch=1, total_epochs=1):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        clip: 梯度裁剪阈值
        device: 设备
        teacher_forcing_ratio: Teacher Forcing比率
        scheduled_sampling: 是否使用Scheduled Sampling（逐渐降低TF比率）
        epoch: 当前epoch
        total_epochs: 总epoch数
    
    Returns:
        平均loss
    """
    model.train()
    total_loss = 0
    
    # Scheduled Sampling: 线性降低teacher forcing比率
    if scheduled_sampling:
        # 从1.0线性降低到0.0
        tf_ratio = max(0.0, 1.0 - epoch / total_epochs)
    else:
        tf_ratio = teacher_forcing_ratio
    
    pbar = tqdm(train_loader, desc=f"Training (TF={tf_ratio:.2f})")
    for batch in pbar:
        src, tgt, src_lens, tgt_lens = batch
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt, src_lens, tf_ratio)
        
        # 计算loss (忽略第一个token <SOS>)
        output = output.reshape(-1, output.size(-1))
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        
        # Backward pass
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'tf': f'{tf_ratio:.2f}'})
    
    return total_loss / len(train_loader), tf_ratio


def evaluate(model, valid_loader, criterion, device):
    """
    验证模型
    
    Args:
        model: 模型
        valid_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
    
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


def translate_sample(model, src, src_vocab_rev, tgt_vocab_rev, device, 
                     max_len=50, use_beam_search=False, beam_size=5,
                     repetition_penalty=1.5):
    """
    翻译单个样本
    
    Args:
        model: 模型
        src: 源序列
        src_vocab_rev: 源语言反向词表
        tgt_vocab_rev: 目标语言反向词表
        device: 设备
        max_len: 最大生成长度
        use_beam_search: 是否使用束搜索
        beam_size: 束大小
        repetition_penalty: 重复惩罚系数
    
    Returns:
        翻译结果token列表
    """
    model.eval()
    src_tensor = src.unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_beam_search:
            translation = model.beam_search(src_tensor, beam_size=beam_size, max_len=max_len)
        else:
            translation = model.translate(src_tensor, max_len=max_len, 
                                          repetition_penalty=repetition_penalty)
    
    translation = translation[0].cpu().tolist()
    translated_tokens = ids_to_tokens(translation, tgt_vocab_rev)
    
    return translated_tokens


def run_single_experiment(
    args,
    experiment_name: str,
    attn_type: str,
    teacher_forcing_ratio: float,
    scheduled_sampling: bool,
    train_loader,
    valid_loader,
    src_vocab,
    tgt_vocab,
    device,
    base_output_dir: str
) -> Dict:
    """
    运行单个实验
    
    Args:
        args: 命令行参数
        experiment_name: 实验名称
        attn_type: 注意力类型
        teacher_forcing_ratio: Teacher Forcing比率
        scheduled_sampling: 是否使用Scheduled Sampling
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
        src_vocab: 源语言词表
        tgt_vocab: 目标语言词表
        device: 设备
        base_output_dir: 输出目录
    
    Returns:
        实验结果字典
    """
    print(f"\n{'='*60}")
    print(f"实验: {experiment_name}")
    print(f"注意力类型: {attn_type}")
    print(f"Teacher Forcing比率: {teacher_forcing_ratio}")
    print(f"Scheduled Sampling: {scheduled_sampling}")
    print(f"{'='*60}")
    
    # 创建实验输出目录
    exp_dir = os.path.join(base_output_dir, experiment_name)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    log_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建模型
    model = create_rnn_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        attn_type=attn_type,
        rnn_type=args.rnn_type,
        device=device
    )
    
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 训练日志
    train_log = {
        'experiment_name': experiment_name,
        'attn_type': attn_type,
        'teacher_forcing_ratio': teacher_forcing_ratio,
        'scheduled_sampling': scheduled_sampling,
        'args': vars(args),
        'train_losses': [],
        'valid_losses': [],
        'tf_ratios': [],
        'best_epoch': 0,
        'best_valid_loss': float('inf')
    }
    
    # 获取反向词表
    tgt_vocab_rev = get_reverse_vocab(tgt_vocab)
    src_vocab_rev = get_reverse_vocab(src_vocab)
    
    # 训练循环
    best_valid_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss, actual_tf_ratio = train_epoch(
            model, train_loader, optimizer, criterion,
            args.grad_clip, device, teacher_forcing_ratio,
            scheduled_sampling, epoch, args.epochs
        )
        
        # 验证
        valid_loss = evaluate(model, valid_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(valid_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}")
        print(f"  TF Ratio: {actual_tf_ratio:.2f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 记录日志
        train_log['train_losses'].append(train_loss)
        train_log['valid_losses'].append(valid_loss)
        train_log['tf_ratios'].append(actual_tf_ratio)
        
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            train_log['best_epoch'] = epoch
            train_log['best_valid_loss'] = valid_loss
            
            checkpoint_path = os.path.join(checkpoint_dir, 'model_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'src_vocab_size': len(src_vocab),
                'tgt_vocab_size': len(tgt_vocab),
                'attn_type': attn_type,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✓ 保存最佳模型到 {checkpoint_path}")
        
        # 每5个epoch生成翻译样例
        if epoch % 5 == 0 or epoch == 1:
            print("\n  翻译样例:")
            sample_batch = next(iter(valid_loader))
            src_sample = sample_batch[0][0]
            tgt_sample = sample_batch[1][0]
            src_lens = sample_batch[2]
            
            # 源文本
            src_tokens = ids_to_tokens(src_sample.tolist(), src_vocab_rev)
            print(f"    源文本 (len={src_lens[0]}): {' '.join(src_tokens)}")
            
            # 参考翻译
            tgt_tokens = ids_to_tokens(tgt_sample.tolist(), tgt_vocab_rev)
            if args.tgt_lang == 'zh':
                print(f"    参考: {''.join(tgt_tokens)}")
            else:
                print(f"    参考: {' '.join(tgt_tokens)}")
            
            # 贪婪解码
            pred_tokens = translate_sample(model, src_sample, src_vocab_rev, tgt_vocab_rev, 
                                          device, use_beam_search=False,
                                          repetition_penalty=args.repetition_penalty)
            if args.tgt_lang == 'zh':
                print(f"    贪婪: {''.join(pred_tokens)}")
            else:
                print(f"    贪婪: {' '.join(pred_tokens)}")
            
            # 束搜索解码
            pred_tokens_beam = translate_sample(model, src_sample, src_vocab_rev, tgt_vocab_rev,
                                                device, use_beam_search=True, beam_size=5)
            if args.tgt_lang == 'zh':
                print(f"    束搜索: {''.join(pred_tokens_beam)}")
            else:
                print(f"    束搜索: {' '.join(pred_tokens_beam)}")
    
    # 训练完成
    total_time = time.time() - start_time
    train_log['total_time_minutes'] = total_time / 60
    
    print(f"\n实验完成: {experiment_name}")
    print(f"  总时间: {total_time/60:.1f} 分钟")
    print(f"  最佳epoch: {train_log['best_epoch']}")
    print(f"  最佳验证loss: {train_log['best_valid_loss']:.4f}")
    
    # 保存训练日志
    log_path = os.path.join(log_dir, 'train_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(train_log, f, indent=2, ensure_ascii=False)
    print(f"  日志保存至 {log_path}")
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(checkpoint_dir, 'model_final.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss': valid_loss,
        'src_vocab_size': len(src_vocab),
        'tgt_vocab_size': len(tgt_vocab),
        'attn_type': attn_type,
        'args': vars(args)
    }, final_checkpoint_path)
    
    return train_log


def run_attention_ablation(args, train_loader, valid_loader, src_vocab, tgt_vocab, device):
    """
    运行注意力机制消融实验
    
    对比三种注意力机制：dot, multiplicative, additive
    """
    print("\n" + "="*80)
    print("注意力机制消融实验")
    print("对比：点积注意力 vs 乘性注意力 vs 加性注意力")
    print("="*80)
    
    results = {}
    base_dir = os.path.join(args.output_dir, 'attention_ablation')
    
    for attn_key, attn_config in ATTENTION_EXPERIMENTS.items():
        exp_name = f"attn_{attn_key}"
        print(f"\n>>> {attn_config['name']}")
        print(f"    {attn_config['description']}")
        
        result = run_single_experiment(
            args=args,
            experiment_name=exp_name,
            attn_type=attn_config['attn_type'],
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            scheduled_sampling=False,
            train_loader=train_loader,
            valid_loader=valid_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            base_output_dir=base_dir
        )
        results[attn_key] = result
    
    # 保存对比结果
    comparison = {
        'experiment_type': 'attention_ablation',
        'description': '注意力机制对比实验：点积 vs 乘性 vs 加性',
        'results': {}
    }
    
    for key, result in results.items():
        comparison['results'][key] = {
            'name': ATTENTION_EXPERIMENTS[key]['name'],
            'best_valid_loss': result['best_valid_loss'],
            'best_epoch': result['best_epoch'],
            'total_time_minutes': result['total_time_minutes']
        }
    
    comparison_path = os.path.join(base_dir, 'comparison_summary.json')
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n注意力机制对比结果已保存至: {comparison_path}")
    return results


def run_training_strategy_ablation(args, train_loader, valid_loader, src_vocab, tgt_vocab, device):
    """
    运行训练策略消融实验
    
    对比三种训练策略：Teacher Forcing, Scheduled Sampling, Free Running
    """
    print("\n" + "="*80)
    print("训练策略消融实验")
    print("对比：Teacher Forcing vs Scheduled Sampling vs Free Running")
    print("="*80)
    
    results = {}
    base_dir = os.path.join(args.output_dir, 'training_strategy_ablation')
    
    for strategy_key, strategy_config in TRAINING_STRATEGY_EXPERIMENTS.items():
        exp_name = f"strategy_{strategy_key}"
        print(f"\n>>> {strategy_config['name']}")
        print(f"    {strategy_config['description']}")
        
        # Scheduled Sampling使用特殊逻辑
        scheduled_sampling = (strategy_key == 'scheduled_sampling_decay')
        
        result = run_single_experiment(
            args=args,
            experiment_name=exp_name,
            attn_type=args.attn_type,  # 使用默认注意力类型
            teacher_forcing_ratio=strategy_config['teacher_forcing_ratio'],
            scheduled_sampling=scheduled_sampling,
            train_loader=train_loader,
            valid_loader=valid_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            base_output_dir=base_dir
        )
        results[strategy_key] = result
    
    # 保存对比结果
    comparison = {
        'experiment_type': 'training_strategy_ablation',
        'description': '训练策略对比实验：Teacher Forcing vs Scheduled Sampling vs Free Running',
        'results': {}
    }
    
    for key, result in results.items():
        comparison['results'][key] = {
            'name': TRAINING_STRATEGY_EXPERIMENTS[key]['name'],
            'best_valid_loss': result['best_valid_loss'],
            'best_epoch': result['best_epoch'],
            'total_time_minutes': result['total_time_minutes']
        }
    
    comparison_path = os.path.join(base_dir, 'comparison_summary.json')
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n训练策略对比结果已保存至: {comparison_path}")
    return results


def run_full_ablation(args, train_loader, valid_loader, src_vocab, tgt_vocab, device):
    """
    运行完整消融实验（注意力 + 训练策略组合）
    """
    print("\n" + "="*80)
    print("完整消融实验")
    print("包含：注意力机制 × 训练策略 的所有组合")
    print("="*80)
    
    all_results = {}
    base_dir = os.path.join(args.output_dir, 'full_ablation')
    
    # 遍历所有注意力类型
    for attn_key, attn_config in ATTENTION_EXPERIMENTS.items():
        # 遍历所有训练策略
        for strategy_key, strategy_config in TRAINING_STRATEGY_EXPERIMENTS.items():
            exp_name = f"{attn_key}_{strategy_key}"
            
            print(f"\n>>> 实验: {exp_name}")
            print(f"    注意力: {attn_config['name']}")
            print(f"    策略: {strategy_config['name']}")
            
            scheduled_sampling = (strategy_key == 'scheduled_sampling_decay')
            
            result = run_single_experiment(
                args=args,
                experiment_name=exp_name,
                attn_type=attn_config['attn_type'],
                teacher_forcing_ratio=strategy_config['teacher_forcing_ratio'],
                scheduled_sampling=scheduled_sampling,
                train_loader=train_loader,
                valid_loader=valid_loader,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                device=device,
                base_output_dir=base_dir
            )
            all_results[exp_name] = result
    
    # 保存完整对比结果
    comparison = {
        'experiment_type': 'full_ablation',
        'description': '完整消融实验：注意力机制 × 训练策略 组合',
        'attention_types': list(ATTENTION_EXPERIMENTS.keys()),
        'training_strategies': list(TRAINING_STRATEGY_EXPERIMENTS.keys()),
        'results': {}
    }
    
    for key, result in all_results.items():
        comparison['results'][key] = {
            'attn_type': result['attn_type'],
            'teacher_forcing_ratio': result['teacher_forcing_ratio'],
            'best_valid_loss': result['best_valid_loss'],
            'best_epoch': result['best_epoch'],
            'total_time_minutes': result['total_time_minutes']
        }
    
    comparison_path = os.path.join(base_dir, 'comparison_summary.json')
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n完整消融实验结果已保存至: {comparison_path}")
    return all_results


def main(args):
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 选择训练文件
    train_file = 'train_10k.jsonl' if args.use_small else 'train_100k.jsonl'
    
    # 准备数据
    print("\n=== 加载数据 ===")
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
    
    # 创建DataLoader
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
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(valid_loader)}")
    print(f"源语言词表大小: {len(src_vocab)}")
    print(f"目标语言词表大小: {len(tgt_vocab)}")
    
    # 根据实验类型运行
    if args.experiment_type == 'attention':
        # 仅运行注意力机制消融实验
        results = run_attention_ablation(
            args, train_loader, valid_loader, src_vocab, tgt_vocab, device
        )
    elif args.experiment_type == 'training_strategy':
        # 仅运行训练策略消融实验
        results = run_training_strategy_ablation(
            args, train_loader, valid_loader, src_vocab, tgt_vocab, device
        )
    elif args.experiment_type == 'full':
        # 运行完整消融实验
        results = run_full_ablation(
            args, train_loader, valid_loader, src_vocab, tgt_vocab, device
        )
    elif args.experiment_type == 'single':
        # 运行单个实验
        result = run_single_experiment(
            args=args,
            experiment_name=args.exp_name or 'single_exp',
            attn_type=args.attn_type,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            scheduled_sampling=args.scheduled_sampling,
            train_loader=train_loader,
            valid_loader=valid_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            base_output_dir=args.output_dir
        )
        results = {args.exp_name or 'single_exp': result}
    else:
        raise ValueError(f"未知实验类型: {args.experiment_type}")
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)
    
    # 打印结果汇总
    print("\n实验结果汇总:")
    print("-" * 60)
    print(f"{'实验名称':<30} {'最佳Loss':<12} {'最佳Epoch':<10} {'时间(分钟)':<10}")
    print("-" * 60)
    for exp_name, result in results.items():
        print(f"{exp_name:<30} {result['best_valid_loss']:<12.4f} {result['best_epoch']:<10} {result['total_time_minutes']:<10.1f}")
    print("-" * 60)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNN Seq2Seq 消融实验')
    
    # 实验类型
    parser.add_argument('--experiment_type', type=str, default='attention',
                        choices=['attention', 'training_strategy', 'full', 'single'],
                        help='实验类型: attention(注意力对比), training_strategy(训练策略对比), '
                             'full(完整消融), single(单个实验)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='实验名称（用于single类型）')
    
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
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='隐藏层维度')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='RNN层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout比率')
    parser.add_argument('--attn_type', type=str, default='dot',
                        choices=['dot', 'additive', 'multiplicative'],
                        help='注意力类型（用于单个实验或训练策略实验）')
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
                        help='Teacher Forcing比率（用于单个实验）')
    parser.add_argument('--scheduled_sampling', action='store_true', default=False,
                        help='是否使用Scheduled Sampling（用于单个实验）')
    parser.add_argument('--repetition_penalty', type=float, default=1.5,
                        help='重复惩罚系数')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, 
                        default='experiments/rnn_ablation',
                        help='实验输出目录')
    
    args = parser.parse_args()
    main(args)

