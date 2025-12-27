"""
Transformer 模型消融实验与超参数敏感性分析脚本

实验内容：
1. 架构消融研究：
   - 位置编码对比：绝对位置编码（sin/cos）vs 可学习位置编码 vs 相对位置编码
   - 归一化方法对比：LayerNorm vs RMSNorm

2. 超参数敏感性分析：
   - 批次大小(batch_size)：32, 64, 128
   - 学习率(learning_rate)：1e-3, 5e-4, 1e-4, 5e-5
   - 模型规模(d_model/nhead/layers)：小型、中型、大型

符合作业要求：
- 比较不同位置编码方案（如绝对位置编码 vs 相对位置编码）
- 比较归一化方法（如LayerNorm vs RMSNorm）
- 调整批次大小、学习率、模型规模，分析其对性能的影响

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
from src.models.transformer import create_transformer_model, count_parameters


# ============================================================
# 实验配置定义
# ============================================================

# 位置编码消融实验配置
POSITION_ENCODING_EXPERIMENTS = {
    'sinusoidal': {
        'name': '绝对位置编码 (Sinusoidal)',
        'pos_encoding_type': 'sinusoidal',
        'description': '原始Transformer使用的固定sin/cos位置编码，不需要学习'
    },
    'learned': {
        'name': '可学习位置编码 (Learned)',
        'pos_encoding_type': 'learned',
        'description': '使用可学习的位置嵌入，可以适应具体任务'
    },
    'relative': {
        'name': '相对位置编码 (Relative)',
        'pos_encoding_type': 'relative',
        'description': '关注token间相对距离的位置编码，类似Transformer-XL'
    }
}

# 归一化方法消融实验配置
NORMALIZATION_EXPERIMENTS = {
    'layernorm': {
        'name': 'LayerNorm 归一化',
        'norm_type': 'layernorm',
        'description': '标准Layer Normalization，计算均值和方差进行归一化'
    },
    'rmsnorm': {
        'name': 'RMSNorm 归一化',
        'norm_type': 'rmsnorm',
        'description': 'Root Mean Square归一化，仅使用RMS，计算更高效'
    }
}

# 批次大小敏感性实验配置
BATCH_SIZE_EXPERIMENTS = {
    'batch_32': {
        'name': '批次大小 32',
        'batch_size': 32,
        'description': '较小批次，梯度更新频繁，内存占用低'
    },
    'batch_64': {
        'name': '批次大小 64',
        'batch_size': 64,
        'description': '中等批次（基线），平衡训练效率和梯度稳定性'
    },
    'batch_128': {
        'name': '批次大小 128',
        'batch_size': 128,
        'description': '较大批次，梯度更稳定，需要更多内存'
    }
}

# 学习率敏感性实验配置
LEARNING_RATE_EXPERIMENTS = {
    'lr_1e-3': {
        'name': '学习率 1e-3',
        'learning_rate': 1e-3,
        'description': '较高学习率，收敛快但可能不稳定'
    },
    'lr_5e-4': {
        'name': '学习率 5e-4',
        'learning_rate': 5e-4,
        'description': '中高学习率'
    },
    'lr_1e-4': {
        'name': '学习率 1e-4',
        'learning_rate': 1e-4,
        'description': '标准学习率（基线）'
    },
    'lr_5e-5': {
        'name': '学习率 5e-5',
        'learning_rate': 5e-5,
        'description': '较低学习率，收敛慢但更稳定'
    }
}

# 模型规模敏感性实验配置
MODEL_SCALE_EXPERIMENTS = {
    'scale_small': {
        'name': '小型模型',
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 512,
        'description': '小型模型：d_model=128, nhead=4, layers=2, ff=512'
    },
    'scale_medium': {
        'name': '中型模型（基线）',
        'd_model': 256,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 1024,
        'description': '中型模型：d_model=256, nhead=4, layers=3, ff=1024'
    },
    'scale_large': {
        'name': '大型模型',
        'd_model': 512,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 2048,
        'description': '大型模型：d_model=512, nhead=8, layers=4, ff=2048'
    }
}


# ============================================================
# 训练函数
# ============================================================

def train_epoch(model, train_loader, optimizer, criterion, clip, device):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        clip: 梯度裁剪阈值
        device: 设备
    
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
        output = model(src, tgt[:, :-1])
        
        # 计算loss
        output = output.reshape(-1, output.size(-1))
        target = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, target)
        
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
            
            # Forward pass
            output = model(src, tgt[:, :-1])
            
            output = output.reshape(-1, output.size(-1))
            target = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(valid_loader)


def translate_sample(model, src, tgt_vocab_rev, device, max_len=50, repetition_penalty=1.5):
    """翻译单个样本"""
    model.eval()
    src_tensor = src.unsqueeze(0).to(device)
    
    with torch.no_grad():
        translation = model.translate(src_tensor, max_len=max_len, 
                                       repetition_penalty=repetition_penalty)
    
    translation = translation[0].cpu().tolist()
    translated_tokens = ids_to_tokens(translation, tgt_vocab_rev)
    
    return translated_tokens


def run_single_experiment(
    args,
    experiment_name: str,
    experiment_config: Dict,
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
        experiment_config: 实验配置
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
    print(f"配置: {experiment_config.get('name', experiment_name)}")
    print(f"描述: {experiment_config.get('description', '')}")
    print(f"{'='*60}")
    
    # 创建实验输出目录
    exp_dir = os.path.join(base_output_dir, experiment_name)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    log_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取实验特定参数
    pos_encoding_type = experiment_config.get('pos_encoding_type', args.pos_encoding_type)
    norm_type = experiment_config.get('norm_type', args.norm_type)
    d_model = experiment_config.get('d_model', args.d_model)
    nhead = experiment_config.get('nhead', args.nhead)
    num_layers = experiment_config.get('num_layers', args.num_encoder_layers)
    dim_feedforward = experiment_config.get('dim_feedforward', args.dim_feedforward)
    learning_rate = experiment_config.get('learning_rate', args.learning_rate)
    batch_size = experiment_config.get('batch_size', args.batch_size)
    
    # 如果批次大小改变，需要重新创建DataLoader
    if batch_size != args.batch_size:
        print(f"重新创建DataLoader，批次大小: {batch_size}")
        # 从args获取数据参数重新创建
        train_file = 'train_10k.jsonl' if args.use_small else 'train_100k.jsonl'
        train_data, valid_data, test_data, _, _ = prepare_data(
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
        train_loader, valid_loader, _ = create_dataloaders(
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            batch_size=batch_size,
            max_len=args.max_len
        )
    
    # 创建模型
    model = create_transformer_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=args.dropout,
        pos_encoding_type=pos_encoding_type,
        norm_type=norm_type,
        device=device
    )
    
    param_count = count_parameters(model)
    print(f"模型参数量: {param_count:,}")
    print(f"位置编码: {pos_encoding_type}, 归一化: {norm_type}")
    print(f"d_model: {d_model}, nhead: {nhead}, layers: {num_layers}, ff: {dim_feedforward}")
    print(f"学习率: {learning_rate}, 批次大小: {batch_size}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 训练日志
    train_log = {
        'experiment_name': experiment_name,
        'config': experiment_config,
        'pos_encoding_type': pos_encoding_type,
        'norm_type': norm_type,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dim_feedforward': dim_feedforward,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'param_count': param_count,
        'args': {k: v for k, v in vars(args).items() if not k.startswith('_')},
        'train_losses': [],
        'valid_losses': [],
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
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            args.grad_clip, device
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
            
            checkpoint_path = os.path.join(checkpoint_dir, 'model_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'src_vocab_size': len(src_vocab),
                'tgt_vocab_size': len(tgt_vocab),
                'pos_encoding_type': pos_encoding_type,
                'norm_type': norm_type,
                'd_model': d_model,
                'nhead': nhead,
                'num_layers': num_layers,
                'dim_feedforward': dim_feedforward,
                'args': {k: v for k, v in vars(args).items() if not k.startswith('_')}
            }, checkpoint_path)
            print(f"  ✓ 保存最佳模型到 {checkpoint_path}")
        
        # 每5个epoch或第1个epoch生成翻译样例
        if epoch % 5 == 0 or epoch == 1:
            print("\n  翻译样例:")
            sample_batch = next(iter(valid_loader))
            src_sample = sample_batch[0][0]
            tgt_sample = sample_batch[1][0]
            
            # 源文本
            src_tokens = ids_to_tokens(src_sample.tolist(), src_vocab_rev)
            print(f"    源文本: {' '.join(src_tokens)}")
            
            # 参考翻译
            tgt_tokens = ids_to_tokens(tgt_sample.tolist(), tgt_vocab_rev)
            if args.tgt_lang == 'zh':
                print(f"    参考: {''.join(tgt_tokens)}")
            else:
                print(f"    参考: {' '.join(tgt_tokens)}")
            
            # 模型翻译
            pred_tokens = translate_sample(model, src_sample, tgt_vocab_rev, device,
                                          repetition_penalty=args.repetition_penalty)
            if args.tgt_lang == 'zh':
                print(f"    预测: {''.join(pred_tokens)}")
            else:
                print(f"    预测: {' '.join(pred_tokens)}")
    
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
        'pos_encoding_type': pos_encoding_type,
        'norm_type': norm_type,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dim_feedforward': dim_feedforward,
        'args': {k: v for k, v in vars(args).items() if not k.startswith('_')}
    }, final_checkpoint_path)
    
    return train_log


# ============================================================
# 消融实验函数
# ============================================================

def run_position_encoding_ablation(args, train_loader, valid_loader, src_vocab, tgt_vocab, device):
    """
    运行位置编码消融实验
    
    对比三种位置编码：sinusoidal（绝对）, learned（可学习）, relative（相对）
    """
    print("\n" + "="*80)
    print("位置编码消融实验")
    print("对比：绝对位置编码（Sinusoidal）vs 可学习位置编码 vs 相对位置编码")
    print("="*80)
    
    results = {}
    base_dir = os.path.join(args.output_dir, 'position_encoding_ablation')
    
    for pos_key, pos_config in POSITION_ENCODING_EXPERIMENTS.items():
        exp_name = f"pos_{pos_key}"
        print(f"\n>>> {pos_config['name']}")
        print(f"    {pos_config['description']}")
        
        result = run_single_experiment(
            args=args,
            experiment_name=exp_name,
            experiment_config=pos_config,
            train_loader=train_loader,
            valid_loader=valid_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            base_output_dir=base_dir
        )
        results[pos_key] = result
    
    # 保存对比结果
    save_comparison_results(
        results=results,
        experiments_config=POSITION_ENCODING_EXPERIMENTS,
        experiment_type='position_encoding_ablation',
        description='位置编码对比实验：绝对位置编码 vs 可学习位置编码 vs 相对位置编码',
        base_dir=base_dir
    )
    
    return results


def run_normalization_ablation(args, train_loader, valid_loader, src_vocab, tgt_vocab, device):
    """
    运行归一化方法消融实验
    
    对比两种归一化方法：LayerNorm vs RMSNorm
    """
    print("\n" + "="*80)
    print("归一化方法消融实验")
    print("对比：LayerNorm vs RMSNorm")
    print("="*80)
    
    results = {}
    base_dir = os.path.join(args.output_dir, 'normalization_ablation')
    
    for norm_key, norm_config in NORMALIZATION_EXPERIMENTS.items():
        exp_name = f"norm_{norm_key}"
        print(f"\n>>> {norm_config['name']}")
        print(f"    {norm_config['description']}")
        
        result = run_single_experiment(
            args=args,
            experiment_name=exp_name,
            experiment_config=norm_config,
            train_loader=train_loader,
            valid_loader=valid_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            base_output_dir=base_dir
        )
        results[norm_key] = result
    
    # 保存对比结果
    save_comparison_results(
        results=results,
        experiments_config=NORMALIZATION_EXPERIMENTS,
        experiment_type='normalization_ablation',
        description='归一化方法对比实验：LayerNorm vs RMSNorm',
        base_dir=base_dir
    )
    
    return results


def run_architecture_ablation(args, train_loader, valid_loader, src_vocab, tgt_vocab, device):
    """
    运行完整架构消融实验（位置编码 × 归一化方法）
    """
    print("\n" + "="*80)
    print("完整架构消融实验")
    print("对比：位置编码类型 × 归一化方法 的所有组合")
    print("="*80)
    
    all_results = {}
    base_dir = os.path.join(args.output_dir, 'architecture_ablation')
    
    # 遍历所有位置编码类型
    for pos_key, pos_config in POSITION_ENCODING_EXPERIMENTS.items():
        # 遍历所有归一化方法
        for norm_key, norm_config in NORMALIZATION_EXPERIMENTS.items():
            exp_name = f"{pos_key}_{norm_key}"
            
            # 合并配置
            combined_config = {
                'name': f"{pos_config['name']} + {norm_config['name']}",
                'pos_encoding_type': pos_config['pos_encoding_type'],
                'norm_type': norm_config['norm_type'],
                'description': f"位置编码: {pos_config['description']}, 归一化: {norm_config['description']}"
            }
            
            print(f"\n>>> 实验: {exp_name}")
            print(f"    位置编码: {pos_config['name']}")
            print(f"    归一化: {norm_config['name']}")
            
            result = run_single_experiment(
                args=args,
                experiment_name=exp_name,
                experiment_config=combined_config,
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
        'experiment_type': 'architecture_ablation',
        'description': '完整架构消融实验：位置编码类型 × 归一化方法 组合',
        'position_encoding_types': list(POSITION_ENCODING_EXPERIMENTS.keys()),
        'normalization_types': list(NORMALIZATION_EXPERIMENTS.keys()),
        'results': {}
    }
    
    for key, result in all_results.items():
        comparison['results'][key] = {
            'pos_encoding_type': result['pos_encoding_type'],
            'norm_type': result['norm_type'],
            'param_count': result['param_count'],
            'best_valid_loss': result['best_valid_loss'],
            'best_epoch': result['best_epoch'],
            'total_time_minutes': result['total_time_minutes']
        }
    
    comparison_path = os.path.join(base_dir, 'comparison_summary.json')
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n完整架构消融实验结果已保存至: {comparison_path}")
    return all_results


# ============================================================
# 超参数敏感性分析函数
# ============================================================

def run_batch_size_sensitivity(args, train_loader, valid_loader, src_vocab, tgt_vocab, device):
    """
    运行批次大小敏感性实验
    
    测试不同批次大小：32, 64, 128
    """
    print("\n" + "="*80)
    print("批次大小敏感性分析")
    print("测试批次大小: 32, 64, 128")
    print("="*80)
    
    results = {}
    base_dir = os.path.join(args.output_dir, 'batch_size_sensitivity')
    
    for batch_key, batch_config in BATCH_SIZE_EXPERIMENTS.items():
        exp_name = batch_key
        print(f"\n>>> {batch_config['name']}")
        print(f"    {batch_config['description']}")
        
        result = run_single_experiment(
            args=args,
            experiment_name=exp_name,
            experiment_config=batch_config,
            train_loader=train_loader,
            valid_loader=valid_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            base_output_dir=base_dir
        )
        results[batch_key] = result
    
    # 保存对比结果
    save_comparison_results(
        results=results,
        experiments_config=BATCH_SIZE_EXPERIMENTS,
        experiment_type='batch_size_sensitivity',
        description='批次大小敏感性分析：32 vs 64 vs 128',
        base_dir=base_dir,
        extra_fields=['batch_size']
    )
    
    return results


def run_learning_rate_sensitivity(args, train_loader, valid_loader, src_vocab, tgt_vocab, device):
    """
    运行学习率敏感性实验
    
    测试不同学习率：1e-3, 5e-4, 1e-4, 5e-5
    """
    print("\n" + "="*80)
    print("学习率敏感性分析")
    print("测试学习率: 1e-3, 5e-4, 1e-4, 5e-5")
    print("="*80)
    
    results = {}
    base_dir = os.path.join(args.output_dir, 'learning_rate_sensitivity')
    
    for lr_key, lr_config in LEARNING_RATE_EXPERIMENTS.items():
        exp_name = lr_key
        print(f"\n>>> {lr_config['name']}")
        print(f"    {lr_config['description']}")
        
        result = run_single_experiment(
            args=args,
            experiment_name=exp_name,
            experiment_config=lr_config,
            train_loader=train_loader,
            valid_loader=valid_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            base_output_dir=base_dir
        )
        results[lr_key] = result
    
    # 保存对比结果
    save_comparison_results(
        results=results,
        experiments_config=LEARNING_RATE_EXPERIMENTS,
        experiment_type='learning_rate_sensitivity',
        description='学习率敏感性分析：1e-3 vs 5e-4 vs 1e-4 vs 5e-5',
        base_dir=base_dir,
        extra_fields=['learning_rate']
    )
    
    return results


def run_model_scale_sensitivity(args, train_loader, valid_loader, src_vocab, tgt_vocab, device):
    """
    运行模型规模敏感性实验
    
    测试不同模型规模：小型、中型、大型
    """
    print("\n" + "="*80)
    print("模型规模敏感性分析")
    print("测试模型规模: 小型、中型、大型")
    print("="*80)
    
    results = {}
    base_dir = os.path.join(args.output_dir, 'model_scale_sensitivity')
    
    for scale_key, scale_config in MODEL_SCALE_EXPERIMENTS.items():
        exp_name = scale_key
        print(f"\n>>> {scale_config['name']}")
        print(f"    {scale_config['description']}")
        
        result = run_single_experiment(
            args=args,
            experiment_name=exp_name,
            experiment_config=scale_config,
            train_loader=train_loader,
            valid_loader=valid_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            base_output_dir=base_dir
        )
        results[scale_key] = result
    
    # 保存对比结果
    save_comparison_results(
        results=results,
        experiments_config=MODEL_SCALE_EXPERIMENTS,
        experiment_type='model_scale_sensitivity',
        description='模型规模敏感性分析：小型 vs 中型 vs 大型',
        base_dir=base_dir,
        extra_fields=['d_model', 'nhead', 'num_layers', 'dim_feedforward', 'param_count']
    )
    
    return results


def run_hyperparameter_sensitivity(args, train_loader, valid_loader, src_vocab, tgt_vocab, device):
    """
    运行完整超参数敏感性分析（批次大小 + 学习率 + 模型规模）
    """
    print("\n" + "="*80)
    print("完整超参数敏感性分析")
    print("包含：批次大小 + 学习率 + 模型规模")
    print("="*80)
    
    all_results = {}
    
    # 1. 批次大小敏感性
    print("\n[1/3] 批次大小敏感性分析...")
    batch_results = run_batch_size_sensitivity(
        args, train_loader, valid_loader, src_vocab, tgt_vocab, device
    )
    all_results['batch_size'] = batch_results
    
    # 2. 学习率敏感性
    print("\n[2/3] 学习率敏感性分析...")
    lr_results = run_learning_rate_sensitivity(
        args, train_loader, valid_loader, src_vocab, tgt_vocab, device
    )
    all_results['learning_rate'] = lr_results
    
    # 3. 模型规模敏感性
    print("\n[3/3] 模型规模敏感性分析...")
    scale_results = run_model_scale_sensitivity(
        args, train_loader, valid_loader, src_vocab, tgt_vocab, device
    )
    all_results['model_scale'] = scale_results
    
    # 保存完整结果
    base_dir = args.output_dir
    summary = {
        'experiment_type': 'hyperparameter_sensitivity',
        'description': '完整超参数敏感性分析',
        'sub_experiments': ['batch_size', 'learning_rate', 'model_scale'],
        'summary': {}
    }
    
    for exp_type, exp_results in all_results.items():
        summary['summary'][exp_type] = {}
        for key, result in exp_results.items():
            summary['summary'][exp_type][key] = {
                'best_valid_loss': result['best_valid_loss'],
                'best_epoch': result['best_epoch'],
                'total_time_minutes': result['total_time_minutes'],
                'param_count': result.get('param_count', 'N/A')
            }
    
    summary_path = os.path.join(base_dir, 'hyperparameter_sensitivity_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n完整超参数敏感性分析结果已保存至: {summary_path}")
    return all_results


# ============================================================
# 辅助函数
# ============================================================

def save_comparison_results(results, experiments_config, experiment_type, description, 
                           base_dir, extra_fields=None):
    """保存对比结果"""
    comparison = {
        'experiment_type': experiment_type,
        'description': description,
        'results': {}
    }
    
    for key, result in results.items():
        result_entry = {
            'name': experiments_config[key]['name'],
            'best_valid_loss': result['best_valid_loss'],
            'best_epoch': result['best_epoch'],
            'total_time_minutes': result['total_time_minutes'],
            'param_count': result.get('param_count', 'N/A')
        }
        
        # 添加额外字段
        if extra_fields:
            for field in extra_fields:
                if field in result:
                    result_entry[field] = result[field]
                elif field in experiments_config[key]:
                    result_entry[field] = experiments_config[key][field]
        
        comparison['results'][key] = result_entry
    
    comparison_path = os.path.join(base_dir, 'comparison_summary.json')
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n对比结果已保存至: {comparison_path}")


# ============================================================
# 主函数
# ============================================================

def main(args):
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
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
    results = None
    
    if args.experiment_type == 'position_encoding':
        # 位置编码消融实验
        results = run_position_encoding_ablation(
            args, train_loader, valid_loader, src_vocab, tgt_vocab, device
        )
    
    elif args.experiment_type == 'normalization':
        # 归一化方法消融实验
        results = run_normalization_ablation(
            args, train_loader, valid_loader, src_vocab, tgt_vocab, device
        )
    
    elif args.experiment_type == 'architecture':
        # 完整架构消融实验
        results = run_architecture_ablation(
            args, train_loader, valid_loader, src_vocab, tgt_vocab, device
        )
    
    elif args.experiment_type == 'batch_size':
        # 批次大小敏感性分析
        results = run_batch_size_sensitivity(
            args, train_loader, valid_loader, src_vocab, tgt_vocab, device
        )
    
    elif args.experiment_type == 'learning_rate':
        # 学习率敏感性分析
        results = run_learning_rate_sensitivity(
            args, train_loader, valid_loader, src_vocab, tgt_vocab, device
        )
    
    elif args.experiment_type == 'model_scale':
        # 模型规模敏感性分析
        results = run_model_scale_sensitivity(
            args, train_loader, valid_loader, src_vocab, tgt_vocab, device
        )
    
    elif args.experiment_type == 'hyperparameter':
        # 完整超参数敏感性分析
        results = run_hyperparameter_sensitivity(
            args, train_loader, valid_loader, src_vocab, tgt_vocab, device
        )
    
    elif args.experiment_type == 'full':
        # 运行所有实验（架构消融 + 超参数敏感性）
        print("\n" + "="*80)
        print("运行完整实验套件（架构消融 + 超参数敏感性分析）")
        print("="*80)
        
        # 架构消融
        print("\n[Part 1] 架构消融实验...")
        arch_results = run_architecture_ablation(
            args, train_loader, valid_loader, src_vocab, tgt_vocab, device
        )
        
        # 超参数敏感性
        print("\n[Part 2] 超参数敏感性分析...")
        hyper_results = run_hyperparameter_sensitivity(
            args, train_loader, valid_loader, src_vocab, tgt_vocab, device
        )
        
        results = {
            'architecture_ablation': arch_results,
            'hyperparameter_sensitivity': hyper_results
        }
    
    elif args.experiment_type == 'single':
        # 运行单个实验
        config = {
            'name': args.exp_name or 'single_exp',
            'pos_encoding_type': args.pos_encoding_type,
            'norm_type': args.norm_type,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_encoder_layers,
            'dim_feedforward': args.dim_feedforward,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'description': '单个实验配置'
        }
        
        result = run_single_experiment(
            args=args,
            experiment_name=args.exp_name or 'single_exp',
            experiment_config=config,
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
    
    # 打印结果汇总
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)
    
    if results and isinstance(results, dict):
        # 检查是否是嵌套结果（如full实验）
        if 'architecture_ablation' in results or 'hyperparameter_sensitivity' in results:
            for exp_category, exp_results in results.items():
                print(f"\n{exp_category} 结果汇总:")
                print("-" * 70)
                if isinstance(exp_results, dict):
                    for sub_category, sub_results in exp_results.items():
                        if isinstance(sub_results, dict) and 'best_valid_loss' in sub_results:
                            print(f"  {sub_category}: Loss={sub_results['best_valid_loss']:.4f}, "
                                  f"Epoch={sub_results['best_epoch']}")
                        elif isinstance(sub_results, dict):
                            for key, result in sub_results.items():
                                if isinstance(result, dict) and 'best_valid_loss' in result:
                                    print(f"    {key}: Loss={result['best_valid_loss']:.4f}")
        else:
            print("\n实验结果汇总:")
            print("-" * 70)
            print(f"{'实验名称':<35} {'最佳Loss':<12} {'最佳Epoch':<10} {'时间(分钟)':<10}")
            print("-" * 70)
            for exp_name, result in results.items():
                if isinstance(result, dict) and 'best_valid_loss' in result:
                    print(f"{exp_name:<35} {result['best_valid_loss']:<12.4f} "
                          f"{result['best_epoch']:<10} {result['total_time_minutes']:<10.1f}")
            print("-" * 70)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer 消融实验与超参数敏感性分析')
    
    # 实验类型
    parser.add_argument('--experiment_type', type=str, default='architecture',
                        choices=['position_encoding', 'normalization', 'architecture',
                                'batch_size', 'learning_rate', 'model_scale',
                                'hyperparameter', 'full', 'single'],
                        help='实验类型: '
                             'position_encoding(位置编码对比), '
                             'normalization(归一化方法对比), '
                             'architecture(完整架构消融), '
                             'batch_size(批次大小敏感性), '
                             'learning_rate(学习率敏感性), '
                             'model_scale(模型规模敏感性), '
                             'hyperparameter(完整超参数分析), '
                             'full(所有实验), '
                             'single(单个实验)')
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
    
    # 模型参数（默认值）
    parser.add_argument('--d_model', type=int, default=256,
                        help='模型维度')
    parser.add_argument('--nhead', type=int, default=4,
                        help='注意力头数')
    parser.add_argument('--num_encoder_layers', type=int, default=3,
                        help='编码器层数')
    parser.add_argument('--num_decoder_layers', type=int, default=3,
                        help='解码器层数')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                        help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout比率')
    
    # 消融实验参数
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal',
                        choices=['sinusoidal', 'learned', 'relative'],
                        help='位置编码类型（用于单个实验）')
    parser.add_argument('--norm_type', type=str, default='layernorm',
                        choices=['layernorm', 'rmsnorm'],
                        help='归一化类型（用于单个实验）')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=15,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--repetition_penalty', type=float, default=1.5,
                        help='重复惩罚系数')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, 
                        default='experiments/transformer_ablation',
                        help='实验输出目录')
    
    args = parser.parse_args()
    main(args)

