"""
评估模块
包含BLEU计算、模型评估、翻译样例生成等功能
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Tuple, Optional

import torch
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import (
    load_data, prepare_data, create_dataloaders,
    tokenize_en, tokenize_zh, load_vocab,
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX,
    get_reverse_vocab, ids_to_tokens
)
from src.models.rnn_seq2seq import create_rnn_model
from src.models.transformer import create_transformer_model

# 尝试导入不同的 BLEU 计算库
try:
    import sacrebleu
    USE_SACREBLEU = True
except ImportError:
    USE_SACREBLEU = False
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction


def calculate_bleu_sacrebleu(hypotheses: List[str], references: List[str]) -> float:
    """
    使用 sacrebleu 计算 BLEU 分数
    
    Args:
        hypotheses: 预测翻译列表（字符串）
        references: 参考翻译列表（字符串）
        
    Returns:
        BLEU 分数
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


def calculate_bleu_nltk(hypotheses: List[List[str]], references: List[List[str]]) -> float:
    """
    使用 nltk 计算 BLEU 分数
    
    Args:
        hypotheses: 预测翻译列表（token列表）
        references: 参考翻译列表（token列表）
        
    Returns:
        BLEU 分数 (0-100)
    """
    # 为 corpus_bleu 准备数据
    # references 需要是 [[[ref1_tokens], [ref2_tokens]], ...] 的格式
    refs = [[ref] for ref in references]
    
    # 使用平滑函数避免0分
    smooth = SmoothingFunction().method1
    
    try:
        bleu = corpus_bleu(refs, hypotheses, smoothing_function=smooth)
        return bleu * 100
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        return 0.0


def tokens_to_string(tokens: List[str], lang: str = 'en') -> str:
    """
    将token列表转换为字符串
    
    Args:
        tokens: token列表
        lang: 语言 ('en' 或 'zh')
        
    Returns:
        字符串
    """
    if lang == 'zh':
        return ''.join(tokens)
    else:
        return ' '.join(tokens)


def evaluate_model(model,
                   test_loader,
                   src_vocab: Dict[str, int],
                   tgt_vocab: Dict[str, int],
                   device: torch.device,
                   tgt_lang: str = 'zh',
                   max_len: int = 100,
                   num_samples: int = 10) -> Dict:
    """
    评估模型性能
    
    Args:
        model: 翻译模型
        test_loader: 测试数据 DataLoader
        src_vocab: 源语言词表
        tgt_vocab: 目标语言词表
        device: 设备
        tgt_lang: 目标语言
        max_len: 最大生成长度
        num_samples: 保存的翻译样例数量
        
    Returns:
        评估结果字典
    """
    model.eval()
    
    tgt_vocab_rev = get_reverse_vocab(tgt_vocab)
    src_vocab_rev = get_reverse_vocab(src_vocab)
    
    all_hypotheses = []
    all_references = []
    all_hypotheses_tokens = []
    all_references_tokens = []
    
    samples = []
    
    print("Evaluating...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Translating")):
            src, tgt, src_lens, tgt_lens = batch
            src = src.to(device)
            
            # 翻译
            translations = model.translate(src, max_len=max_len)
            
            # 处理每个样本
            for i in range(src.size(0)):
                # 获取预测结果
                pred_ids = translations[i].cpu().tolist()
                pred_tokens = ids_to_tokens(pred_ids, tgt_vocab_rev)
                
                # 获取参考翻译
                ref_ids = tgt[i].cpu().tolist()
                ref_tokens = ids_to_tokens(ref_ids, tgt_vocab_rev)
                
                # 转换为字符串
                pred_str = tokens_to_string(pred_tokens, tgt_lang)
                ref_str = tokens_to_string(ref_tokens, tgt_lang)
                
                all_hypotheses.append(pred_str)
                all_references.append(ref_str)
                all_hypotheses_tokens.append(pred_tokens)
                all_references_tokens.append(ref_tokens)
                
                # 保存样例
                if len(samples) < num_samples:
                    src_ids = src[i].cpu().tolist()
                    src_tokens = ids_to_tokens(src_ids, src_vocab_rev)
                    src_str = tokens_to_string(src_tokens, 'en' if tgt_lang == 'zh' else 'zh')
                    
                    samples.append({
                        'source': src_str,
                        'reference': ref_str,
                        'prediction': pred_str
                    })
    
    # 计算 BLEU
    if USE_SACREBLEU:
        bleu = calculate_bleu_sacrebleu(all_hypotheses, all_references)
    else:
        bleu = calculate_bleu_nltk(all_hypotheses_tokens, all_references_tokens)
    
    results = {
        'bleu': bleu,
        'num_samples': len(all_hypotheses),
        'samples': samples
    }
    
    print(f"\nBLEU-4: {bleu:.2f}")
    print(f"Total samples: {len(all_hypotheses)}")
    
    return results


def load_model(model_type: str,
               checkpoint_path: str,
               src_vocab_size: int,
               tgt_vocab_size: int,
               device: torch.device,
               **kwargs) -> torch.nn.Module:
    """
    加载模型
    
    Args:
        model_type: 模型类型 ('rnn' 或 'transformer')
        checkpoint_path: checkpoint 路径
        src_vocab_size: 源语言词表大小
        tgt_vocab_size: 目标语言词表大小
        device: 设备
        **kwargs: 模型参数
        
    Returns:
        加载的模型
    """
    if model_type == 'rnn':
        model = create_rnn_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=kwargs.get('embed_dim', 256),
            hidden_dim=kwargs.get('hidden_dim', 256),
            n_layers=kwargs.get('n_layers', 2),
            dropout=kwargs.get('dropout', 0.3),
            attn_type=kwargs.get('attn_type', 'dot'),
            rnn_type=kwargs.get('rnn_type', 'lstm'),
            device=device
        )
    elif model_type == 'transformer':
        model = create_transformer_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=kwargs.get('d_model', 256),
            nhead=kwargs.get('nhead', 4),
            num_encoder_layers=kwargs.get('num_encoder_layers', 3),
            num_decoder_layers=kwargs.get('num_decoder_layers', 3),
            dim_feedforward=kwargs.get('dim_feedforward', 1024),
            dropout=kwargs.get('dropout', 0.1),
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载词表
    print("\n=== Loading Vocabularies ===")
    src_vocab_path = os.path.join(args.vocab_dir, f'vocab_{args.src_lang}.json')
    tgt_vocab_path = os.path.join(args.vocab_dir, f'vocab_{args.tgt_lang}.json')
    
    src_vocab = load_vocab(src_vocab_path)
    tgt_vocab = load_vocab(tgt_vocab_path)
    
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    
    # 加载测试数据
    print("\n=== Loading Test Data ===")
    test_data = load_data(os.path.join(args.data_dir, 'test.jsonl'))
    print(f"Test samples: {len(test_data)}")
    
    # 创建 DataLoader
    from src.data_utils import TranslationDataset, collate_fn
    from torch.utils.data import DataLoader
    
    test_dataset = TranslationDataset(
        test_data, src_vocab, tgt_vocab,
        args.src_lang, args.tgt_lang, args.max_len
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    
    # 加载模型
    print(f"\n=== Loading Model ({args.model_type}) ===")
    
    # 从 checkpoint 中获取模型参数
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model_args = checkpoint.get('args', {})
    
    if args.model_type == 'rnn':
        model = create_rnn_model(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embed_dim=model_args.get('embed_dim', 256),
            hidden_dim=model_args.get('hidden_dim', 256),
            n_layers=model_args.get('n_layers', 2),
            dropout=0.0,  # 评估时不使用dropout
            attn_type=model_args.get('attn_type', 'dot'),
            rnn_type=model_args.get('rnn_type', 'lstm'),
            device=device
        )
    else:
        model = create_transformer_model(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=model_args.get('d_model', 256),
            nhead=model_args.get('nhead', 4),
            num_encoder_layers=model_args.get('num_encoder_layers', 3),
            num_decoder_layers=model_args.get('num_decoder_layers', 3),
            dim_feedforward=model_args.get('dim_feedforward', 1024),
            dropout=0.0,  # 评估时不使用dropout
            device=device
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.checkpoint_path}")
    print(f"Best epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Valid loss: {checkpoint.get('valid_loss', 'N/A'):.4f}")
    
    # 评估
    print("\n=== Evaluating ===")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        tgt_lang=args.tgt_lang,
        max_len=args.max_len,
        num_samples=args.num_samples
    )
    
    # 保存结果
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output_path}")
    
    # 打印翻译样例
    print("\n=== Sample Translations ===")
    for i, sample in enumerate(results['samples'][:5]):
        print(f"\n[{i+1}]")
        print(f"  Source:     {sample['source']}")
        print(f"  Reference:  {sample['reference']}")
        print(f"  Prediction: {sample['prediction']}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate translation model')
    
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['rnn', 'transformer'],
                        help='模型类型')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--data_dir', type=str,
                        default='AP0004_Midterm&Final_translation_dataset_zh_en',
                        help='数据目录')
    parser.add_argument('--vocab_dir', type=str, default='data/vocab',
                        help='词表目录')
    parser.add_argument('--src_lang', type=str, default='en',
                        help='源语言')
    parser.add_argument('--tgt_lang', type=str, default='zh',
                        help='目标语言')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--max_len', type=int, default=100,
                        help='最大序列长度')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='保存的翻译样例数量')
    parser.add_argument('--output_path', type=str, default=None,
                        help='结果保存路径')
    
    args = parser.parse_args()
    main(args)

