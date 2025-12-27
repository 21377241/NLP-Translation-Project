"""
RNN消融实验评估脚本

功能：
1. 支持多种解码策略评估：贪婪解码 vs 束搜索解码
2. 评估不同注意力机制训练的模型
3. 评估不同训练策略训练的模型
4. 生成详细的对比报告

作者: NLP课程项目
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import (
    load_data, prepare_data, create_dataloaders,
    tokenize_en, tokenize_zh, load_vocab,
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX,
    get_reverse_vocab, ids_to_tokens, TranslationDataset, collate_fn
)
from src.models.rnn_seq2seq import create_rnn_model

# 尝试导入不同的 BLEU 计算库
try:
    import sacrebleu
    USE_SACREBLEU = True
except ImportError:
    USE_SACREBLEU = False
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction


# ============================================================
# 解码策略配置
# ============================================================

DECODING_STRATEGIES = {
    'greedy': {
        'name': '贪婪解码 (Greedy Decoding)',
        'description': '每一步选择概率最高的token',
        'use_beam_search': False,
        'beam_size': 1
    },
    'beam_3': {
        'name': '束搜索 (Beam=3)',
        'description': '维护3个候选序列，选择整体得分最高的',
        'use_beam_search': True,
        'beam_size': 3
    },
    'beam_5': {
        'name': '束搜索 (Beam=5)',
        'description': '维护5个候选序列，选择整体得分最高的',
        'use_beam_search': True,
        'beam_size': 5
    },
    'beam_10': {
        'name': '束搜索 (Beam=10)',
        'description': '维护10个候选序列，选择整体得分最高的',
        'use_beam_search': True,
        'beam_size': 10
    }
}


# ============================================================
# BLEU计算函数
# ============================================================

def calculate_bleu_sacrebleu(hypotheses: List[str], references: List[str]) -> float:
    """使用 sacrebleu 计算 BLEU 分数"""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


def calculate_bleu_nltk(hypotheses: List[List[str]], references: List[List[str]]) -> float:
    """使用 nltk 计算 BLEU 分数"""
    refs = [[ref] for ref in references]
    smooth = SmoothingFunction().method1
    
    try:
        bleu = corpus_bleu(refs, hypotheses, smoothing_function=smooth)
        return bleu * 100
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        return 0.0


def tokens_to_string(tokens: List[str], lang: str = 'en') -> str:
    """将token列表转换为字符串"""
    if lang == 'zh':
        return ''.join(tokens)
    else:
        return ' '.join(tokens)


# ============================================================
# 评估函数
# ============================================================

def evaluate_with_decoding_strategy(
    model,
    test_loader,
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    device: torch.device,
    tgt_lang: str = 'zh',
    max_len: int = 100,
    use_beam_search: bool = False,
    beam_size: int = 5,
    repetition_penalty: float = 1.5,
    num_samples: int = 10
) -> Dict:
    """
    使用指定解码策略评估模型
    
    Args:
        model: 翻译模型
        test_loader: 测试数据 DataLoader
        src_vocab: 源语言词表
        tgt_vocab: 目标语言词表
        device: 设备
        tgt_lang: 目标语言
        max_len: 最大生成长度
        use_beam_search: 是否使用束搜索
        beam_size: 束大小
        repetition_penalty: 重复惩罚系数（仅贪婪解码使用）
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
    
    total_time = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Translating")):
            src, tgt, src_lens, tgt_lens = batch
            src = src.to(device)
            batch_size = src.size(0)
            
            start_time = time.time()
            
            # 根据解码策略翻译
            if use_beam_search:
                # 束搜索需要逐个样本处理
                translations_list = []
                for i in range(batch_size):
                    src_single = src[i:i+1]
                    translation = model.beam_search(src_single, beam_size=beam_size, max_len=max_len)
                    translations_list.append(translation)
                
                # 处理长度不一致的问题
                max_trans_len = max(t.size(1) for t in translations_list)
                translations = torch.full((batch_size, max_trans_len), PAD_IDX, dtype=torch.long, device=device)
                for i, trans in enumerate(translations_list):
                    translations[i, :trans.size(1)] = trans[0]
            else:
                translations = model.translate(src, max_len=max_len, 
                                               repetition_penalty=repetition_penalty)
            
            batch_time = time.time() - start_time
            total_time += batch_time
            total_samples += batch_size
            
            # 处理每个样本
            for i in range(batch_size):
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
    
    # 计算平均推理时间
    avg_time_per_sample = (total_time / total_samples) * 1000  # 毫秒
    
    results = {
        'bleu': bleu,
        'num_samples': len(all_hypotheses),
        'total_inference_time_seconds': total_time,
        'avg_inference_time_ms': avg_time_per_sample,
        'samples': samples
    }
    
    return results


def evaluate_single_model_all_strategies(
    model,
    test_loader,
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    device: torch.device,
    tgt_lang: str = 'zh',
    max_len: int = 100,
    repetition_penalty: float = 1.5,
    num_samples: int = 10
) -> Dict:
    """
    使用所有解码策略评估单个模型
    
    Returns:
        包含所有策略评估结果的字典
    """
    results = {}
    
    for strategy_key, strategy_config in DECODING_STRATEGIES.items():
        print(f"\n>>> 解码策略: {strategy_config['name']}")
        print(f"    {strategy_config['description']}")
        
        result = evaluate_with_decoding_strategy(
            model=model,
            test_loader=test_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            tgt_lang=tgt_lang,
            max_len=max_len,
            use_beam_search=strategy_config['use_beam_search'],
            beam_size=strategy_config['beam_size'],
            repetition_penalty=repetition_penalty,
            num_samples=num_samples
        )
        
        results[strategy_key] = {
            'strategy_name': strategy_config['name'],
            'bleu': result['bleu'],
            'avg_inference_time_ms': result['avg_inference_time_ms'],
            'samples': result['samples']
        }
        
        print(f"    BLEU: {result['bleu']:.2f}")
        print(f"    平均推理时间: {result['avg_inference_time_ms']:.2f}ms")
    
    return results


def load_and_evaluate_model(
    checkpoint_path: str,
    model_args: Dict,
    test_loader,
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    device: torch.device,
    tgt_lang: str = 'zh',
    max_len: int = 100,
    repetition_penalty: float = 1.5,
    num_samples: int = 10
) -> Dict:
    """
    加载模型并进行评估
    """
    # 创建模型
    model = create_rnn_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=model_args.get('embed_dim', 256),
        hidden_dim=model_args.get('hidden_dim', 512),
        n_layers=model_args.get('n_layers', 2),
        dropout=0.0,  # 评估时不使用dropout
        attn_type=model_args.get('attn_type', 'dot'),
        rnn_type=model_args.get('rnn_type', 'lstm'),
        device=device
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"  加载模型: {checkpoint_path}")
    print(f"  最佳epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  验证loss: {checkpoint.get('valid_loss', 'N/A'):.4f}")
    
    # 使用所有解码策略评估
    return evaluate_single_model_all_strategies(
        model=model,
        test_loader=test_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        tgt_lang=tgt_lang,
        max_len=max_len,
        repetition_penalty=repetition_penalty,
        num_samples=num_samples
    )


def evaluate_ablation_experiments(
    experiment_dir: str,
    test_loader,
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    device: torch.device,
    tgt_lang: str = 'zh',
    max_len: int = 100,
    repetition_penalty: float = 1.5,
    num_samples: int = 10
) -> Dict:
    """
    评估消融实验目录下的所有模型
    
    Args:
        experiment_dir: 消融实验目录（如 experiments/rnn_ablation/attention_ablation）
    
    Returns:
        所有模型的评估结果
    """
    all_results = {}
    
    # 遍历实验目录
    if not os.path.exists(experiment_dir):
        print(f"实验目录不存在: {experiment_dir}")
        return all_results
    
    exp_subdirs = [d for d in os.listdir(experiment_dir) 
                   if os.path.isdir(os.path.join(experiment_dir, d))]
    
    for exp_name in sorted(exp_subdirs):
        exp_path = os.path.join(experiment_dir, exp_name)
        checkpoint_path = os.path.join(exp_path, 'checkpoints', 'model_best.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"跳过 {exp_name}: 未找到模型文件")
            continue
        
        print(f"\n{'='*60}")
        print(f"评估实验: {exp_name}")
        print(f"{'='*60}")
        
        # 加载训练日志获取模型配置
        log_path = os.path.join(exp_path, 'logs', 'train_log.json')
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                train_log = json.load(f)
            model_args = train_log.get('args', {})
            model_args['attn_type'] = train_log.get('attn_type', 'dot')
        else:
            # 从checkpoint加载
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_args = checkpoint.get('args', {})
        
        # 评估模型
        results = load_and_evaluate_model(
            checkpoint_path=checkpoint_path,
            model_args=model_args,
            test_loader=test_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            tgt_lang=tgt_lang,
            max_len=max_len,
            repetition_penalty=repetition_penalty,
            num_samples=num_samples
        )
        
        all_results[exp_name] = {
            'model_args': {
                'attn_type': model_args.get('attn_type', 'dot'),
                'teacher_forcing_ratio': model_args.get('teacher_forcing_ratio', 1.0),
                'rnn_type': model_args.get('rnn_type', 'lstm')
            },
            'decoding_results': results
        }
    
    return all_results


def generate_comparison_report(all_results: Dict, output_path: str):
    """
    生成对比报告
    """
    report = {
        'summary': {
            'total_experiments': len(all_results),
            'decoding_strategies': list(DECODING_STRATEGIES.keys()),
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'detailed_results': all_results,
        'comparison_tables': {}
    }
    
    # 创建对比表格 - 按解码策略分组
    for strategy_key in DECODING_STRATEGIES.keys():
        table = {}
        for exp_name, exp_results in all_results.items():
            if strategy_key in exp_results.get('decoding_results', {}):
                strategy_result = exp_results['decoding_results'][strategy_key]
                table[exp_name] = {
                    'bleu': strategy_result['bleu'],
                    'avg_inference_time_ms': strategy_result['avg_inference_time_ms']
                }
        report['comparison_tables'][strategy_key] = table
    
    # 找出每种解码策略的最佳模型
    best_models = {}
    for strategy_key, table in report['comparison_tables'].items():
        if table:
            best_exp = max(table.items(), key=lambda x: x[1]['bleu'])
            best_models[strategy_key] = {
                'experiment': best_exp[0],
                'bleu': best_exp[1]['bleu']
            }
    report['best_models'] = best_models
    
    # 保存报告
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "="*80)
    print("评估报告摘要")
    print("="*80)
    
    print(f"\n总实验数: {len(all_results)}")
    print(f"\n解码策略对比:")
    print("-" * 70)
    print(f"{'实验名称':<30} {'Greedy':<12} {'Beam-3':<12} {'Beam-5':<12} {'Beam-10':<12}")
    print("-" * 70)
    
    for exp_name, exp_results in all_results.items():
        row = [exp_name[:28]]
        for strategy in ['greedy', 'beam_3', 'beam_5', 'beam_10']:
            if strategy in exp_results.get('decoding_results', {}):
                bleu = exp_results['decoding_results'][strategy]['bleu']
                row.append(f"{bleu:.2f}")
            else:
                row.append("N/A")
        print(f"{row[0]:<30} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")
    
    print("-" * 70)
    
    print(f"\n各解码策略最佳模型:")
    for strategy, best in best_models.items():
        print(f"  {DECODING_STRATEGIES[strategy]['name']}: {best['experiment']} (BLEU={best['bleu']:.2f})")
    
    print(f"\n详细报告已保存至: {output_path}")
    
    return report


def main(args):
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载词表
    print("\n=== 加载词表 ===")
    src_vocab_path = os.path.join(args.vocab_dir, f'vocab_{args.src_lang}.json')
    tgt_vocab_path = os.path.join(args.vocab_dir, f'vocab_{args.tgt_lang}.json')
    
    src_vocab = load_vocab(src_vocab_path)
    tgt_vocab = load_vocab(tgt_vocab_path)
    
    print(f"源语言词表大小: {len(src_vocab)}")
    print(f"目标语言词表大小: {len(tgt_vocab)}")
    
    # 加载测试数据
    print("\n=== 加载测试数据 ===")
    test_data = load_data(os.path.join(args.data_dir, 'test.jsonl'))
    print(f"测试样本数: {len(test_data)}")
    
    # 创建 DataLoader
    from torch.utils.data import DataLoader
    
    test_dataset = TranslationDataset(
        test_data, src_vocab, tgt_vocab,
        args.src_lang, args.tgt_lang, args.max_len
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    
    if args.mode == 'single':
        # 评估单个模型
        print(f"\n=== 评估单个模型 ===")
        print(f"Checkpoint: {args.checkpoint_path}")
        
        # 加载checkpoint获取模型参数
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model_args = checkpoint.get('args', {})
        
        results = load_and_evaluate_model(
            checkpoint_path=args.checkpoint_path,
            model_args=model_args,
            test_loader=test_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            tgt_lang=args.tgt_lang,
            max_len=args.max_len,
            repetition_penalty=args.repetition_penalty,
            num_samples=args.num_samples
        )
        
        # 保存结果
        if args.output_path:
            os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存至: {args.output_path}")
        
    elif args.mode == 'ablation':
        # 评估消融实验
        print(f"\n=== 评估消融实验 ===")
        print(f"实验目录: {args.experiment_dir}")
        
        all_results = evaluate_ablation_experiments(
            experiment_dir=args.experiment_dir,
            test_loader=test_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            tgt_lang=args.tgt_lang,
            max_len=args.max_len,
            repetition_penalty=args.repetition_penalty,
            num_samples=args.num_samples
        )
        
        # 生成对比报告
        output_path = args.output_path or os.path.join(args.experiment_dir, 'evaluation_report.json')
        report = generate_comparison_report(all_results, output_path)
    
    else:
        raise ValueError(f"未知模式: {args.mode}")
    
    print("\n评估完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNN消融实验评估')
    
    # 评估模式
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'ablation'],
                        help='评估模式: single(单个模型), ablation(消融实验批量评估)')
    
    # 单模型评估参数
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='模型checkpoint路径（single模式）')
    
    # 消融实验评估参数
    parser.add_argument('--experiment_dir', type=str, default=None,
                        help='消融实验目录（ablation模式）')
    
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
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--max_len', type=int, default=100,
                        help='最大序列长度')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='保存的翻译样例数量')
    parser.add_argument('--repetition_penalty', type=float, default=1.5,
                        help='重复惩罚系数（贪婪解码使用）')
    
    # 输出参数
    parser.add_argument('--output_path', type=str, default=None,
                        help='结果保存路径')
    
    args = parser.parse_args()
    
    # 参数验证
    if args.mode == 'single' and not args.checkpoint_path:
        parser.error("single模式需要指定 --checkpoint_path")
    if args.mode == 'ablation' and not args.experiment_dir:
        parser.error("ablation模式需要指定 --experiment_dir")
    
    main(args)

