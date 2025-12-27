#!/usr/bin/env python3
"""
统一评估脚本
评估所有已训练的模型（RNN、Transformer、T5）并生成结果报告

包含：
1. BLEU分数计算
2. 翻译样例展示
3. 结果保存到results目录
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    load_data, load_vocab, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX,
    get_reverse_vocab, ids_to_tokens, TranslationDataset,
    tokenize_en, tokenize_zh, collate_fn
)
from torch.utils.data import DataLoader
from src.models.rnn_seq2seq import create_rnn_model
from src.models.transformer import create_transformer_model

# 导入T5相关
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers library not found. T5 models will be skipped.")

# BLEU计算
try:
    import sacrebleu
    USE_SACREBLEU = True
except ImportError:
    USE_SACREBLEU = False
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def calculate_bleu(hypotheses: List[List[str]], references: List[List[str]]) -> float:
    """
    计算BLEU-4分数
    
    Args:
        hypotheses: 预测翻译列表（token列表）
        references: 参考翻译列表（token列表）
        
    Returns:
        BLEU分数 (0-100)
    """
    if USE_SACREBLEU:
        # 转换为字符串
        hyp_strs = [' '.join(h) for h in hypotheses]
        ref_strs = [' '.join(r) for r in references]
        bleu = sacrebleu.corpus_bleu(hyp_strs, [ref_strs])
        return bleu.score
    else:
        # 使用NLTK
        refs = [[ref] for ref in references]
        smooth = SmoothingFunction().method1
        try:
            bleu = corpus_bleu(refs, hypotheses, smoothing_function=smooth)
            return bleu * 100
        except:
            return 0.0


def tokens_to_string(tokens: List[str], lang: str = 'en') -> str:
    """将token列表转换为字符串"""
    if lang == 'zh':
        return ''.join(tokens)
    else:
        return ' '.join(tokens)


class ModelEvaluator:
    """模型评估器基类"""
    
    def __init__(self, model_name: str, direction: str, device: torch.device):
        self.model_name = model_name
        self.direction = direction
        self.device = device
        self.src_lang, self.tgt_lang = direction.split('2')
    
    def evaluate(self, test_loader, src_vocab, tgt_vocab) -> Dict:
        """评估模型，返回结果字典"""
        raise NotImplementedError


class RNNEvaluator(ModelEvaluator):
    """RNN模型评估器"""
    
    def __init__(self, checkpoint_path: str, direction: str, device: torch.device):
        super().__init__('RNN', direction, device)
        self.checkpoint_path = checkpoint_path
        self.model = None
    
    def load_model(self, src_vocab, tgt_vocab):
        """加载RNN模型"""
        print(f"Loading RNN model from {self.checkpoint_path}")
        
        # 加载checkpoint获取配置
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 从checkpoint中读取训练时的配置
        args = checkpoint.get('args', {})
        embed_dim = args.get('embed_dim', 256)
        hidden_dim = args.get('hidden_dim', 512)
        n_layers = args.get('n_layers', 2)
        dropout = args.get('dropout', 0.3)
        rnn_type = args.get('rnn_type', 'lstm')
        attn_type = args.get('attn_type', 'multiplicative')
        
        print(f"  Config: embed_dim={embed_dim}, hidden_dim={hidden_dim}, attn_type={attn_type}")
        
        # 创建模型
        self.model = create_rnn_model(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            attn_type=attn_type,
            device=self.device
        )
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"  ✓ Model loaded successfully")
    
    def evaluate(self, test_loader, src_vocab, tgt_vocab) -> Dict:
        """评估RNN模型"""
        if self.model is None:
            self.load_model(src_vocab, tgt_vocab)
        
        tgt_vocab_rev = get_reverse_vocab(tgt_vocab)
        
        hypotheses = []
        references = []
        translation_samples = []
        
        print(f"Evaluating RNN {self.direction}...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Translating")):
                # collate_fn返回元组: (src, tgt, src_lens, tgt_lens)
                src, tgt, src_lens, tgt_lens = batch
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                src_lens = src_lens.to(self.device)
                
                batch_size = src.size(0)
                
                # 使用贪婪解码
                predictions = self._greedy_decode(src, src_lens, max_len=100)
                
                # 转换为tokens
                for i in range(batch_size):
                    # 预测
                    pred_ids = predictions[i].cpu().tolist()
                    if EOS_IDX in pred_ids:
                        eos_idx = pred_ids.index(EOS_IDX)
                        pred_ids = pred_ids[:eos_idx]
                    pred_tokens = ids_to_tokens(pred_ids, tgt_vocab_rev)
                    hypotheses.append(pred_tokens)
                    
                    # 参考
                    ref_ids = tgt[i].cpu().tolist()
                    if SOS_IDX in ref_ids:
                        ref_ids = ref_ids[1:]
                    if EOS_IDX in ref_ids:
                        eos_idx = ref_ids.index(EOS_IDX)
                        ref_ids = ref_ids[:eos_idx]
                    ref_tokens = ids_to_tokens(ref_ids, tgt_vocab_rev)
                    references.append(ref_tokens)
                    
                    # 保存前5个样例
                    if len(translation_samples) < 5:
                        translation_samples.append({
                            'prediction': tokens_to_string(pred_tokens, self.tgt_lang),
                            'reference': tokens_to_string(ref_tokens, self.tgt_lang)
                        })
        
        # 计算BLEU
        bleu_score = calculate_bleu(hypotheses, references)
        
        return {
            'model': 'RNN',
            'direction': self.direction,
            'bleu_score': round(bleu_score, 2),
            'num_samples': len(hypotheses),
            'translation_samples': translation_samples
        }
    
    def _greedy_decode(self, src, src_lens, max_len=100):
        """贪婪解码"""
        batch_size = src.size(0)
        
        # 编码
        encoder_outputs, hidden = self.model.encoder(src, src_lens)
        src_mask = (src != PAD_IDX)
        
        # 初始化
        input_token = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=self.device)
        predictions = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        
        # 解码
        for t in range(max_len):
            output, hidden, _ = self.model.decoder(input_token, hidden, encoder_outputs, src_mask)
            pred_token = output.argmax(dim=-1)
            predictions[:, t] = pred_token
            input_token = pred_token
            
            if (pred_token == EOS_IDX).all():
                break
        
        return predictions


class TransformerEvaluator(ModelEvaluator):
    """Transformer模型评估器"""
    
    def __init__(self, checkpoint_path: str, direction: str, device: torch.device):
        super().__init__('Transformer', direction, device)
        self.checkpoint_path = checkpoint_path
        self.model = None
    
    def load_model(self, src_vocab, tgt_vocab):
        """加载Transformer模型"""
        print(f"Loading Transformer model from {self.checkpoint_path}")
        
        # 加载checkpoint获取配置
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 从checkpoint中读取训练时的配置
        args = checkpoint.get('args', {})
        d_model = args.get('d_model', 256)
        nhead = args.get('nhead', 8)
        num_encoder_layers = args.get('num_encoder_layers', 3)
        num_decoder_layers = args.get('num_decoder_layers', 3)
        dim_feedforward = args.get('dim_feedforward', 512)
        dropout = args.get('dropout', 0.1)
        
        print(f"  Config: d_model={d_model}, nhead={nhead}, layers={num_encoder_layers}")
        
        # 创建模型
        self.model = create_transformer_model(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device=self.device
        )
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"  ✓ Model loaded successfully")
    
    def evaluate(self, test_loader, src_vocab, tgt_vocab) -> Dict:
        """评估Transformer模型"""
        if self.model is None:
            self.load_model(src_vocab, tgt_vocab)
        
        tgt_vocab_rev = get_reverse_vocab(tgt_vocab)
        
        hypotheses = []
        references = []
        translation_samples = []
        
        print(f"Evaluating Transformer {self.direction}...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Translating")):
                # collate_fn返回元组: (src, tgt, src_lens, tgt_lens)
                src, tgt, src_lens, tgt_lens = batch
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                batch_size = src.size(0)
                
                # 使用贪婪解码
                predictions = self.model.greedy_decode(src, max_len=100)
                
                # 转换为tokens
                for i in range(batch_size):
                    # 预测
                    pred_ids = predictions[i].cpu().tolist()
                    if EOS_IDX in pred_ids:
                        eos_idx = pred_ids.index(EOS_IDX)
                        pred_ids = pred_ids[:eos_idx]
                    pred_tokens = ids_to_tokens(pred_ids, tgt_vocab_rev)
                    hypotheses.append(pred_tokens)
                    
                    # 参考
                    ref_ids = tgt[i].cpu().tolist()
                    if SOS_IDX in ref_ids:
                        ref_ids = ref_ids[1:]
                    if EOS_IDX in ref_ids:
                        eos_idx = ref_ids.index(EOS_IDX)
                        ref_ids = ref_ids[:eos_idx]
                    ref_tokens = ids_to_tokens(ref_ids, tgt_vocab_rev)
                    references.append(ref_tokens)
                    
                    # 保存前5个样例
                    if len(translation_samples) < 5:
                        translation_samples.append({
                            'prediction': tokens_to_string(pred_tokens, self.tgt_lang),
                            'reference': tokens_to_string(ref_tokens, self.tgt_lang)
                        })
        
        # 计算BLEU
        bleu_score = calculate_bleu(hypotheses, references)
        
        return {
            'model': 'Transformer',
            'direction': self.direction,
            'bleu_score': round(bleu_score, 2),
            'num_samples': len(hypotheses),
            'translation_samples': translation_samples
        }


class T5Evaluator(ModelEvaluator):
    """T5模型评估器"""
    
    def __init__(self, model_path: str, direction: str, device: torch.device):
        super().__init__('T5', direction, device)
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """加载T5模型"""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required for T5 evaluation")
        
        print(f"Loading T5 model from {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        except:
            # 如果加载失败，尝试使用MT5Tokenizer
            from transformers import MT5Tokenizer
            self.tokenizer = MT5Tokenizer.from_pretrained(self.model_path)
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"  ✓ Model loaded successfully")
    
    def evaluate(self, test_data: List[Dict]) -> Dict:
        """评估T5模型"""
        if self.model is None:
            self.load_model()
        
        hypotheses = []
        references = []
        translation_samples = []
        
        # 设置任务前缀
        if self.direction == 'en2zh':
            prefix = "translate English to Chinese: "
        else:
            prefix = "translate Chinese to English: "
        
        print(f"Evaluating T5 {self.direction}...")
        
        with torch.no_grad():
            for item in tqdm(test_data[:200], desc="Translating"):  # 测试集200个样本
                # 准备输入
                if self.direction == 'en2zh':
                    src_text = item['en']
                    tgt_text = item['zh']
                else:
                    src_text = item['zh']
                    tgt_text = item['en']
                
                input_text = prefix + src_text
                
                # 编码
                inputs = self.tokenizer(
                    input_text,
                    max_length=128,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                # 生成翻译
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
                
                # 解码
                pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 分词（用于BLEU计算）
                if self.tgt_lang == 'zh':
                    pred_tokens = list(pred_text.replace(' ', ''))
                    ref_tokens = list(tgt_text.replace(' ', ''))
                else:
                    pred_tokens = pred_text.split()
                    ref_tokens = tgt_text.split()
                
                hypotheses.append(pred_tokens)
                references.append(ref_tokens)
                
                # 保存前5个样例
                if len(translation_samples) < 5:
                    translation_samples.append({
                        'prediction': pred_text,
                        'reference': tgt_text
                    })
        
        # 计算BLEU
        bleu_score = calculate_bleu(hypotheses, references)
        
        return {
            'model': 'T5',
            'direction': self.direction,
            'bleu_score': round(bleu_score, 2),
            'num_samples': len(hypotheses),
            'translation_samples': translation_samples
        }


def evaluate_all_models(
    data_dir: str,
    experiments_dir: str,
    results_dir: str,
    device: torch.device
):
    """
    评估所有6个模型
    
    Args:
        data_dir: 数据目录
        experiments_dir: 实验目录
        results_dir: 结果保存目录
        device: 设备
    """
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 模型配置
    model_configs = [
        {
            'type': 'rnn',
            'direction': 'en2zh',
            'checkpoint': f'{experiments_dir}/rnn_en2zh/checkpoints/model_best.pt',
            'name': 'rnn_en2zh'
        },
        {
            'type': 'rnn',
            'direction': 'zh2en',
            'checkpoint': f'{experiments_dir}/rnn_zh2en/checkpoints/model_best.pt',
            'name': 'rnn_zh2en'
        },
        {
            'type': 'transformer',
            'direction': 'en2zh',
            'checkpoint': f'{experiments_dir}/transformer_en2zh/checkpoints/model_best.pt',
            'name': 'transformer_en2zh'
        },
        {
            'type': 'transformer',
            'direction': 'zh2en',
            'checkpoint': f'{experiments_dir}/transformer_zh2en/checkpoints/model_best.pt',
            'name': 'transformer_zh2en'
        },
        {
            'type': 't5',
            'direction': 'en2zh',
            'checkpoint': f'{experiments_dir}/t5_en2zh/best_model',
            'name': 't5_en2zh'
        },
        {
            'type': 't5',
            'direction': 'zh2en',
            'checkpoint': f'{experiments_dir}/t5_zh2en/best_model',
            'name': 't5_zh2en'
        }
    ]
    
    # 加载测试数据
    print("="*70)
    print("Loading test data...")
    print("="*70)
    test_data = load_data(os.path.join(data_dir, 'test.jsonl'))
    print(f"Test samples: {len(test_data)}")
    
    # 评估结果汇总
    all_results = []
    
    # 评估每个模型
    for config in model_configs:
        print("\n" + "="*70)
        print(f"Evaluating {config['name'].upper()}")
        print("="*70)
        
        # 检查checkpoint是否存在
        if not os.path.exists(config['checkpoint']):
            print(f"  ⚠️  Checkpoint not found: {config['checkpoint']}")
            print(f"  Skipping {config['name']}")
            continue
        
        try:
            if config['type'] in ['rnn', 'transformer']:
                # 加载词表
                direction = config['direction']
                if direction == 'en2zh':
                    src_vocab = load_vocab(os.path.join(data_dir, 'vocab_en.json'))
                    tgt_vocab = load_vocab(os.path.join(data_dir, 'vocab_zh.json'))
                    src_lang, tgt_lang = 'en', 'zh'
                else:
                    src_vocab = load_vocab(os.path.join(data_dir, 'vocab_zh.json'))
                    tgt_vocab = load_vocab(os.path.join(data_dir, 'vocab_en.json'))
                    src_lang, tgt_lang = 'zh', 'en'
                
                # 准备数据集
                test_dataset = TranslationDataset(
                    test_data, src_vocab, tgt_vocab, 
                    src_lang, tgt_lang
                )
                
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=32, 
                    shuffle=False,
                    collate_fn=collate_fn
                )
                
                # 创建评估器
                if config['type'] == 'rnn':
                    evaluator = RNNEvaluator(config['checkpoint'], direction, device)
                else:
                    evaluator = TransformerEvaluator(config['checkpoint'], direction, device)
                
                # 评估
                result = evaluator.evaluate(test_loader, src_vocab, tgt_vocab)
                
            else:  # T5
                if not HAS_TRANSFORMERS:
                    print("  ⚠️  Transformers library not available, skipping T5")
                    continue
                
                evaluator = T5Evaluator(config['checkpoint'], config['direction'], device)
                result = evaluator.evaluate(test_data)
            
            # 保存结果
            result['checkpoint'] = config['checkpoint']
            result['evaluation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            all_results.append(result)
            
            # 保存单个模型结果
            result_file = os.path.join(results_dir, f"{config['name']}_results.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ {config['name']} evaluation complete")
            print(f"  BLEU Score: {result['bleu_score']:.2f}")
            print(f"  Results saved to: {result_file}")
            
        except Exception as e:
            print(f"\n✗ Error evaluating {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存汇总结果
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)
    
    summary = {
        'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_models_evaluated': len(all_results),
        'results': all_results
    }
    
    summary_file = os.path.join(results_dir, 'evaluation_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印对比表格
    print("\nModel Performance Comparison:")
    print("-" * 70)
    print(f"{'Model':<20} {'Direction':<10} {'BLEU Score':<12} {'Samples':<10}")
    print("-" * 70)
    
    for result in all_results:
        print(f"{result['model']:<20} {result['direction']:<10} "
              f"{result['bleu_score']:<12.2f} {result['num_samples']:<10}")
    
    print("-" * 70)
    
    # 打印翻译样例（第一个模型）
    if all_results:
        print("\nTranslation Samples (from first model):")
        print("-" * 70)
        for i, sample in enumerate(all_results[0]['translation_samples'][:3], 1):
            print(f"\nSample {i}:")
            print(f"  Prediction: {sample['prediction']}")
            print(f"  Reference:  {sample['reference']}")
    
    print("\n" + "="*70)
    print(f"✓ All results saved to: {results_dir}")
    print(f"✓ Summary saved to: {summary_file}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='评估所有训练好的模型（RNN、Transformer、T5）'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/mnt/afs/250010036/course/NLP/data',
        help='数据目录'
    )
    
    parser.add_argument(
        '--experiments_dir',
        type=str,
        default='/mnt/afs/250010036/course/NLP/experiments',
        help='实验目录'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='/mnt/afs/250010036/course/NLP/results',
        help='结果保存目录'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='设备'
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 评估所有模型
    evaluate_all_models(
        data_dir=args.data_dir,
        experiments_dir=args.experiments_dir,
        results_dir=args.results_dir,
        device=device
    )


if __name__ == '__main__':
    main()

