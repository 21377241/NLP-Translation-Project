#!/usr/bin/env python3
"""
一键推理脚本 - 作业必需文件

用法示例:
    # 使用 RNN 模型进行英译中（贪婪解码）
    python inference.py --model rnn --input "Hello world" --direction en2zh
    
    # 使用 RNN 模型进行英译中（束搜索解码）
    python inference.py --model rnn --input "Hello world" --direction en2zh --beam_size 5
    
    # 使用 Transformer 模型进行中译英
    python inference.py --model transformer --input "你好世界" --direction zh2en
    
    # 使用 T5 预训练模型进行翻译
    python inference.py --model t5 --input "Hello world" --direction en2zh
    
    # 指定 checkpoint 路径
    python inference.py --model rnn --input "Hello" --checkpoint path/to/model.pt
"""

import os
import sys
import argparse
from typing import List, Dict, Optional

import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_utils import (
    tokenize_en, tokenize_zh, load_vocab,
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX,
    get_reverse_vocab, ids_to_tokens, tokens_to_ids
)
from src.models.rnn_seq2seq import create_rnn_model
from src.models.transformer import create_transformer_model


# 默认 checkpoint 路径
DEFAULT_CHECKPOINTS = {
    'rnn': {
        'en2zh': 'experiments/rnn_en2zh/checkpoints/model_best.pt',
        'zh2en': 'experiments/rnn_zh2en/checkpoints/model_best.pt'
    },
    'transformer': {
        'en2zh': 'experiments/transformer_en2zh/checkpoints/model_best.pt',
        'zh2en': 'experiments/transformer_zh2en/checkpoints/model_best.pt'
    },
    't5': {
        'en2zh': 'experiments/t5_en2zh/best_model',
        'zh2en': 'experiments/t5_zh2en/best_model'
    }
}

# 默认词表路径
DEFAULT_VOCAB_DIR = 'data/vocab'


def load_vocabs(direction: str, vocab_dir: str = DEFAULT_VOCAB_DIR) -> tuple:
    """
    加载词表
    
    Args:
        direction: 翻译方向 ('en2zh' 或 'zh2en')
        vocab_dir: 词表目录
        
    Returns:
        (src_vocab, tgt_vocab)
    """
    if direction == 'en2zh':
        src_lang, tgt_lang = 'en', 'zh'
    else:
        src_lang, tgt_lang = 'zh', 'en'
    
    src_vocab_path = os.path.join(vocab_dir, f'vocab_{src_lang}.json')
    tgt_vocab_path = os.path.join(vocab_dir, f'vocab_{tgt_lang}.json')
    
    if not os.path.exists(src_vocab_path):
        raise FileNotFoundError(f"Source vocabulary not found: {src_vocab_path}")
    if not os.path.exists(tgt_vocab_path):
        raise FileNotFoundError(f"Target vocabulary not found: {tgt_vocab_path}")
    
    src_vocab = load_vocab(src_vocab_path)
    tgt_vocab = load_vocab(tgt_vocab_path)
    
    return src_vocab, tgt_vocab


def load_model(model_type: str,
               checkpoint_path: str,
               src_vocab_size: int,
               tgt_vocab_size: int,
               device: torch.device) -> torch.nn.Module:
    """
    加载模型
    
    Args:
        model_type: 模型类型 ('rnn' 或 'transformer')
        checkpoint_path: checkpoint 路径
        src_vocab_size: 源语言词表大小
        tgt_vocab_size: 目标语言词表大小
        device: 设备
        
    Returns:
        加载的模型
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint.get('args', {})
    
    if model_type == 'rnn':
        model = create_rnn_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=model_args.get('embed_dim', 256),
            hidden_dim=model_args.get('hidden_dim', 256),
            n_layers=model_args.get('n_layers', 2),
            dropout=0.0,  # 推理时不使用 dropout
            attn_type=model_args.get('attn_type', 'dot'),
            rnn_type=model_args.get('rnn_type', 'lstm'),
            device=device
        )
    elif model_type == 'transformer':
        model = create_transformer_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=model_args.get('d_model', 256),
            nhead=model_args.get('nhead', 4),
            num_encoder_layers=model_args.get('num_encoder_layers', 3),
            num_decoder_layers=model_args.get('num_decoder_layers', 3),
            dim_feedforward=model_args.get('dim_feedforward', 1024),
            dropout=0.0,  # 推理时不使用 dropout
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def load_t5_model(checkpoint_path: str, src_lang: str, tgt_lang: str, device: torch.device):
    """
    加载 T5/mT5 模型
    
    Args:
        checkpoint_path: T5 模型路径（微调后的模型目录）
        src_lang: 源语言
        tgt_lang: 目标语言
        device: 设备
        
    Returns:
        T5Translator 模型
    """
    try:
        from src.models.t5_finetune import T5Translator
    except ImportError:
        raise ImportError("T5 model requires transformers library. "
                         "Install with: pip install transformers sentencepiece")
    
    # 检查是否存在微调后的模型
    if os.path.exists(checkpoint_path):
        # 检查是否有配置文件
        config_path = os.path.join(checkpoint_path, 'translator_config.json')
        if os.path.exists(config_path):
            # 从微调模型加载
            print(f"Loading fine-tuned model from {checkpoint_path}")
            model = T5Translator.from_pretrained(checkpoint_path, device=device)
        else:
            # 尝试作为 HuggingFace 模型加载
            print(f"Loading model from {checkpoint_path}")
            model = T5Translator(
                model_name=checkpoint_path,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                device=device
            )
    else:
        # 使用本地的 mT5-small 模型
        print(f"Fine-tuned model not found at {checkpoint_path}")
        print("Loading local mT5-small model from ./models/mt5-small...")
        model = T5Translator(
            model_name='./models/mt5-small',
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=device
        )
    
    return model


def translate(text: str,
              model: torch.nn.Module,
              src_vocab: Dict[str, int],
              tgt_vocab: Dict[str, int],
              direction: str,
              device: torch.device,
              max_len: int = 100,
              beam_size: int = 1) -> str:
    """
    翻译单个句子（支持贪婪解码和束搜索）
    
    Args:
        text: 输入文本
        model: 翻译模型
        src_vocab: 源语言词表
        tgt_vocab: 目标语言词表
        direction: 翻译方向
        device: 设备
        max_len: 最大生成长度
        beam_size: 束搜索大小（1 表示贪婪解码）
        
    Returns:
        翻译结果
    """
    model.eval()
    
    # 选择分词器
    if direction == 'en2zh':
        tokenizer = tokenize_en
        join_func = lambda tokens: ''.join(tokens)  # 中文不加空格
    else:
        tokenizer = tokenize_zh
        join_func = lambda tokens: ' '.join(tokens)  # 英文加空格
    
    # 分词
    tokens = tokenizer(text)
    
    # 转换为 ID
    ids = [SOS_IDX] + tokens_to_ids(tokens, src_vocab) + [EOS_IDX]
    src_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    
    # 翻译
    with torch.no_grad():
        if beam_size > 1:
            # 使用束搜索
            translation = model.beam_search(src_tensor, beam_size=beam_size, max_len=max_len)
        else:
            # 使用贪婪解码
            translation = model.translate(src_tensor, max_len=max_len)
    
    # 转换回文本
    tgt_vocab_rev = get_reverse_vocab(tgt_vocab)
    pred_ids = translation[0].cpu().tolist()
    pred_tokens = ids_to_tokens(pred_ids, tgt_vocab_rev)
    
    result = join_func(pred_tokens)
    
    return result


def translate_t5(text: str,
                 model,
                 direction: str,
                 max_len: int = 100,
                 num_beams: int = 4) -> str:
    """
    使用 T5 模型翻译
    
    Args:
        text: 输入文本
        model: T5Translator 模型
        direction: 翻译方向（仅用于显示）
        max_len: 最大生成长度
        num_beams: 束搜索大小
        
    Returns:
        翻译结果
    """
    return model.translate(text, max_length=max_len, num_beams=num_beams)


def batch_translate(texts: List[str],
                    model: torch.nn.Module,
                    src_vocab: Dict[str, int],
                    tgt_vocab: Dict[str, int],
                    direction: str,
                    device: torch.device,
                    max_len: int = 100,
                    beam_size: int = 1) -> List[str]:
    """
    批量翻译
    
    Args:
        texts: 输入文本列表
        model: 翻译模型
        src_vocab: 源语言词表
        tgt_vocab: 目标语言词表
        direction: 翻译方向
        device: 设备
        max_len: 最大生成长度
        beam_size: 束搜索大小
        
    Returns:
        翻译结果列表
    """
    results = []
    for text in texts:
        result = translate(text, model, src_vocab, tgt_vocab, direction, device, max_len, beam_size)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Translation Inference Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Translate English to Chinese using RNN model (greedy decoding)
    python inference.py --model rnn --input "Hello world" --direction en2zh
    
    # Translate with beam search
    python inference.py --model rnn --input "Hello world" --direction en2zh --beam_size 5
    
    # Translate Chinese to English using Transformer model
    python inference.py --model transformer --input "你好世界" --direction zh2en
    
    # Use T5 pre-trained model
    python inference.py --model t5 --input "Hello world" --direction en2zh
    
    # Use custom checkpoint
    python inference.py --model rnn --input "Hello" --checkpoint path/to/model.pt
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['rnn', 'transformer', 't5'],
                        help='Model type (rnn, transformer, or t5)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input text to translate')
    parser.add_argument('--direction', type=str, default='en2zh',
                        choices=['en2zh', 'zh2en'],
                        help='Translation direction (default: en2zh)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--vocab_dir', type=str, default=DEFAULT_VOCAB_DIR,
                        help='Vocabulary directory (not used for T5)')
    parser.add_argument('--max_len', type=int, default=100,
                        help='Maximum output length')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='Beam search size (1 for greedy decoding)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, default: auto)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Direction: {args.direction}")
    print(f"Decoding: {'Beam search (size=' + str(args.beam_size) + ')' if args.beam_size > 1 else 'Greedy'}")
    
    # T5 模型使用自己的分词器，不需要加载词表
    if args.model == 't5':
        # 确定源语言和目标语言
        if args.direction == 'en2zh':
            src_lang, tgt_lang = 'en', 'zh'
        else:
            src_lang, tgt_lang = 'zh', 'en'
        
        # 确定 checkpoint 路径
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = DEFAULT_CHECKPOINTS['t5'][args.direction]
        
        # 加载 T5 模型
        print(f"\nLoading T5 model...")
        try:
            model = load_t5_model(checkpoint_path, src_lang, tgt_lang, device)
        except ImportError as e:
            print(f"Error: {e}")
            return
        except Exception as e:
            print(f"Error loading T5 model: {e}")
            return
        
        # 翻译
        print("\n" + "=" * 50)
        print("Translation:")
        print("=" * 50)
        
        result = translate_t5(
            text=args.input,
            model=model,
            direction=args.direction,
            max_len=args.max_len,
            num_beams=args.beam_size if args.beam_size > 1 else 4
        )
        
        print(f"\nInput:  {args.input}")
        print(f"Output: {result}")
        print("=" * 50)
    
    else:
        # RNN / Transformer 模型
        
        # 加载词表
        print("\nLoading vocabularies...")
        try:
            src_vocab, tgt_vocab = load_vocabs(args.direction, args.vocab_dir)
            print(f"  Source vocab size: {len(src_vocab)}")
            print(f"  Target vocab size: {len(tgt_vocab)}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nPlease make sure vocabularies exist. You can generate them by running:")
            print("  python src/data_utils.py")
            return
        
        # 确定 checkpoint 路径
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = DEFAULT_CHECKPOINTS[args.model][args.direction]
        
        # 加载模型
        print(f"\nLoading model from: {checkpoint_path}")
        try:
            model = load_model(
                model_type=args.model,
                checkpoint_path=checkpoint_path,
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                device=device
            )
            print("Model loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nPlease train the model first or specify a valid checkpoint path.")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # 翻译
        print("\n" + "=" * 50)
        print("Translation:")
        print("=" * 50)
        
        result = translate(
            text=args.input,
            model=model,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            direction=args.direction,
            device=device,
            max_len=args.max_len,
            beam_size=args.beam_size
        )
        
        print(f"\nInput:  {args.input}")
        print(f"Output: {result}")
        print("=" * 50)


if __name__ == '__main__':
    main()
