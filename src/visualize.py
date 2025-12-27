"""
可视化工具模块
用于生成训练曲线、BLEU 对比图、注意力热力图等
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization functions will not work.")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def setup_chinese_font():
    """设置中文字体支持"""
    if not HAS_MATPLOTLIB:
        return
    
    # 尝试使用系统中的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 
                     'Noto Sans CJK SC', 'Arial Unicode MS']
    
    for font_name in chinese_fonts:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path and os.path.exists(font_path):
                plt.rcParams['font.family'] = font_name
                plt.rcParams['axes.unicode_minus'] = False
                return
        except:
            continue
    
    # 如果没有找到中文字体，使用默认字体
    print("Warning: No Chinese font found. Chinese characters may not display correctly.")


def plot_training_curves(history: Union[Dict, str],
                         save_path: Optional[str] = None,
                         title: str = 'Training Curves',
                         show: bool = True) -> None:
    """
    绘制训练曲线（损失和 BLEU）
    
    Args:
        history: 训练历史字典或 JSON 文件路径
        save_path: 保存路径（可选）
        title: 图表标题
        show: 是否显示图表
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Cannot plot.")
        return
    
    setup_chinese_font()
    
    # 加载历史数据
    if isinstance(history, str):
        with open(history, 'r') as f:
            history = json.load(f)
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # 绘制损失曲线
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'valid_loss' in history:
        ax1.plot(epochs, history['valid_loss'], 'r--', label='Valid Loss', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制 BLEU 曲线
    ax2 = axes[1]
    if 'valid_bleu' in history:
        ax2.plot(epochs, history['valid_bleu'], 'g-', label='Valid BLEU', linewidth=2, marker='o')
    if 'train_bleu' in history:
        ax2.plot(epochs, history['train_bleu'], 'b--', label='Train BLEU', linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('BLEU Score', fontsize=12)
    ax2.set_title('BLEU Score', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_bleu_comparison(results: Dict[str, float],
                         save_path: Optional[str] = None,
                         title: str = 'Model BLEU Comparison',
                         show: bool = True) -> None:
    """
    绘制不同模型的 BLEU 分数对比柱状图
    
    Args:
        results: 模型名称到 BLEU 分数的字典
        save_path: 保存路径（可选）
        title: 图表标题
        show: 是否显示图表
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Cannot plot.")
        return
    
    setup_chinese_font()
    
    models = list(results.keys())
    bleu_scores = list(results.values())
    
    # 设置颜色
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, bleu_scores, color=colors, edgecolor='black', linewidth=1.2)
    
    # 添加数值标签
    for bar, score in zip(bars, bleu_scores):
        height = bar.get_height()
        ax.annotate(f'{score:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('BLEU-4 Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, max(bleu_scores) * 1.15)  # 留出空间显示标签
    
    # 添加网格
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"BLEU comparison saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_attention_heatmap(attention: np.ndarray,
                           src_tokens: List[str],
                           tgt_tokens: List[str],
                           save_path: Optional[str] = None,
                           title: str = 'Attention Weights',
                           show: bool = True) -> None:
    """
    绘制注意力权重热力图
    
    Args:
        attention: [tgt_len, src_len] 注意力权重矩阵
        src_tokens: 源序列 token 列表
        tgt_tokens: 目标序列 token 列表
        save_path: 保存路径（可选）
        title: 图表标题
        show: 是否显示图表
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Cannot plot.")
        return
    
    setup_chinese_font()
    
    # 确保维度匹配
    attention = np.array(attention)
    if attention.shape[0] != len(tgt_tokens) or attention.shape[1] != len(src_tokens):
        # 裁剪或填充
        attention = attention[:len(tgt_tokens), :len(src_tokens)]
    
    fig, ax = plt.subplots(figsize=(max(10, len(src_tokens) * 0.5), 
                                     max(8, len(tgt_tokens) * 0.4)))
    
    if HAS_SEABORN:
        sns.heatmap(attention, 
                    xticklabels=src_tokens,
                    yticklabels=tgt_tokens,
                    cmap='Blues',
                    annot=len(src_tokens) <= 15,  # 只在小矩阵显示数值
                    fmt='.2f' if len(src_tokens) <= 15 else '',
                    ax=ax)
    else:
        im = ax.imshow(attention, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(src_tokens)))
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_xticklabels(src_tokens)
        ax.set_yticklabels(tgt_tokens)
        plt.colorbar(im, ax=ax)
    
    ax.set_xlabel('Source Tokens', fontsize=12)
    ax.set_ylabel('Target Tokens', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention heatmap saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_rate(learning_rates: List[float],
                       save_path: Optional[str] = None,
                       title: str = 'Learning Rate Schedule',
                       show: bool = True) -> None:
    """
    绘制学习率变化曲线
    
    Args:
        learning_rates: 学习率列表
        save_path: 保存路径（可选）
        title: 图表标题
        show: 是否显示图表
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Cannot plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    steps = range(1, len(learning_rates) + 1)
    ax.plot(steps, learning_rates, 'b-', linewidth=1.5)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 使用科学计数法
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Learning rate curve saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_decoding_comparison(results: Dict[str, Dict[str, float]],
                             save_path: Optional[str] = None,
                             title: str = 'Decoding Strategy Comparison',
                             show: bool = True) -> None:
    """
    绘制不同解码策略的对比图
    
    Args:
        results: 模型 -> {策略: BLEU分数} 的嵌套字典
        save_path: 保存路径（可选）
        title: 图表标题
        show: 是否显示图表
        
    Example:
        results = {
            'RNN': {'Greedy': 12.5, 'Beam-3': 14.2, 'Beam-5': 14.8},
            'Transformer': {'Greedy': 18.3, 'Beam-3': 20.1, 'Beam-5': 20.5}
        }
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Cannot plot.")
        return
    
    setup_chinese_font()
    
    models = list(results.keys())
    strategies = list(results[models[0]].keys())
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    for i, strategy in enumerate(strategies):
        scores = [results[model][strategy] for model in models]
        offset = (i - len(strategies) / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=strategy, color=colors[i % len(colors)])
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.annotate(f'{score:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('BLEU-4 Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Decoding comparison saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_hyperparameter_sensitivity(results: Dict[str, List[Tuple[float, float]]],
                                    param_name: str,
                                    save_path: Optional[str] = None,
                                    title: Optional[str] = None,
                                    show: bool = True) -> None:
    """
    绘制超参数敏感性分析图
    
    Args:
        results: 模型名 -> [(参数值, BLEU分数), ...] 的字典
        param_name: 超参数名称
        save_path: 保存路径（可选）
        title: 图表标题
        show: 是否显示图表
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Cannot plot.")
        return
    
    setup_chinese_font()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'd', 'v', '<', '>']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for i, (model_name, data) in enumerate(results.items()):
        param_values = [x[0] for x in data]
        bleu_scores = [x[1] for x in data]
        
        ax.plot(param_values, bleu_scores, 
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=2, markersize=8,
                label=model_name)
    
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('BLEU-4 Score', fontsize=12)
    ax.set_title(title or f'{param_name} Sensitivity Analysis', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sensitivity analysis saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_all_visualizations(experiment_dir: str, output_dir: Optional[str] = None):
    """
    根据实验目录生成所有可视化
    
    Args:
        experiment_dir: 实验目录（包含 training_history.json 等）
        output_dir: 输出目录（默认为实验目录下的 visualizations/）
    """
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载训练历史
    history_path = os.path.join(experiment_dir, 'training_history.json')
    if os.path.exists(history_path):
        print("Generating training curves...")
        plot_training_curves(
            history_path,
            save_path=os.path.join(output_dir, 'training_curves.png'),
            show=False
        )
    
    # 生成学习率曲线
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        if 'learning_rate' in history:
            print("Generating learning rate curve...")
            plot_learning_rate(
                history['learning_rate'],
                save_path=os.path.join(output_dir, 'learning_rate.png'),
                show=False
            )
    
    print(f"\nVisualizations saved to {output_dir}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='Visualization Tools for NMT')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # 训练曲线
    train_parser = subparsers.add_parser('training', help='Plot training curves')
    train_parser.add_argument('--history', type=str, required=True,
                              help='Path to training history JSON file')
    train_parser.add_argument('--output', type=str, default='training_curves.png',
                              help='Output file path')
    train_parser.add_argument('--title', type=str, default='Training Curves',
                              help='Plot title')
    
    # BLEU 对比
    bleu_parser = subparsers.add_parser('bleu', help='Plot BLEU comparison')
    bleu_parser.add_argument('--results', type=str, required=True,
                             help='Path to results JSON file')
    bleu_parser.add_argument('--output', type=str, default='bleu_comparison.png',
                             help='Output file path')
    bleu_parser.add_argument('--title', type=str, default='Model BLEU Comparison',
                             help='Plot title')
    
    # 注意力热力图
    attn_parser = subparsers.add_parser('attention', help='Plot attention heatmap')
    attn_parser.add_argument('--attention', type=str, required=True,
                             help='Path to attention weights (.npy)')
    attn_parser.add_argument('--src_tokens', type=str, required=True,
                             help='Source tokens (comma-separated)')
    attn_parser.add_argument('--tgt_tokens', type=str, required=True,
                             help='Target tokens (comma-separated)')
    attn_parser.add_argument('--output', type=str, default='attention.png',
                             help='Output file path')
    
    # 生成所有
    all_parser = subparsers.add_parser('all', help='Generate all visualizations')
    all_parser.add_argument('--exp_dir', type=str, required=True,
                            help='Experiment directory')
    all_parser.add_argument('--output_dir', type=str, default=None,
                            help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'training':
        plot_training_curves(args.history, args.output, args.title, show=False)
    
    elif args.command == 'bleu':
        with open(args.results, 'r') as f:
            results = json.load(f)
        plot_bleu_comparison(results, args.output, args.title, show=False)
    
    elif args.command == 'attention':
        attention = np.load(args.attention)
        src_tokens = args.src_tokens.split(',')
        tgt_tokens = args.tgt_tokens.split(',')
        plot_attention_heatmap(attention, src_tokens, tgt_tokens, args.output, show=False)
    
    elif args.command == 'all':
        generate_all_visualizations(args.exp_dir, args.output_dir)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

