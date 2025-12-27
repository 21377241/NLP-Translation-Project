"""
RNN Ablation Study Visualization Script

Features:
1. Attention mechanism comparison visualization
2. Training strategy comparison visualization
3. Decoding strategy comparison visualization
4. Training curves comparison
5. Comprehensive comparison report

Author: NLP Course Project
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False  # Handle minus sign display

import numpy as np


# ============================================================
# Color Scheme
# ============================================================

COLORS = {
    'attention': {
        'dot': '#2ecc71',           # green
        'multiplicative': '#3498db', # blue
        'additive': '#e74c3c'        # red
    },
    'training_strategy': {
        'teacher_forcing': '#9b59b6',     # purple
        'scheduled_sampling': '#f39c12',  # orange
        'free_running': '#1abc9c'         # cyan
    },
    'decoding': {
        'greedy': '#34495e',    # dark gray
        'beam_3': '#3498db',    # blue
        'beam_5': '#2ecc71',    # green
        'beam_10': '#e74c3c'    # red
    }
}

ATTENTION_NAMES = {
    'dot': 'Dot-Product Attention',
    'multiplicative': 'Multiplicative Attention',
    'additive': 'Additive Attention'
}

STRATEGY_NAMES = {
    'teacher_forcing': 'Teacher Forcing',
    'scheduled_sampling': 'Scheduled Sampling',
    'free_running': 'Free Running'
}

DECODING_NAMES = {
    'greedy': 'Greedy Decoding',
    'beam_3': 'Beam Search (k=3)',
    'beam_5': 'Beam Search (k=5)',
    'beam_10': 'Beam Search (k=10)'
}


# ============================================================
# Data Loading
# ============================================================

def load_train_logs(experiment_dir: str) -> Dict[str, Dict]:
    """Load training logs from all experiments"""
    logs = {}
    
    if not os.path.exists(experiment_dir):
        return logs
    
    for exp_name in os.listdir(experiment_dir):
        exp_path = os.path.join(experiment_dir, exp_name)
        log_path = os.path.join(exp_path, 'logs', 'train_log.json')
        
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                logs[exp_name] = json.load(f)
    
    return logs


def load_evaluation_report(report_path: str) -> Optional[Dict]:
    """Load evaluation report"""
    if not os.path.exists(report_path):
        return None
    
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# Visualization Functions
# ============================================================

def plot_training_curves(logs: Dict[str, Dict], output_path: str, title: str = "Training Curves Comparison"):
    """
    Plot training curves comparison
    
    Args:
        logs: Experiment log dictionary
        output_path: Output image path
        title: Chart title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Determine colors
    colors = list(plt.cm.tab10.colors)
    
    for idx, (exp_name, log) in enumerate(logs.items()):
        color = colors[idx % len(colors)]
        epochs = range(1, len(log['train_losses']) + 1)
        
        # Plot training loss
        axes[0].plot(epochs, log['train_losses'], 
                    label=exp_name, color=color, linewidth=2)
        
        # Plot validation loss
        axes[1].plot(epochs, log['valid_losses'], 
                    label=exp_name, color=color, linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Train Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Valid Loss', fontsize=12)
    axes[1].set_title('Validation Loss', fontsize=14)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {output_path}")


def plot_attention_comparison(eval_report: Dict, output_path: str):
    """
    Plot attention mechanism comparison bar chart
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract data
    exp_names = []
    bleu_greedy = []
    bleu_beam = []
    times = []
    
    for exp_name, results in eval_report.get('detailed_results', {}).items():
        if 'attn_' in exp_name:
            attn_type = exp_name.replace('attn_', '')
            display_name = ATTENTION_NAMES.get(attn_type, attn_type)
            exp_names.append(display_name)
            
            decoding_results = results.get('decoding_results', {})
            bleu_greedy.append(decoding_results.get('greedy', {}).get('bleu', 0))
            bleu_beam.append(decoding_results.get('beam_5', {}).get('bleu', 0))
            times.append(decoding_results.get('greedy', {}).get('avg_inference_time_ms', 0))
    
    if not exp_names:
        print("No attention mechanism experiment data found")
        return
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    # BLEU comparison
    bars1 = axes[0].bar(x - width/2, bleu_greedy, width, label='Greedy Decoding', color='#3498db')
    bars2 = axes[0].bar(x + width/2, bleu_beam, width, label='Beam Search (k=5)', color='#2ecc71')
    
    axes[0].set_ylabel('BLEU Score', fontsize=12)
    axes[0].set_title('Attention Mechanism BLEU Comparison', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(exp_names, fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Inference time comparison
    colors = [COLORS['attention'].get(exp.replace('attn_', ''), '#95a5a6') 
              for exp in eval_report.get('detailed_results', {}).keys() if 'attn_' in exp]
    bars3 = axes[1].bar(x, times, color=colors if colors else '#3498db')
    
    axes[1].set_ylabel('Inference Time (ms)', fontsize=12)
    axes[1].set_title('Attention Mechanism Inference Time Comparison', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(exp_names, fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar in bars3:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}ms',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Attention Mechanism Ablation Study Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Attention mechanism comparison chart saved to: {output_path}")


def plot_training_strategy_comparison(eval_report: Dict, output_path: str):
    """
    Plot training strategy comparison bar chart
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract data
    exp_names = []
    bleu_greedy = []
    bleu_beam = []
    
    for exp_name, results in eval_report.get('detailed_results', {}).items():
        if 'strategy_' in exp_name:
            strategy_type = exp_name.replace('strategy_', '')
            display_name = STRATEGY_NAMES.get(strategy_type, strategy_type)
            exp_names.append(display_name)
            
            decoding_results = results.get('decoding_results', {})
            bleu_greedy.append(decoding_results.get('greedy', {}).get('bleu', 0))
            bleu_beam.append(decoding_results.get('beam_5', {}).get('bleu', 0))
    
    if not exp_names:
        print("No training strategy experiment data found")
        return
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    # BLEU comparison
    bars1 = axes[0].bar(x - width/2, bleu_greedy, width, label='Greedy Decoding', color='#9b59b6')
    bars2 = axes[0].bar(x + width/2, bleu_beam, width, label='Beam Search (k=5)', color='#f39c12')
    
    axes[0].set_ylabel('BLEU Score', fontsize=12)
    axes[0].set_title('Training Strategy BLEU Comparison', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(exp_names, fontsize=10, rotation=15, ha='right')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Best validation loss comparison
    valid_losses = []
    for exp_name, results in eval_report.get('detailed_results', {}).items():
        if 'strategy_' in exp_name:
            # Get best_valid_loss from training log
            valid_losses.append(results.get('model_args', {}).get('best_valid_loss', 0))
    
    if valid_losses and any(v > 0 for v in valid_losses):
        colors = ['#9b59b6', '#f39c12', '#1abc9c'][:len(exp_names)]
        bars3 = axes[1].bar(x, valid_losses, color=colors)
        axes[1].set_ylabel('Best Validation Loss', fontsize=12)
        axes[1].set_title('Training Strategy Validation Loss Comparison', fontsize=14)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(exp_names, fontsize=10, rotation=15, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
    else:
        # If no valid_loss data, show BLEU difference between greedy and beam search
        diff = [b - g for g, b in zip(bleu_greedy, bleu_beam)]
        colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in diff]
        bars3 = axes[1].bar(x, diff, color=colors)
        axes[1].set_ylabel('BLEU Difference (Beam - Greedy)', fontsize=12)
        axes[1].set_title('Decoding Strategy Improvement', fontsize=14)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(exp_names, fontsize=10, rotation=15, ha='right')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Training Strategy Ablation Study Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training strategy comparison chart saved to: {output_path}")


def plot_decoding_comparison(eval_report: Dict, output_path: str):
    """
    Plot decoding strategy comparison chart
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Collect decoding strategy results from all experiments
    all_results = defaultdict(dict)
    
    for exp_name, results in eval_report.get('detailed_results', {}).items():
        decoding_results = results.get('decoding_results', {})
        for strategy, data in decoding_results.items():
            all_results[strategy][exp_name] = {
                'bleu': data.get('bleu', 0),
                'time': data.get('avg_inference_time_ms', 0)
            }
    
    strategies = list(all_results.keys())
    experiments = list(next(iter(all_results.values())).keys()) if all_results else []
    
    if not strategies or not experiments:
        print("No decoding strategy comparison data found")
        return
    
    # BLEU comparison - grouped by decoding strategy
    x = np.arange(len(strategies))
    width = 0.8 / len(experiments)
    
    for i, exp_name in enumerate(experiments):
        bleu_scores = [all_results[s].get(exp_name, {}).get('bleu', 0) for s in strategies]
        offset = (i - len(experiments)/2 + 0.5) * width
        bars = axes[0].bar(x + offset, bleu_scores, width, label=exp_name)
    
    axes[0].set_ylabel('BLEU Score', fontsize=12)
    axes[0].set_title('Decoding Strategy BLEU Comparison', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DECODING_NAMES.get(s, s) for s in strategies], fontsize=10)
    axes[0].legend(loc='upper left', fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Inference time comparison
    for i, exp_name in enumerate(experiments):
        times = [all_results[s].get(exp_name, {}).get('time', 0) for s in strategies]
        offset = (i - len(experiments)/2 + 0.5) * width
        bars = axes[1].bar(x + offset, times, width, label=exp_name)
    
    axes[1].set_ylabel('Inference Time (ms)', fontsize=12)
    axes[1].set_title('Decoding Strategy Inference Time Comparison', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DECODING_NAMES.get(s, s) for s in strategies], fontsize=10)
    axes[1].legend(loc='upper left', fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Decoding Strategy Ablation Study Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Decoding strategy comparison chart saved to: {output_path}")


def plot_comprehensive_heatmap(eval_report: Dict, output_path: str):
    """
    Plot comprehensive heatmap
    """
    # Prepare data
    experiments = list(eval_report.get('detailed_results', {}).keys())
    strategies = ['greedy', 'beam_3', 'beam_5', 'beam_10']
    
    if not experiments:
        print("No experiment data found")
        return
    
    # Create BLEU matrix
    bleu_matrix = np.zeros((len(experiments), len(strategies)))
    
    for i, exp_name in enumerate(experiments):
        results = eval_report['detailed_results'][exp_name]
        decoding_results = results.get('decoding_results', {})
        for j, strategy in enumerate(strategies):
            bleu_matrix[i, j] = decoding_results.get(strategy, {}).get('bleu', 0)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(experiments) * 0.5)))
    
    im = ax.imshow(bleu_matrix, cmap='YlGnBu', aspect='auto')
    
    # Set axes
    ax.set_xticks(np.arange(len(strategies)))
    ax.set_yticks(np.arange(len(experiments)))
    ax.set_xticklabels([DECODING_NAMES.get(s, s) for s in strategies], fontsize=11)
    ax.set_yticklabels(experiments, fontsize=10)
    
    # Add value annotations
    for i in range(len(experiments)):
        for j in range(len(strategies)):
            text = ax.text(j, i, f'{bleu_matrix[i, j]:.1f}',
                          ha='center', va='center', fontsize=10,
                          color='white' if bleu_matrix[i, j] > bleu_matrix.max() * 0.6 else 'black')
    
    ax.set_title('BLEU Score Heatmap\n(Experiment x Decoding Strategy)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('BLEU Score', rotation=-90, va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive heatmap saved to: {output_path}")


def generate_all_visualizations(experiment_dir: str, output_dir: str):
    """
    Generate all visualization charts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training logs
    train_logs = load_train_logs(experiment_dir)
    
    # Load evaluation report
    eval_report_path = os.path.join(experiment_dir, 'evaluation_report.json')
    eval_report = load_evaluation_report(eval_report_path)
    
    # Generate training curves
    if train_logs:
        plot_training_curves(
            train_logs,
            os.path.join(output_dir, 'training_curves.png'),
            title='Ablation Study Training Curves Comparison'
        )
    
    if eval_report:
        # Check for attention mechanism experiments
        if any('attn_' in exp for exp in eval_report.get('detailed_results', {})):
            plot_attention_comparison(
                eval_report,
                os.path.join(output_dir, 'attention_comparison.png')
            )
        
        # Check for training strategy experiments
        if any('strategy_' in exp for exp in eval_report.get('detailed_results', {})):
            plot_training_strategy_comparison(
                eval_report,
                os.path.join(output_dir, 'training_strategy_comparison.png')
            )
        
        # Decoding strategy comparison
        plot_decoding_comparison(
            eval_report,
            os.path.join(output_dir, 'decoding_comparison.png')
        )
        
        # Comprehensive heatmap
        plot_comprehensive_heatmap(
            eval_report,
            os.path.join(output_dir, 'bleu_heatmap.png')
        )
    
    print(f"\nAll visualization charts saved to: {output_dir}")


def main(args):
    """Main function"""
    print("="*60)
    print("RNN Ablation Study Visualization")
    print("="*60)
    
    if args.mode == 'single':
        # Visualize single experiment directory
        generate_all_visualizations(args.experiment_dir, args.output_dir)
    
    elif args.mode == 'all':
        # Visualize all ablation experiments
        base_dir = args.experiment_dir
        
        for direction in ['en2zh', 'zh2en']:
            for exp_type in ['attention_ablation', 'training_strategy_ablation', 'full_ablation']:
                exp_dir = os.path.join(base_dir, direction, exp_type)
                if os.path.exists(exp_dir):
                    output_dir = os.path.join(args.output_dir, direction, exp_type)
                    print(f"\n>>> Processing: {exp_dir}")
                    generate_all_visualizations(exp_dir, output_dir)
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNN Ablation Study Visualization')
    
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'all'],
                        help='Visualization mode: single (single directory), all (all experiments)')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Experiment directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: visualizations subdirectory under experiment_dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, 'visualizations')
    
    main(args)

