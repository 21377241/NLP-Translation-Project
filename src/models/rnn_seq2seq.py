"""
RNN Seq2Seq 机器翻译模型
包含Encoder、Attention、Decoder和完整的Seq2Seq模型
符合作业要求：单向2层LSTM/GRU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# 导入特殊token索引
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import PAD_IDX, SOS_IDX, EOS_IDX


class Encoder(nn.Module):
    """
    单向RNN编码器（符合作业要求）
    
    输入：源语言序列 [batch, src_len]
    输出：编码器输出 [batch, src_len, hidden_dim], 最终隐藏状态
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int = 256,
                 hidden_dim: int = 256,
                 n_layers: int = 2,
                 dropout: float = 0.3,
                 rnn_type: str = 'lstm'):
        """
        Args:
            vocab_size: 词表大小
            embed_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
            n_layers: RNN层数
            dropout: Dropout比率
            rnn_type: RNN类型 ('lstm' 或 'gru')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type.lower()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        
        # 选择RNN类型
        rnn_class = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_class(
            embed_dim, 
            hidden_dim, 
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False  # 单向（符合作业要求）
        )
    
    def forward(self, src: torch.Tensor, src_lens: Optional[torch.Tensor] = None) -> Tuple:
        """
        Args:
            src: [batch, src_len] 源序列
            src_lens: [batch] 源序列长度（可选，用于pack_padded_sequence）
            
        Returns:
            outputs: [batch, src_len, hidden_dim] 编码器输出
            hidden: 最终隐藏状态
        """
        # 词嵌入
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, embed_dim]
        
        # RNN编码
        if src_lens is not None:
            # 使用pack_padded_sequence加速
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, hidden = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, hidden = self.rnn(embedded)
        
        return outputs, hidden


class Attention(nn.Module):
    """
    注意力机制
    支持三种类型：点积(dot)、加性(additive)、乘性(multiplicative)
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 attn_type: str = 'dot'):
        """
        Args:
            hidden_dim: 隐藏层维度
            attn_type: 注意力类型 ('dot', 'additive', 'multiplicative')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.attn_type = attn_type.lower()
        
        if self.attn_type == 'additive':
            # Bahdanau注意力
            self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v = nn.Linear(hidden_dim, 1, bias=False)
        elif self.attn_type == 'multiplicative':
            # Luong乘性注意力
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, 
                query: torch.Tensor, 
                keys: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, hidden_dim] 解码器当前隐藏状态
            keys: [batch, src_len, hidden_dim] 编码器输出
            mask: [batch, src_len] 填充掩码（True表示需要mask的位置）
            
        Returns:
            context: [batch, hidden_dim] 上下文向量
            attn_weights: [batch, src_len] 注意力权重
        """
        batch_size, src_len, _ = keys.size()
        
        if self.attn_type == 'dot':
            # 点积注意力
            # query: [batch, hidden] -> [batch, hidden, 1]
            # keys: [batch, src_len, hidden]
            scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)  # [batch, src_len]
            
        elif self.attn_type == 'additive':
            # 加性注意力 (Bahdanau)
            query_proj = self.W1(query).unsqueeze(1)  # [batch, 1, hidden]
            keys_proj = self.W2(keys)  # [batch, src_len, hidden]
            scores = self.v(torch.tanh(query_proj + keys_proj)).squeeze(2)  # [batch, src_len]
            
        elif self.attn_type == 'multiplicative':
            # 乘性注意力 (Luong)
            query_proj = self.W(query)  # [batch, hidden]
            scores = torch.bmm(keys, query_proj.unsqueeze(2)).squeeze(2)  # [batch, src_len]
        
        else:
            raise ValueError(f"Unknown attention type: {self.attn_type}")
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=1)  # [batch, src_len]
        
        # 计算上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)  # [batch, hidden]
        
        return context, attn_weights


class Decoder(nn.Module):
    """
    带注意力的单向RNN解码器（符合作业要求）
    """
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 256,
                 hidden_dim: int = 256,
                 n_layers: int = 2,
                 dropout: float = 0.3,
                 attn_type: str = 'dot',
                 rnn_type: str = 'lstm'):
        """
        Args:
            vocab_size: 词表大小
            embed_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
            n_layers: RNN层数
            dropout: Dropout比率
            attn_type: 注意力类型
            rnn_type: RNN类型
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type.lower()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        
        # 注意力机制
        self.attention = Attention(hidden_dim, attn_type)
        
        # RNN输入：embedding + context
        rnn_class = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_class(
            embed_dim + hidden_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self,
                input_token: torch.Tensor,
                hidden: Tuple,
                encoder_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple, torch.Tensor]:
        """
        单步解码
        
        Args:
            input_token: [batch] 当前输入token
            hidden: 当前隐藏状态
            encoder_outputs: [batch, src_len, hidden_dim] 编码器输出
            mask: [batch, src_len] 填充掩码
            
        Returns:
            output: [batch, vocab_size] 输出logits
            hidden: 新的隐藏状态
            attn_weights: [batch, src_len] 注意力权重
        """
        # 词嵌入
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))  # [batch, 1, embed_dim]
        
        # 获取当前解码器状态（用于注意力）
        if self.rnn_type == 'lstm':
            query = hidden[0][-1]  # 取最后一层的隐藏状态 [batch, hidden]
        else:
            query = hidden[-1]  # [batch, hidden]
        
        # 计算注意力
        context, attn_weights = self.attention(query, encoder_outputs, mask)
        
        # 拼接embedding和context
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # [batch, 1, embed+hidden]
        
        # RNN解码
        output, hidden = self.rnn(rnn_input, hidden)  # output: [batch, 1, hidden]
        
        # 输出投影
        output = self.fc_out(output.squeeze(1))  # [batch, vocab_size]
        
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    """
    完整的Seq2Seq翻译模型
    """
    
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 device: torch.device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_lens: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        """
        训练时的前向传播
        
        Args:
            src: [batch, src_len] 源序列
            tgt: [batch, tgt_len] 目标序列
            src_lens: [batch] 源序列长度
            teacher_forcing_ratio: Teacher Forcing比率
            
        Returns:
            outputs: [batch, tgt_len-1, vocab_size] 输出logits
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.vocab_size
        
        # 存储输出
        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size).to(self.device)
        
        # 编码
        encoder_outputs, hidden = self.encoder(src, src_lens)
        
        # 创建源序列掩码
        src_mask = (src == PAD_IDX)
        
        # 初始输入为<SOS>
        input_token = tgt[:, 0]
        
        # 逐步解码
        for t in range(1, tgt_len):
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, src_mask)
            outputs[:, t - 1, :] = output
            
            # Teacher Forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt[:, t]  # 使用真实标签
            else:
                input_token = output.argmax(dim=1)  # 使用预测结果
        
        return outputs
    
    def translate(self,
                  src: torch.Tensor,
                  src_lens: Optional[torch.Tensor] = None,
                  max_len: int = 100,
                  return_attention: bool = False,
                  repetition_penalty: float = 1.2) -> Tuple:
        """
        推理时的翻译（贪婪解码 + 重复惩罚）
        
        Args:
            src: [batch, src_len] 源序列
            src_lens: [batch] 源序列长度
            max_len: 最大生成长度
            return_attention: 是否返回注意力权重
            repetition_penalty: 重复惩罚系数（>1减少重复，建议1.2-2.0）
            
        Returns:
            translations: [batch, max_len] 翻译结果
            attention_weights: [batch, max_len, src_len] 注意力权重（可选）
        """
        self.eval()
        batch_size = src.size(0)
        
        with torch.no_grad():
            # 编码
            encoder_outputs, hidden = self.encoder(src, src_lens)
            
            # 创建源序列掩码
            src_mask = (src == PAD_IDX)
            
            # 存储翻译结果和注意力
            translations = torch.full((batch_size, max_len), PAD_IDX, dtype=torch.long).to(self.device)
            translations[:, 0] = SOS_IDX
            
            if return_attention:
                attention_weights = torch.zeros(batch_size, max_len, src.size(1)).to(self.device)
            
            # 初始输入
            input_token = torch.full((batch_size,), SOS_IDX, dtype=torch.long).to(self.device)
            
            # 记录是否已生成EOS
            finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            
            for t in range(1, max_len):
                output, hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs, src_mask)
                
                # 应用重复惩罚
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        # 获取已生成的token
                        generated_tokens = translations[i, :t]
                        # 对已出现的token进行惩罚
                        for token in generated_tokens:
                            if token != PAD_IDX and token != SOS_IDX:
                                # 如果logit>0，除以penalty；如果<0，乘以penalty
                                if output[i, token] < 0:
                                    output[i, token] *= repetition_penalty
                                else:
                                    output[i, token] /= repetition_penalty
                
                # 贪婪解码
                next_token = output.argmax(dim=1)
                translations[:, t] = next_token
                
                if return_attention:
                    attention_weights[:, t, :attn_weights.size(1)] = attn_weights
                
                # 检查是否生成EOS
                finished = finished | (next_token == EOS_IDX)
                if finished.all():
                    break
                
                input_token = next_token
        
        if return_attention:
            return translations, attention_weights
        return translations

    def beam_search(self,
                    src: torch.Tensor,
                    src_lens: Optional[torch.Tensor] = None,
                    beam_size: int = 5,
                    max_len: int = 100) -> torch.Tensor:
        """
        束搜索解码
        
        Args:
            src: [batch, src_len] 源序列（目前仅支持 batch_size=1）
            src_lens: [batch] 源序列长度
            beam_size: 束大小
            max_len: 最大生成长度
            
        Returns:
            [1, seq_len] 最佳翻译结果
        """
        self.eval()
        device = self.device
        
        # 目前仅支持单个样本
        assert src.size(0) == 1, "Beam search currently only supports batch_size=1"
        
        with torch.no_grad():
            # 编码
            encoder_outputs, hidden = self.encoder(src, src_lens)
            
            # 创建源序列掩码
            src_mask = (src == PAD_IDX)
            
            # 扩展 encoder_outputs 和 hidden 为 beam_size
            encoder_outputs = encoder_outputs.repeat(beam_size, 1, 1)  # [beam, src_len, hidden]
            src_mask = src_mask.repeat(beam_size, 1)  # [beam, src_len]
            
            # 扩展 hidden state
            if self.encoder.rnn_type == 'lstm':
                hidden = (
                    hidden[0].repeat(1, beam_size, 1),  # [n_layers, beam, hidden]
                    hidden[1].repeat(1, beam_size, 1)
                )
            else:
                hidden = hidden.repeat(1, beam_size, 1)
            
            # 初始化 beam
            # beams: [beam_size, seq_len] 存储生成的序列
            beams = torch.full((beam_size, 1), SOS_IDX, dtype=torch.long, device=device)
            beam_scores = torch.zeros(beam_size, device=device)  # 累积 log 概率
            
            # 完成的序列
            finished_beams = []
            finished_scores = []
            
            # 当前输入 token
            input_token = torch.full((beam_size,), SOS_IDX, dtype=torch.long, device=device)
            
            for step in range(max_len - 1):
                # 解码一步
                output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, src_mask)
                
                # 计算 log 概率
                log_probs = torch.log_softmax(output, dim=-1)  # [beam_size, vocab_size]
                vocab_size = log_probs.size(-1)
                
                # 计算新的分数
                # next_scores: [beam_size, vocab_size]
                next_scores = beam_scores.unsqueeze(1) + log_probs
                
                if step == 0:
                    # 第一步只使用第一个 beam（因为都是从 SOS 开始）
                    next_scores = next_scores[0].unsqueeze(0)
                
                # 展平并选择 top-k
                next_scores_flat = next_scores.view(-1)  # [beam_size * vocab_size] 或 [vocab_size]
                top_scores, top_indices = next_scores_flat.topk(beam_size, dim=0)
                
                # 恢复 beam 和 token 索引
                if step == 0:
                    beam_indices = torch.zeros(beam_size, dtype=torch.long, device=device)
                    token_indices = top_indices
                else:
                    beam_indices = top_indices // vocab_size
                    token_indices = top_indices % vocab_size
                
                # 更新 beams
                new_beams = torch.cat([
                    beams[beam_indices],
                    token_indices.unsqueeze(1)
                ], dim=1)
                
                # 更新 hidden state
                if self.encoder.rnn_type == 'lstm':
                    hidden = (
                        hidden[0][:, beam_indices, :],
                        hidden[1][:, beam_indices, :]
                    )
                else:
                    hidden = hidden[:, beam_indices, :]
                
                # 更新 encoder_outputs 和 src_mask（根据 beam_indices 重排）
                encoder_outputs = encoder_outputs[beam_indices]
                src_mask = src_mask[beam_indices]
                
                # 检查是否有完成的 beam
                finished_mask = token_indices == EOS_IDX
                for i in range(beam_size):
                    if finished_mask[i]:
                        finished_beams.append(new_beams[i].clone())
                        finished_scores.append(top_scores[i].item())
                
                # 过滤未完成的 beams
                unfinished_mask = ~finished_mask
                if unfinished_mask.sum() == 0:
                    break
                
                # 如果有 beam 完成了，需要调整
                if finished_mask.any():
                    # 重新选择未完成的 beams
                    remaining_indices = unfinished_mask.nonzero(as_tuple=True)[0]
                    new_beams = new_beams[remaining_indices]
                    top_scores = top_scores[remaining_indices]
                    
                    if self.encoder.rnn_type == 'lstm':
                        hidden = (
                            hidden[0][:, remaining_indices, :],
                            hidden[1][:, remaining_indices, :]
                        )
                    else:
                        hidden = hidden[:, remaining_indices, :]
                    
                    encoder_outputs = encoder_outputs[remaining_indices]
                    src_mask = src_mask[remaining_indices]
                    
                    current_beam_size = new_beams.size(0)
                    if current_beam_size == 0:
                        break
                else:
                    current_beam_size = beam_size
                
                beams = new_beams
                beam_scores = top_scores
                input_token = token_indices[unfinished_mask] if finished_mask.any() else token_indices
            
            # 如果没有完成的序列，使用当前最佳
            if not finished_beams:
                finished_beams = [beams[0]]
                finished_scores = [beam_scores[0].item()]
            
            # 选择最佳结果（归一化分数）
            # 使用长度归一化
            normalized_scores = [score / len(beam) for score, beam in zip(finished_scores, finished_beams)]
            best_idx = normalized_scores.index(max(normalized_scores))
            best_beam = finished_beams[best_idx]
            
            return best_beam.unsqueeze(0)


def create_rnn_model(src_vocab_size: int,
                     tgt_vocab_size: int,
                     embed_dim: int = 256,
                     hidden_dim: int = 256,
                     n_layers: int = 2,
                     dropout: float = 0.3,
                     attn_type: str = 'dot',
                     rnn_type: str = 'lstm',
                     device: torch.device = None) -> Seq2Seq:
    """
    创建RNN翻译模型
    
    Args:
        src_vocab_size: 源语言词表大小
        tgt_vocab_size: 目标语言词表大小
        embed_dim: 词嵌入维度
        hidden_dim: 隐藏层维度
        n_layers: RNN层数
        dropout: Dropout比率
        attn_type: 注意力类型
        rnn_type: RNN类型
        device: 设备
        
    Returns:
        Seq2Seq模型
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder(
        vocab_size=src_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        rnn_type=rnn_type
    )
    
    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        attn_type=attn_type,
        rnn_type=rnn_type
    )
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    return model


def count_parameters(model: nn.Module) -> int:
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试代码
    print("Testing RNN Seq2Seq model...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 模拟参数
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    batch_size = 4
    src_len = 20
    tgt_len = 25
    
    # 创建模型
    model = create_rnn_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_dim=256,
        hidden_dim=256,
        n_layers=2,
        dropout=0.3,
        attn_type='dot',
        rnn_type='lstm',
        device=device
    )
    
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # 创建随机输入
    src = torch.randint(4, src_vocab_size, (batch_size, src_len)).to(device)
    tgt = torch.randint(4, tgt_vocab_size, (batch_size, tgt_len)).to(device)
    src[:, 0] = SOS_IDX
    src[:, -1] = EOS_IDX
    tgt[:, 0] = SOS_IDX
    tgt[:, -1] = EOS_IDX
    
    src_lens = torch.tensor([src_len] * batch_size)
    
    # 测试forward
    print("\nTesting forward pass...")
    model.train()
    outputs = model(src, tgt, src_lens)
    print(f"  Output shape: {outputs.shape}")  # [batch, tgt_len-1, vocab_size]
    assert outputs.shape == (batch_size, tgt_len - 1, tgt_vocab_size)
    
    # 测试translate
    print("\nTesting translation...")
    model.eval()
    translations, attention = model.translate(src, src_lens, max_len=30, return_attention=True)
    print(f"  Translation shape: {translations.shape}")  # [batch, max_len]
    print(f"  Attention shape: {attention.shape}")
    
    # 测试不同注意力类型
    print("\nTesting different attention types...")
    for attn_type in ['dot', 'additive', 'multiplicative']:
        model = create_rnn_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            attn_type=attn_type,
            device=device
        )
        model.train()
        outputs = model(src, tgt, src_lens)
        print(f"  {attn_type}: output shape = {outputs.shape}")
    
    print("\n✓ All tests passed!")

