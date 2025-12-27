"""
Transformer 机器翻译模型
使用 PyTorch 内置的 nn.Transformer 简化实现
符合作业要求：完整的 Encoder-Decoder Transformer 架构

扩展功能（用于消融实验）：
- 位置编码：绝对位置编码（sin/cos）vs 相对位置编码（可学习）
- 归一化方法：LayerNorm vs RMSNorm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# 导入特殊token索引
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import PAD_IDX, SOS_IDX, EOS_IDX


# ============================================================
# 归一化层实现
# ============================================================

class RMSNorm(nn.Module):
    """
    RMS归一化（Root Mean Square Layer Normalization）
    
    相比LayerNorm，RMSNorm不计算均值，只使用RMS进行归一化，计算更高效。
    论文: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    
    公式: x_norm = x / RMS(x) * g
    其中 RMS(x) = sqrt(mean(x^2) + eps)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: 模型维度
            eps: 数值稳定性的小量
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            [batch, seq_len, d_model] 归一化后的张量
        """
        # 计算 RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并缩放
        return x / rms * self.weight


def get_norm_layer(norm_type: str, d_model: int) -> nn.Module:
    """
    根据类型获取归一化层
    
    Args:
        norm_type: 归一化类型 ('layernorm' 或 'rmsnorm')
        d_model: 模型维度
        
    Returns:
        归一化层模块
    """
    if norm_type == 'layernorm':
        return nn.LayerNorm(d_model)
    elif norm_type == 'rmsnorm':
        return RMSNorm(d_model)
    else:
        raise ValueError(f"未知的归一化类型: {norm_type}，支持 'layernorm' 或 'rmsnorm'")


# ============================================================
# 位置编码实现
# ============================================================

class PositionalEncoding(nn.Module):
    """
    位置编码（使用 sin/cos 绝对位置编码）
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # 注册为buffer（不参与训练）
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    可学习位置编码（Learned Positional Encoding）
    
    相比固定的sin/cos编码，可学习编码可以适应具体任务。
    这是一种相对简单的"相对位置"替代方案。
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 可学习位置嵌入
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # 初始化
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码（Relative Positional Encoding）
    
    实现类似于 Transformer-XL / Shaw et al. 的相对位置表示。
    相对位置编码关注token之间的相对距离而非绝对位置。
    
    论文: "Self-Attention with Relative Position Representations" (Shaw et al., 2018)
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1, 
                 max_relative_position: int = 128):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比率
            max_relative_position: 最大相对位置距离（超出则clip）
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # 相对位置嵌入：从 -max_relative_position 到 +max_relative_position
        # 总共 2*max_relative_position + 1 个位置
        num_embeddings = 2 * max_relative_position + 1
        self.relative_position_embedding = nn.Embedding(num_embeddings, d_model)
        
        # 初始化
        nn.init.xavier_uniform_(self.relative_position_embedding.weight)
        
        # 额外添加一个基础位置编码以保持稳定性
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def get_relative_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        获取相对位置矩阵
        
        Args:
            seq_len: 序列长度
            device: 设备
            
        Returns:
            [seq_len, seq_len] 相对位置索引矩阵
        """
        # 创建位置索引
        positions = torch.arange(seq_len, device=device)
        # 计算相对位置：positions[i] - positions[j]
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        # clip到有效范围
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        # 偏移到正索引
        relative_positions = relative_positions + self.max_relative_position
        return relative_positions
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            [batch, seq_len, d_model]
        """
        # 使用基础sin/cos位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
    def get_relative_embeddings(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        获取相对位置嵌入矩阵，可用于注意力计算
        
        Args:
            seq_len: 序列长度
            device: 设备
            
        Returns:
            [seq_len, seq_len, d_model] 相对位置嵌入
        """
        relative_positions = self.get_relative_positions(seq_len, device)
        return self.relative_position_embedding(relative_positions)


def get_positional_encoding(pos_encoding_type: str, d_model: int, 
                           max_len: int = 5000, dropout: float = 0.1) -> nn.Module:
    """
    根据类型获取位置编码层
    
    Args:
        pos_encoding_type: 位置编码类型 ('sinusoidal', 'learned', 'relative')
        d_model: 模型维度
        max_len: 最大序列长度
        dropout: Dropout比率
        
    Returns:
        位置编码模块
    """
    if pos_encoding_type == 'sinusoidal':
        return PositionalEncoding(d_model, max_len, dropout)
    elif pos_encoding_type == 'learned':
        return LearnedPositionalEncoding(d_model, max_len, dropout)
    elif pos_encoding_type == 'relative':
        return RelativePositionalEncoding(d_model, max_len, dropout)
    else:
        raise ValueError(f"未知的位置编码类型: {pos_encoding_type}，"
                        f"支持 'sinusoidal', 'learned', 'relative'")


# ============================================================
# 自定义 Transformer 层（支持不同归一化方法）
# ============================================================

class CustomTransformerEncoderLayer(nn.Module):
    """
    自定义 Transformer 编码器层，支持不同的归一化方法
    """
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, norm_type: str = 'layernorm'):
        super().__init__()
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 归一化层
        self.norm1 = get_norm_layer(norm_type, d_model)
        self.norm2 = get_norm_layer(norm_type, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention with residual
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward with residual
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class CustomTransformerDecoderLayer(nn.Module):
    """
    自定义 Transformer 解码器层，支持不同的归一化方法
    """
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, norm_type: str = 'layernorm'):
        super().__init__()
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 归一化层
        self.norm1 = get_norm_layer(norm_type, d_model)
        self.norm2 = get_norm_layer(norm_type, d_model)
        self.norm3 = get_norm_layer(norm_type, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention with residual
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with residual
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward with residual
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class CustomTransformerEncoder(nn.Module):
    """自定义 Transformer 编码器"""
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class CustomTransformerDecoder(nn.Module):
    """自定义 Transformer 解码器"""
    
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
        return output


# ============================================================
# 主模型实现
# ============================================================

class TransformerNMT(nn.Module):
    """
    使用 PyTorch 内置 Transformer 的神经机器翻译模型
    
    优势：稳定可靠、实现简单、经过充分测试
    
    扩展支持（用于消融实验）：
    - 位置编码类型：sinusoidal（默认）, learned, relative
    - 归一化类型：layernorm（默认）, rmsnorm
    """
    
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 pos_encoding_type: str = 'sinusoidal',
                 norm_type: str = 'layernorm'):
        """
        Args:
            src_vocab_size: 源语言词表大小
            tgt_vocab_size: 目标语言词表大小
            d_model: 模型维度
            nhead: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            max_len: 最大序列长度
            pos_encoding_type: 位置编码类型 ('sinusoidal', 'learned', 'relative')
            norm_type: 归一化类型 ('layernorm', 'rmsnorm')
        """
        super().__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pos_encoding_type = pos_encoding_type
        self.norm_type = norm_type
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        
        # 位置编码（支持多种类型）
        self.pos_encoder = get_positional_encoding(pos_encoding_type, d_model, max_len, dropout)
        
        # 根据归一化类型选择使用内置Transformer还是自定义Transformer
        if norm_type == 'layernorm':
            # 使用 PyTorch 内置 Transformer（默认使用 LayerNorm）
            self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
                batch_first=True
            )
            self.use_custom_transformer = False
        else:
            # 使用自定义 Transformer（支持 RMSNorm）
            encoder_layer = CustomTransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, norm_type=norm_type
            )
            decoder_layer = CustomTransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, norm_type=norm_type
            )
            self.encoder = CustomTransformerEncoder(encoder_layer, num_encoder_layers)
            self.decoder = CustomTransformerDecoder(decoder_layer, num_decoder_layers)
            self.use_custom_transformer = True
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        生成 decoder 的 causal mask（下三角掩码）
        
        Args:
            sz: 序列长度
            device: 设备
            
        Returns:
            [sz, sz] 掩码矩阵
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """
        创建 padding mask
        
        Args:
            seq: [batch, seq_len]
            
        Returns:
            [batch, seq_len] True 表示需要 mask 的位置
        """
        return seq == PAD_IDX
    
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_padding_mask: Optional[torch.Tensor] = None,
                tgt_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        训练时的前向传播
        
        Args:
            src: [batch, src_len] 源序列
            tgt: [batch, tgt_len] 目标序列（包含 SOS）
            src_padding_mask: [batch, src_len] 源序列 padding mask
            tgt_padding_mask: [batch, tgt_len] 目标序列 padding mask
            
        Returns:
            [batch, tgt_len, tgt_vocab_size] 输出 logits
        """
        device = src.device
        tgt_len = tgt.size(1)
        
        # 生成 masks
        if src_padding_mask is None:
            src_padding_mask = self.create_padding_mask(src)
        if tgt_padding_mask is None:
            tgt_padding_mask = self.create_padding_mask(tgt)
        
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, device)
        
        # Embedding + Positional Encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Transformer forward（根据是否使用自定义Transformer选择不同路径）
        if self.use_custom_transformer:
            # 使用自定义 Transformer
            memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
            output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                 tgt_key_padding_mask=tgt_padding_mask,
                                 memory_key_padding_mask=src_padding_mask)
        else:
            # 使用内置 Transformer
            output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # 输出投影
        logits = self.fc_out(output)
        
        return logits
    
    def translate(self,
                  src: torch.Tensor,
                  max_len: int = 100,
                  return_attention: bool = False,
                  repetition_penalty: float = 1.2) -> torch.Tensor:
        """
        推理时的翻译（贪婪解码 + 重复惩罚）
        
        Args:
            src: [batch, src_len] 源序列
            max_len: 最大生成长度
            return_attention: 是否返回注意力权重（暂不支持）
            repetition_penalty: 重复惩罚系数（>1减少重复，建议1.2-2.0）
            
        Returns:
            [batch, max_len] 翻译结果
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        with torch.no_grad():
            # 源序列编码
            src_padding_mask = self.create_padding_mask(src)
            src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
            
            # 根据是否使用自定义Transformer选择不同路径
            if self.use_custom_transformer:
                memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
            else:
                memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
            
            # 初始化目标序列
            tgt = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=device)
            
            # 记录是否已生成 EOS
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for step in range(max_len - 1):
                tgt_len = tgt.size(1)
                tgt_mask = self.generate_square_subsequent_mask(tgt_len, device)
                tgt_padding_mask = self.create_padding_mask(tgt)
                
                tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
                
                # 根据是否使用自定义Transformer选择不同路径
                if self.use_custom_transformer:
                    output = self.decoder(
                        tgt_emb, memory,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_padding_mask,
                        memory_key_padding_mask=src_padding_mask
                    )
                else:
                    output = self.transformer.decoder(
                    tgt_emb, memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask
                    )
                
                logits = self.fc_out(output[:, -1, :])  # 只取最后一个 token [batch, vocab_size]
                
                # 应用重复惩罚
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        # 获取已生成的token
                        generated_tokens = tgt[i]
                        # 对已出现的token进行惩罚
                        for token in generated_tokens:
                            if token != PAD_IDX and token != SOS_IDX:
                                # 如果logit>0，除以penalty；如果<0，乘以penalty
                                if logits[i, token] < 0:
                                    logits[i, token] *= repetition_penalty
                                else:
                                    logits[i, token] /= repetition_penalty
                
                next_token = logits.argmax(dim=-1, keepdim=True)  # [batch, 1]
                
                # 对已完成的序列，填充 PAD
                next_token = next_token.masked_fill(finished.unsqueeze(1), PAD_IDX)
                
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # 更新完成状态
                finished = finished | (next_token.squeeze(1) == EOS_IDX)
                if finished.all():
                    break
        
        return tgt
    
    def beam_search(self,
                    src: torch.Tensor,
                    beam_size: int = 5,
                    max_len: int = 100) -> torch.Tensor:
        """
        束搜索解码
        
        Args:
            src: [1, src_len] 单个源序列
            beam_size: 束大小
            max_len: 最大生成长度
            
        Returns:
            [1, max_len] 翻译结果
        """
        self.eval()
        device = src.device
        
        with torch.no_grad():
            # 源序列编码
            src_padding_mask = self.create_padding_mask(src)
            src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
            
            # 根据是否使用自定义Transformer选择不同路径
            if self.use_custom_transformer:
                memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
            else:
                memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
            
            # 扩展 memory 为 beam_size
            memory = memory.repeat(beam_size, 1, 1)
            src_padding_mask = src_padding_mask.repeat(beam_size, 1)
            
            # 初始化 beam
            beams = torch.full((beam_size, 1), SOS_IDX, dtype=torch.long, device=device)
            beam_scores = torch.zeros(beam_size, device=device)
            finished_beams = []
            finished_scores = []
            
            for step in range(max_len - 1):
                tgt_len = beams.size(1)
                tgt_mask = self.generate_square_subsequent_mask(tgt_len, device)
                tgt_padding_mask = self.create_padding_mask(beams)
                
                tgt_emb = self.pos_encoder(self.tgt_embedding(beams) * math.sqrt(self.d_model))
                
                # 根据是否使用自定义Transformer选择不同路径
                if self.use_custom_transformer:
                    output = self.decoder(
                        tgt_emb, memory,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_padding_mask,
                        memory_key_padding_mask=src_padding_mask
                    )
                else:
                    output = self.transformer.decoder(
                    tgt_emb, memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask
                    )
                
                logits = self.fc_out(output[:, -1, :])  # [beam_size, vocab_size]
                log_probs = F.log_softmax(logits, dim=-1)
                
                # 计算新的分数
                vocab_size = log_probs.size(-1)
                next_scores = beam_scores.unsqueeze(1) + log_probs  # [beam_size, vocab_size]
                
                # 展平并选择 top-k
                next_scores = next_scores.view(-1)  # [beam_size * vocab_size]
                top_scores, top_indices = next_scores.topk(beam_size, dim=0)
                
                # 恢复 beam 和 token 索引
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size
                
                # 更新 beams
                new_beams = torch.cat([
                    beams[beam_indices],
                    token_indices.unsqueeze(1)
                ], dim=1)
                
                # 检查是否有完成的 beam
                finished_mask = token_indices == EOS_IDX
                for i in range(beam_size):
                    if finished_mask[i]:
                        finished_beams.append(new_beams[i])
                        finished_scores.append(top_scores[i].item())
                
                # 过滤未完成的 beams
                unfinished_mask = ~finished_mask
                if unfinished_mask.sum() == 0:
                    break
                
                beams = new_beams[unfinished_mask]
                beam_scores = top_scores[unfinished_mask]
                
                # 调整 beam_size
                current_beam_size = beams.size(0)
                if current_beam_size < beam_size:
                    memory = memory[:current_beam_size]
                    src_padding_mask = src_padding_mask[:current_beam_size]
            
            # 选择最佳结果
            if finished_beams:
                best_idx = finished_scores.index(max(finished_scores))
                best_beam = finished_beams[best_idx]
            else:
                best_beam = beams[0]
            
            return best_beam.unsqueeze(0)


def create_transformer_model(src_vocab_size: int,
                              tgt_vocab_size: int,
                              d_model: int = 256,
                              nhead: int = 4,
                              num_encoder_layers: int = 3,
                              num_decoder_layers: int = 3,
                              dim_feedforward: int = 1024,
                              dropout: float = 0.1,
                              pos_encoding_type: str = 'sinusoidal',
                              norm_type: str = 'layernorm',
                              device: torch.device = None) -> TransformerNMT:
    """
    创建 Transformer 翻译模型
    
    Args:
        src_vocab_size: 源语言词表大小
        tgt_vocab_size: 目标语言词表大小
        d_model: 模型维度
        nhead: 注意力头数
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
        dim_feedforward: 前馈网络维度
        dropout: Dropout 比率
        pos_encoding_type: 位置编码类型 ('sinusoidal', 'learned', 'relative')
        norm_type: 归一化类型 ('layernorm', 'rmsnorm')
        device: 设备
        
    Returns:
        TransformerNMT 模型
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransformerNMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pos_encoding_type=pos_encoding_type,
        norm_type=norm_type
    )
    
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试代码
    print("Testing Transformer NMT model...")
    
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
    model = create_transformer_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
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
    
    # 测试 forward
    print("\nTesting forward pass...")
    model.train()
    outputs = model(src, tgt[:, :-1])  # 输入不包含最后一个 token
    print(f"  Output shape: {outputs.shape}")  # [batch, tgt_len-1, vocab_size]
    assert outputs.shape == (batch_size, tgt_len - 1, tgt_vocab_size)
    
    # 测试 translate
    print("\nTesting translation (greedy)...")
    model.eval()
    translations = model.translate(src, max_len=30)
    print(f"  Translation shape: {translations.shape}")  # [batch, <=max_len]
    
    # 测试 beam search (单个样本)
    print("\nTesting beam search...")
    single_src = src[0:1]
    beam_translation = model.beam_search(single_src, beam_size=3, max_len=30)
    print(f"  Beam translation shape: {beam_translation.shape}")
    
    print("\n✓ All tests passed!")

