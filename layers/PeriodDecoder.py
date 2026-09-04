# LightGTS (ICML 2025) period-parallel decoding components, ported verbatim for SparseTimeLLM.
#
# Provenance (reference/LightGTS @ 36ad5bf):
#   - Transpose, sinusoidal_position_embedding, RoPE, RoPE_decoder,
#     ScaledDotProductAttention, MultiheadAttention
#       <- src/models/layers/basics.py:7-13, src/models/layers/attention.py
#   - CMlp, DecoderLayer, Decoder, causal_attention_mask
#       <- src/models/layers/decoder_cnn.py
#   - resize, decoder_PredictHead
#       <- src/models/LightGTS_pretrain_period.py:204-222, 413-420
#
# Only behavior-preserving cleanups were applied:
#   - decoder_PredictHead.forward: removed the local `Linear` whose resampled
#     weights were never used (dead code in the source; the effective path is
#     self.linear -> resize interpolation).

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import numpy as np


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device, factor=1.0):
    # (max_len * factor, 1)
    position = torch.arange(0, max_len * factor, 1 / factor, dtype=torch.float).unsqueeze(-1)
    # (output_dim//2)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  # i 范围是 [0, d/2]
    theta = torch.pow(10000, -2 * ids / output_dim)

    # (max_len * factor, output_dim//2)
    embeddings = position * theta

    # (max_len * factor, output_dim//2, 2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    # (bs, head, max_len * factor, output_dim//2, 2)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))

    # (bs, head, max_len * factor, output_dim)
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, -1, output_dim))
    embeddings = embeddings.to(device)

    # 如果 factor > 1, 使用插值位置来生成更细粒度的嵌入
    if factor > 1.0:
        interpolation_indices = torch.linspace(0, embeddings.shape[2] - 1, max_len).long()
        embeddings = embeddings[:, :, interpolation_indices, :]

    return embeddings


def RoPE(q, k):
    # q,k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device, factor=1)

    # cos_pos,sin_pos: (bs, head, max_len, output_dim)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了

    # 更新qw, *对应位置相乘
    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos + k2 * sin_pos

    return q, k


def RoPE_decoder(q, k):
    # q,k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    q_max_len = q.shape[2]
    k_max_len = k.shape[2]
    output_dim = q.shape[-1]

    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, k_max_len + q_max_len, output_dim, q.device, factor=1)

    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)

    # 更新qw, *对应位置相乘
    q = q * cos_pos[:,:,-q_max_len:,:] + q2 * sin_pos[:,:,-q_max_len:,:]

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos[:,:,:k_max_len,:] + k2 * sin_pos[:,:,:k_max_len,:]
    return q, k


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False, rope_type=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        self.rope_type = rope_type

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
        '''
        # using RoPE
        if self.rope_type:
            q, k = RoPE_decoder(q, k.permute(0, 1, 3, 2))
        else:
            q, k = RoPE(q, k.permute(0, 1, 3, 2))
        k = k.permute(0, 1, 3, 2)

        # Scaled MatMul (q, k)
        attn_scores = torch.matmul(q, k) * self.scale      # [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)      # [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)             # [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False, rope_type=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa, rope_type=rope_type)

        # Project output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)       # [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)   # [bs x n_heads x d_k x q_len]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)       # [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)  # [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1)
        return x


def causal_attention_mask(seq_length):
    """
    创建一个因果注意力掩码。掩码中的每个位置 (i, j)
    表示在计算第i个位置的attention时, 第j个位置是否可以被看见。
    如果j <= i, 这个位置被设为1(可见), 否则设为0(不可见)。

    Args:
        seq_length (int): 序列的长度

    Returns:
        torch.Tensor: 因果注意力掩码，大小为 (seq_length, seq_length)
    """
    mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
    return mask


class DecoderLayer(nn.Module):
    def __init__(self, patch_len, d_model, n_heads, d_ff=None, attn_dropout=0.2, dropout=0.5, norm="BatchNorm"):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model, n_heads, res_attention=False, attn_dropout=attn_dropout)
        self.cross_attention = MultiheadAttention(d_model, n_heads, attn_dropout=attn_dropout, rope_type=True)

        if 'batch' in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm3 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.MLP1 = CMlp(in_features=d_model, hidden_features=d_ff, out_features=d_model, drop=dropout)

    def forward(self, x, cross):
        batch, n_vars, num_patch, d_model = x.shape
        x = x.reshape(batch * n_vars, num_patch, d_model)

        cross = cross.reshape(batch * n_vars, -1, d_model)

        attention_mask = causal_attention_mask(num_patch).to(x.device)
        x_attn, _ = self.self_attention(x, attn_mask=attention_mask)
        x_attn = self.norm1(x_attn) + x

        x_cross, _ = self.cross_attention(x_attn, cross, cross)
        x_cross = self.dropout(self.norm2(x_cross)) + x_attn

        x_ff = self.MLP1(x_cross)
        x_ff = self.norm3(x_ff) + x_cross

        x_ff = x_ff.reshape(batch, n_vars, num_patch, d_model)

        return x_ff


class Decoder(nn.Module):
    def __init__(self, d_layers, patch_len, d_model, n_heads, d_ff=None, attn_dropout=0.2, dropout=0.1, norm="BatchNorm"):
        super(Decoder, self).__init__()

        self.decoder_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decoder_layers.append(DecoderLayer(patch_len, d_model, n_heads, d_ff, attn_dropout, dropout, norm))

    def forward(self, x, cross):
        output = x
        for layer in self.decoder_layers:
            output = layer(output, cross)
        return output


def resize(x, target_patch_len):
    '''
    x: tensor [bs x num_patch x n_vars x patch_len]]
    '''
    bs, num_patch, n_vars, patch_len = x.shape
    x = x.reshape(bs * num_patch, n_vars, patch_len)
    x = F.interpolate(x, size=target_patch_len, mode='linear', align_corners=False)
    return x.reshape(bs, num_patch, n_vars, target_patch_len)


class decoder_PredictHead(nn.Module):
    def __init__(self, d_model, target_patch_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, target_patch_len)

    def forward(self, x, patch_len):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x num_patch x patch_len x nvars] -> [bs, num_patch*patch_len, nvars]
        """
        x = x.transpose(2, 3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )        # [bs x nvars x num_patch x target_patch_len]
        x = resize(x, target_patch_len=patch_len)  # [bs x nvars x num_patch x patch_len]
        x = x.permute(0, 2, 3, 1)                 # [bs x num_patch x patch_len x nvars]
        return x.reshape(x.shape[0], -1, x.shape[3])
