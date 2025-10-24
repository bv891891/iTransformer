import copy
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import FullAttention


class RevIN(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=False, keepdim=True) + self.eps)
        x_norm = (x - mean) / std
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm, mean, std

    def inverse(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        if self.affine:
            x = (x - self.beta) / (self.gamma + self.eps)
        return x * std + mean


class MovingAverageTrend(nn.Module):
    def __init__(self, seq_len: int, window_candidates: Optional[Iterable[int]] = None):
        super().__init__()
        if window_candidates is None:
            window_candidates = (4, 8, 12, 24)
        self.window_candidates = tuple(sorted(set(int(w) for w in window_candidates if w > 1)))
        if len(self.window_candidates) == 0:
            raise ValueError("window_candidates must contain at least one integer greater than 1")
        self.predictor = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, len(self.window_candidates))
        )

    def _moving_average(self, series: torch.Tensor, window: int):
        B, L, N = series.shape
        reshaped = series.permute(0, 2, 1).reshape(B * N, 1, L)
        padded = F.pad(reshaped, (window - 1, 0), mode='replicate')
        trend = F.avg_pool1d(padded, kernel_size=window, stride=1)
        return trend.view(B, N, L).permute(0, 2, 1)

    def forward(self, series: torch.Tensor):
        B, L, N = series.shape
        trends = [self._moving_average(series, w) for w in self.window_candidates]
        trend_stack = torch.stack(trends, dim=-1)  # [B, L, N, W]

        mean_feat = series.mean(dim=1)
        std_feat = torch.sqrt(torch.var(series, dim=1, unbiased=False) + 1e-6)
        stats = torch.stack([mean_feat, std_feat], dim=-1)  # [B, N, 2]
        weights = self.predictor(stats).softmax(dim=-1)  # [B, N, W]
        trend = (trend_stack * weights.unsqueeze(1)).sum(dim=-1)  # [B, L, N]

        trend_flat = trend.permute(0, 2, 1)  # [B, N, L]
        norm_trend = F.normalize(trend_flat, p=2, dim=-1)
        guidance = torch.matmul(norm_trend, norm_trend.transpose(-1, -2))  # [B, N, N]
        reg = (guidance.mean(dim=(-1, -2)) - 0.5) ** 2
        reg = reg.mean()
        return trend, guidance, reg


class VTGAFullAttention(nn.Module):
    def __init__(self, attention: FullAttention):
        super().__init__()
        self.attention = attention

    def forward(self, queries, keys, values, attn_mask=None, guidance=None, tau=None, delta=None):
        context, attn = self.attention(queries, keys, values, attn_mask=attn_mask, tau=tau, delta=delta)
        if attn is None or guidance is None:
            return context, attn
        guide = torch.sigmoid(guidance)
        if guide.size(-2) != attn.size(-2) or guide.size(-1) != attn.size(-1):
            B = attn.size(0)
            target_q, target_k = attn.size(-2), attn.size(-1)
            padded = torch.full((B, target_q, target_k), 0.5, device=guide.device, dtype=guide.dtype)
            q = min(target_q, guide.size(-2))
            k = min(target_k, guide.size(-1))
            padded[:, :q, :k] = guide[:, :q, :k]
            guide = padded
        guide = guide.unsqueeze(1)
        attn = attn * guide
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        attn = self.attention.dropout(attn)
        context = torch.einsum("bhls,bshd->blhd", attn, values)
        return context.contiguous(), attn


class VTGAAttentionLayer(nn.Module):
    def __init__(self, attention: FullAttention, d_model: int, n_heads: int, d_keys: Optional[int] = None,
                 d_values: Optional[int] = None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = VTGAFullAttention(attention)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, guidance=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        context, attn = self.inner_attention(queries, keys, values, attn_mask=attn_mask, guidance=guidance,
                                             tau=tau, delta=delta)
        out = context.view(B, L, -1)
        return self.out_projection(out), attn


class VTGAEncoderLayer(nn.Module):
    def __init__(self, attention_layer: VTGAAttentionLayer, d_model: int, d_ff: Optional[int] = None,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention_layer
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, guidance=None, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, guidance=guidance, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class VTGAEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, guidance=None, attn_mask=None, tau=None, delta=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, guidance=guidance, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.revin = RevIN(configs.enc_in)
        self.embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                configs.dropout)
        attention = FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=True)
        attn_layers = [
            VTGAEncoderLayer(
                VTGAAttentionLayer(copy.deepcopy(attention), configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            )
            for _ in range(configs.e_layers)
        ]
        self.encoder = VTGAEncoder(attn_layers, norm_layer=nn.LayerNorm(configs.d_model))
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.trend = MovingAverageTrend(configs.seq_len)
        self.guide_loss = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_norm, mean, std = self.revin(x_enc)
        _, guidance, reg = self.trend(x_norm)
        self.guide_loss = reg
        enc_out = self.embedding(x_norm, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, guidance=guidance)
        dec_out = self.projector(enc_out).permute(0, 2, 1)
        dec_out = dec_out[:, :, :x_enc.size(-1)]
        dec_out = self.revin.inverse(dec_out, mean, std)
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
