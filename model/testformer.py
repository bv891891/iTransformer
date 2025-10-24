import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class RevIN(nn.Module):
    def __init__(self, num_features, affine=True, eps=1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        self._cached_mean = None
        self._cached_std = None

    def forward(self, x, mode: str):
        if mode == 'norm':
            mean = x.mean(1, keepdim=True).detach()
            std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False).detach() + self.eps)
            self._cached_mean = mean
            self._cached_std = std
            x = (x - mean) / std
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        if mode == 'denorm':
            if self._cached_mean is None or self._cached_std is None:
                raise RuntimeError('RevIN: statistics not available for denormalization.')
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            return x * self._cached_std + self._cached_mean
        raise ValueError(f"RevIN received unsupported mode: {mode}")


class ChannelDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if (not self.training) or self.p == 0.0:
            return x
        keep_prob = 1 - self.p
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(keep_prob)
        return x * mask / keep_prob


class Model(nn.Module):
    """iTransformer variant enhanced with RevIN and channel dropout."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.use_revin = getattr(configs, 'use_revin', True)

        self.revin = RevIN(configs.enc_in) if self.use_revin else None
        self.channel_dropout = ChannelDropout(configs.dropout)

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.class_strategy = configs.class_strategy
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.revin is not None:
            x_enc = self.revin(x_enc, 'norm')
            stats = None
        elif self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
            stats = (means, stdev)
        else:
            stats = (None, None)

        x_enc = self.channel_dropout(x_enc)

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.revin is not None:
            dec_out = self.revin(dec_out, 'denorm')
        elif self.use_norm:
            means, stdev = stats
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
