import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_token_len = configs.input_token_len
        self.embedding = nn.Linear(self.input_token_len, configs.d_model, bias=False)
        self.position_embedding = PositionalEmbedding(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head = nn.Linear(configs.d_model, configs.output_token_len)
        self.use_norm = configs.use_norm

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        # [B, L, C]
        B, _, C = x_enc.shape
        # [B, C, L]
        x_enc = x_enc.permute(0, 2, 1)
        # [B, C, N, P]
        x_enc = x_enc.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        N = x_enc.shape[2]
        # [B * C, N, P]
        x_enc = x_enc.reshape(B * C, N, -1)
        # [B * C, N, D]
        enc_out = self.embedding(x_enc) + self.position_embedding(x_enc)
        enc_out = self.dropout(enc_out)
        enc_out, attns = self.encoder(enc_out)
        # [B * C, N, P]
        dec_out = self.head(enc_out)
        # [B, C, L]
        dec_out = dec_out.reshape(B, C, -1)
        # [B, L, C]
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_norm:
            dec_out = dec_out * stdev + means
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
