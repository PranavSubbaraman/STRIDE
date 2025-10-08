import torch
import time
from torch import nn
from layers.Transformer_EncDec import TimerBlock, TimerLayer
from layers.SelfAttention_Family import AttentionLayer, TimeAttention


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_token_len = configs.input_token_len
        self.embedding = nn.Linear(self.input_token_len, configs.d_model)
        self.output_attention = configs.output_attention
        self.blocks = TimerBlock(
            [
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(True, attention_dropout=configs.dropout,
                                    output_attention=self.output_attention, 
                                    d_model=configs.d_model, num_heads=configs.n_heads,
                                    covariate=configs.covariate, flash_attention=configs.flash_attention),
                                    configs.d_model, configs.n_heads),
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
        # timing controls
        self.enable_timing = bool(getattr(configs, 'trace_inference_breakdown', False))
        self.last_forward_breakdown = None
        # propagate toggle to layers
        if self.enable_timing and hasattr(self.blocks, 'attn_layers'):
            for lyr in self.blocks.attn_layers:
                if hasattr(lyr, 'enable_timing'):
                    lyr.enable_timing = True

    def forecast(self, x, x_mark, y_mark):
        timing_enabled = bool(getattr(self, 'enable_timing', False))
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter() if timing_enabled else None
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t1 = time.perf_counter() if timing_enabled else None
        B, _, C = x.shape
        # [B, C, L]
        x = x.permute(0, 2, 1)
        # [B, C, N, P]
        x = x.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t2 = time.perf_counter() if timing_enabled else None
        N = x.shape[2]
        # [B, C, N, D]
        embed_out = self.embedding(x)
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t3 = time.perf_counter() if timing_enabled else None
        # [B, C * N, D]
        embed_out = embed_out.reshape(B, C * N, -1)
        embed_out, attns = self.blocks(embed_out, n_vars=C, n_tokens=N)
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t4 = time.perf_counter() if timing_enabled else None
        # [B, C * N, P]
        dec_out = self.head(embed_out)
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t5 = time.perf_counter() if timing_enabled else None
        # [B, C, N * P]
        dec_out = dec_out.reshape(B, C, N, -1).reshape(B, C, -1)
        # [B, L, C]
        dec_out = dec_out.permute(0, 2, 1)
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t6 = time.perf_counter() if timing_enabled else None

        if self.use_norm:
            dec_out = dec_out * stdev + means
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t7 = time.perf_counter() if timing_enabled else None

        if timing_enabled:
            block_times = getattr(self.blocks, 'last_forward_timing', None)
            self.last_forward_breakdown = {
                'norm_in': (_t1 - _t0) if self.use_norm else 0.0,
                'patch_permute_unfold': (_t2 - _t1),
                'embedding': (_t3 - _t2),
                'blocks': (_t4 - _t3),
                'head': (_t5 - _t4),
                'reshape_permute_out': (_t6 - _t5),
                'denorm_out': (_t7 - _t6) if self.use_norm else 0.0,
            }
            if isinstance(block_times, dict):
                prefixed = {f"blocks.{k}": float(v) for k, v in block_times.items()}
                self.last_forward_breakdown.update(prefixed)
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def forward(self, x, x_mark, y_mark):
        return self.forecast(x, x_mark, y_mark)
