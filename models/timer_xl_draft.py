import torch
from torch import nn
from layers.Transformer_EncDec import TimerBlock, TimerLayer
from layers.SelfAttention_Family import AttentionLayer, TimeAttention


class Model(nn.Module):
    """
    Draft Timer-XL: downscaled variant for speculative decoding.

    Mirrors `models.timer_xl.Model` API but uses reduced dimensions
    (e.g., d_model, n_heads, d_ff, e_layers) for faster inference.
    """
    def __init__(self, configs):
        super().__init__()

        # Original config values (target model hyperparameters)
        original_d_model = int(getattr(configs, 'd_model', 512))
        original_n_heads = int(getattr(configs, 'n_heads', 8))
        original_d_ff = int(getattr(configs, 'd_ff', 2048))
        original_e_layers = int(getattr(configs, 'e_layers', 1))

        # Scaling factors or absolute overrides (optional; default halve)
        scale_d_model = float(getattr(configs, 'draft_scale_d_model', 0.5))
        scale_n_heads = float(getattr(configs, 'draft_scale_n_heads', 0.5))
        scale_d_ff = float(getattr(configs, 'draft_scale_d_ff', 0.5))
        scale_e_layers = float(getattr(configs, 'draft_scale_e_layers', 0.5))

        d_model = max(64, int(round(original_d_model * scale_d_model)))
        n_heads = max(1, int(round(original_n_heads * scale_n_heads)))
        d_ff = max(256, int(round(original_d_ff * scale_d_ff)))
        e_layers = max(1, int(round(original_e_layers * scale_e_layers)))

        # Ensure divisibility: d_model % n_heads == 0
        if d_model % n_heads != 0:
            # Try decreasing n_heads until it divides, but at least 1
            for candidate in range(n_heads, 0, -1):
                if d_model % candidate == 0:
                    n_heads = candidate
                    break
            # If still not divisible, increase d_model to nearest divisible value
            if d_model % n_heads != 0:
                d_model = (d_model // n_heads + 1) * n_heads

        self.input_token_len = configs.input_token_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Expose draft hyperparameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.e_layers = e_layers

        # Optional adapter to project embeddings/features to target dim
        target_d_model = getattr(configs, 'target_d_model', None)
        self.adapter_to_target = None
        if getattr(configs, 'adapter_to_target', False) and target_d_model is not None:
            self.adapter_to_target = nn.Linear(self.d_model, int(target_d_model))

        # Token (patch) embedding and output head keep token lengths identical
        self.embedding = nn.Linear(self.input_token_len, self.d_model)

        self.blocks = TimerBlock(
            [
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(
                            True,
                            attention_dropout=configs.dropout,
                            output_attention=self.output_attention,
                            d_model=self.d_model,
                            num_heads=self.n_heads,
                            covariate=configs.covariate,
                            flash_attention=configs.flash_attention,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
        )

        self.head = nn.Linear(self.d_model, configs.output_token_len)

    def forecast(self, x, x_mark, y_mark):
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x /= stdev

        B, _, C = x.shape
        # [B, C, L]
        x = x.permute(0, 2, 1)
        # [B, C, N, P]
        x = x.unfold(dimension=-1, size=self.input_token_len, step=self.input_token_len)
        N = x.shape[2]
        # [B, C, N, D]
        embed_out = self.embedding(x)
        # [B, C * N, D]
        embed_out = embed_out.reshape(B, C * N, -1)
        embed_out, attns = self.blocks(embed_out, n_vars=C, n_tokens=N)

        # Optional projection to target d_model if requested (not used by head)
        if self.adapter_to_target is not None:
            adapted = self.adapter_to_target(embed_out)
            # Expose for external consumers (e.g., speculative verification)
            self.last_adapted_features = adapted

        # [B, C * N, P]
        dec_out = self.head(embed_out)
        # [B, C, N * P]
        dec_out = dec_out.reshape(B, C, N, -1).reshape(B, C, -1)
        # [B, L, C]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev + means
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def forward(self, x, x_mark, y_mark):
        return self.forecast(x, x_mark, y_mark)

    # Convenience checkpoint helpers (optional usage)
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path, map_location=None):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state, strict=False)
        return self


