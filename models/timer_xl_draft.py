import torch
import time
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
        original_d_model = int(getattr(configs, 'd_model', 1024))
        original_n_heads = int(getattr(configs, 'n_heads', 8))
        original_d_ff = int(getattr(configs, 'd_ff', 2048))
        original_e_layers = int(getattr(configs, 'e_layers', 1))

        # Scaling factors or absolute overrides (optional)
        # If not provided, do NOT scale (use original values)
        scale_d_model = getattr(configs, 'draft_scale_d_model', None)
        scale_n_heads = getattr(configs, 'draft_scale_n_heads', None)
        scale_d_ff = getattr(configs, 'draft_scale_d_ff', None)
        scale_e_layers = getattr(configs, 'draft_scale_e_layers', None)

        # More robust scaling that ensures RoPE compatibility
        # RoPE requires head_dim to be divisible by 4 (since proj_width = head_dim // 2 must be even)
        # If neither d_model nor n_heads scales are provided, use originals directly (no scaling)
        if scale_d_model is None and scale_n_heads is None:
            d_model = int(original_d_model)
            n_heads = int(original_n_heads)
        else:
            # Use provided scales; missing ones default to 1.0 (no change)
            _sdm = 1.0 if scale_d_model is None else float(scale_d_model)
            _snh = 1.0 if scale_n_heads is None else float(scale_n_heads)
            target_d_model = int(round(original_d_model * _sdm))
            target_n_heads = int(round(original_n_heads * _snh))
            
            # Find the best compatible combination
            best_d_model, best_n_heads = None, None
            min_param_diff = float('inf')
            
            # Try different head counts around the target
            for heads in range(max(1, target_n_heads - 2), target_n_heads + 3):
                # Find d_model that's divisible by heads and gives head_dim divisible by 4
                base_head_dim = max(4, (target_d_model // heads // 4) * 4)  # Round down to multiple of 4 (min 4)
                candidate_d_model = base_head_dim * heads
                
                # Prefer staying close to target parameters
                param_diff = abs(candidate_d_model - target_d_model) + abs(heads - target_n_heads) * 20
                if param_diff < min_param_diff:
                    min_param_diff = param_diff
                    best_d_model, best_n_heads = candidate_d_model, heads
            
            d_model = max(16, best_d_model)
            n_heads = max(1, best_n_heads)
        
        # For d_ff and e_layers: if no scale provided, keep originals
        d_ff = int(original_d_ff) if scale_d_ff is None else max(64, int(round(original_d_ff * float(scale_d_ff))))
        e_layers = int(original_e_layers) if scale_e_layers is None else max(1, int(round(original_e_layers * float(scale_e_layers))))
        
        # print(f"Draft model: d_model={d_model}, n_heads={n_heads}, head_dim={d_model//n_heads}, d_ff={d_ff}, e_layers={e_layers}")

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

        # Stochastic draft: output both mean and log_variance for NLL training
        self.draft_stochastic = getattr(configs, 'draft_stochastic', False)
        if self.draft_stochastic:
            self.head_mean = nn.Linear(self.d_model, configs.output_token_len)
            self.head_logvar = nn.Linear(self.d_model, configs.output_token_len)
            # Initialize log_var head to output small values (high confidence initially)
            nn.init.constant_(self.head_logvar.bias, -2.0)  # log(sigma^2) = -2 => sigma ≈ 0.37
        else:
            self.head = nn.Linear(self.d_model, configs.output_token_len)
        # timing controls
        self.enable_timing = bool(getattr(configs, 'trace_inference_breakdown', False))
        self.last_forward_breakdown = None
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
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x /= stdev
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t1 = time.perf_counter() if timing_enabled else None

        B, _, C = x.shape
        # [B, C, L]
        x = x.permute(0, 2, 1)
        # [B, C, N, P]
        x = x.unfold(dimension=-1, size=self.input_token_len, step=self.input_token_len)
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

        # Optional projection to target d_model if requested (not used by head)
        if self.adapter_to_target is not None:
            adapted = self.adapter_to_target(embed_out)
            # Expose for external consumers (e.g., speculative verification)
            self.last_adapted_features = adapted
        _t4b = time.perf_counter() if timing_enabled else None

        # [B, C * N, P] - output mean (and optionally log_variance)
        if self.draft_stochastic:
            mean_out = self.head_mean(embed_out)
            logvar_out = self.head_logvar(embed_out)
            # Clamp log_variance for numerical stability
            logvar_out = torch.clamp(logvar_out, min=-10, max=10)
        else:
            dec_out = self.head(embed_out)
        
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t5 = time.perf_counter() if timing_enabled else None
        
        # [B, C, N * P] and [B, L, C]
        if self.draft_stochastic:
            mean_out = mean_out.reshape(B, C, N, -1).reshape(B, C, -1).permute(0, 2, 1)
            logvar_out = logvar_out.reshape(B, C, N, -1).reshape(B, C, -1).permute(0, 2, 1)
        else:
            dec_out = dec_out.reshape(B, C, N, -1).reshape(B, C, -1).permute(0, 2, 1)
        
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t6 = time.perf_counter() if timing_enabled else None

        if self.draft_stochastic:
            if self.use_norm:
                mean_out = mean_out * stdev + means
                # Variance scales with stdev^2
                logvar_out = logvar_out + 2 * torch.log(stdev)
        else:
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
                'adapter_to_target': max(0.0, (_t4b - _t4)) if self.adapter_to_target is not None else 0.0,
                'head': (_t5 - _t4b) if self.adapter_to_target is not None else (_t5 - _t4),
                'reshape_permute_out': (_t6 - _t5),
                'denorm_out': (_t7 - _t6) if self.use_norm else 0.0,
            }
            if isinstance(block_times, dict):
                prefixed = {f"blocks.{k}": float(v) for k, v in block_times.items()}
                self.last_forward_breakdown.update(prefixed)
        
        # Return based on mode
        if self.draft_stochastic:
            if self.output_attention:
                return (mean_out, logvar_out), attns
            return mean_out, logvar_out
        else:
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


