import torch
import torch.nn as nn
import time
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class DecoderOnlyLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderOnlyLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class TimerLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(TimerLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # Optional timing instrumentation (enabled externally)
        self.enable_timing = False
        self.timing_accum = {}
        self.timing_calls = 0

    def forward(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        timing_enabled = bool(getattr(self, 'enable_timing', False))
        if timing_enabled and hasattr(self, 'last_forward_timing'):
            # clear previous run record
            self.last_forward_timing = {}
        if timing_enabled and hasattr(self.attention, 'enable_timing'):
            # propagate if attention supports it (no-op otherwise)
            pass

        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter() if timing_enabled else None
        new_x, attn = self.attention(
            x, x, x,
            n_vars=n_vars,
            n_tokens=n_tokens,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t1 = time.perf_counter() if timing_enabled else None

        x = x + self.dropout(new_x)
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t2 = time.perf_counter() if timing_enabled else None

        y = x = self.norm1(x)
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t3 = time.perf_counter() if timing_enabled else None

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t4 = time.perf_counter() if timing_enabled else None

        y = self.dropout(self.conv2(y).transpose(-1, 1))
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t5 = time.perf_counter() if timing_enabled else None

        out = self.norm2(x + y)
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t6 = time.perf_counter() if timing_enabled else None

        if timing_enabled:
            last = {
                'attn': (_t1 - _t0),
                'residual1_dropout': (_t2 - _t1),
                'norm1': (_t3 - _t2),
                'mlp_conv1_act': (_t4 - _t3),
                'mlp_conv2': (_t5 - _t4),
                'residual2_norm2': (_t6 - _t5),
            }
            self.last_forward_timing = last
            # accumulate
            if not self.timing_accum:
                self.timing_accum = {k: 0.0 for k in last}
            for k, v in last.items():
                self.timing_accum[k] = self.timing_accum.get(k, 0.0) + float(v)
            self.timing_calls = int(getattr(self, 'timing_calls', 0)) + 1

        return out, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(
            conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask,
                      cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class DecoderOnly(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(DecoderOnly, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(
            conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class TimerBlock(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(TimerBlock, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(
            conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        # per-block timing accumulation from layers (optional)
        block_times = None
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
                if hasattr(attn_layer, 'last_forward_timing') and isinstance(attn_layer.last_forward_timing, dict):
                    if block_times is None:
                        block_times = {k: 0.0 for k in attn_layer.last_forward_timing}
                    for k, v in attn_layer.last_forward_timing.items():
                        block_times[k] += float(v)
            x, attn = self.attn_layers[-1](x, n_vars,
                                           n_tokens, tau=tau, delta=None)
            attns.append(attn)
            if hasattr(self.attn_layers[-1], 'last_forward_timing') and isinstance(self.attn_layers[-1].last_forward_timing, dict):
                if block_times is None:
                    block_times = {k: 0.0 for k in self.attn_layers[-1].last_forward_timing}
                for k, v in self.attn_layers[-1].last_forward_timing.items():
                    block_times[k] += float(v)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, n_vars, n_tokens,
                                     attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)
                if hasattr(attn_layer, 'last_forward_timing') and isinstance(attn_layer.last_forward_timing, dict):
                    if block_times is None:
                        block_times = {k: 0.0 for k in attn_layer.last_forward_timing}
                    for k, v in attn_layer.last_forward_timing.items():
                        block_times[k] += float(v)

        if self.norm is not None:
            x = self.norm(x)

        # expose block timing (sum over layers) for external consumers
        if isinstance(block_times, dict):
            self.last_forward_timing = block_times
            # accumulate across calls
            if not hasattr(self, 'timing_accum') or not isinstance(self.timing_accum, dict):
                self.timing_accum = {k: 0.0 for k in block_times}
            for k, v in block_times.items():
                self.timing_accum[k] = self.timing_accum.get(k, 0.0) + float(v)
            self.timing_calls = int(getattr(self, 'timing_calls', 0)) + 1

        return x, attns
