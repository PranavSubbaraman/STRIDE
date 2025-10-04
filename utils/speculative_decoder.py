import torch
import torch.nn.functional as F


def top_p_filtering(logits, top_p=1.0):
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    mask = cumulative_probs > top_p
    # Shift mask right to always include at least one token
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    filtered_logits = logits.clone()
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=torch.where(mask, torch.full_like(sorted_logits, float('-inf')), sorted_logits))
    return filtered_logits


class SpeculativeDecoder:
    def __init__(self, target_model, draft_model, args):
        self.target_model = target_model
        self.draft_model = draft_model
        self.args = args
        self.k = args.spec_k
        self.temp = args.spec_temp
        self.topp = args.spec_topp
        self.adaptive = args.spec_adaptive
        
        # Store normalization statistics for consistent normalization
        self.use_norm = getattr(args, 'use_norm', False)
        self.norm_means = None
        self.norm_stdev = None
        # debug
        self.debug_accept = bool(getattr(args, 'spec_debug_accept', False))
        self.debug_out = getattr(args, 'spec_debug_out', 'spec_accept_debug.csv')
        self.debug_n = int(getattr(args, 'spec_debug_n', 3))
        self.debug_max_batches = int(getattr(args, 'spec_debug_max_batches', 3))
        self.debug_max_rounds = int(getattr(args, 'spec_debug_max_rounds', 4))
        self._debug_batches_logged = 0
        # adaptive sigma
        self.sigma_mode = getattr(args, 'spec_sigma_mode', 'fixed')  # 'fixed' | 'adaptive'
        self.sigma_adapt_c = float(getattr(args, 'spec_sigma_adapt_c', 1.5))
        # timing controls
        self.time_exclude_norm = bool(getattr(args, 'time_exclude_norm', False))

    def _compute_norm_stats(self, x):
        """Compute normalization statistics from input sequence, matching Timer-XL's approach"""
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            return means, stdev
        return None, None
    
    def _normalize_patch(self, patch):
        """Normalize a patch using stored statistics"""
        if self.use_norm and self.norm_means is not None and self.norm_stdev is not None:
            # patch: [B, P, C] where P is patch length
            # norm stats: [B, 1, C]
            patch_normalized = (patch - self.norm_means) / self.norm_stdev
            return patch_normalized
        return patch

    @torch.no_grad()
    def generate(self, x_init, x_mark, y_mark, steps):
        # x_init: [B, L, C]
        device = next(self.target_model.parameters()).device
        x = x_init.clone()
        accepted = 0
        attempted = 0

        # accumulate variable number of patches per sample, then stack
        B = x.shape[0]
        preds_per_sample = [[] for _ in range(B)]
        patch_len = self.args.output_token_len
        max_return_len = steps * patch_len
        # timing breakdowns
        t_draft = 0.0
        t_prep_verify = 0.0
        t_target_verify = 0.0
        t_target_sample = 0.0
        t_accept_cpu = 0.0
        n_draft_calls = 0
        n_target_verify_calls = 0
        n_target_sample_calls = 0
        
        # Early-exit loop: stop when all samples have produced enough tokens
        # Each round guarantees at least one patch per sample (via rejection draw)
        while True:
            # Check completion
            all_done = True
            for b in range(B):
                if len(preds_per_sample[b]) * patch_len < max_return_len:
                    all_done = False
                    break
            if all_done:
                break
            # Draft speculation: propose K patches (x1..xk) from q1..qk
            proposals = []  # [x1, ..., xk]
            q_list = []     # [q1, ..., qk]
            context = x
            for _k in range(self.k):
                # time draft forward
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                import time as _pytime
                _t0 = _pytime.perf_counter()
                # AMP autocast for draft forward
                amp_enabled = bool(getattr(self.args, 'amp', False))
                amp_dtype_arg = str(getattr(self.args, 'amp_dtype', 'bf16')).lower()
                amp_dtype = torch.bfloat16 if amp_dtype_arg in ['bf16', 'bfloat16'] else torch.float16
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    draft_out = self.draft_model(context, x_mark, y_mark)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_draft += (_pytime.perf_counter() - _t0)
                n_draft_calls += 1
                
                # Handle stochastic draft output (mean, logvar)
                draft_stochastic = getattr(self.args, 'draft_stochastic', False)
                if draft_stochastic and isinstance(draft_out, tuple):
                    mean_draft, logvar_draft = draft_out
                    mu_q = mean_draft[:, -self.args.output_token_len:, :]
                    logvar_q = logvar_draft[:, -self.args.output_token_len:, :]
                    # Store both mean and logvar for acceptance criterion
                    q_list.append((mu_q, logvar_q))
                    # Sample using learned variance
                    sigma_q = torch.exp(0.5 * logvar_q)
                    sampled = mu_q + sigma_q * torch.randn_like(mu_q)
                    # Debug: print once per batch
                    if _k == 0 and len(q_list) == 1:  # First proposal of first round
                        print(f"[DEBUG] Stochastic draft active: logvar range=[{logvar_q.min():.2f}, {logvar_q.max():.2f}], sigma range=[{sigma_q.min():.4f}, {sigma_q.max():.4f}]")
                else:
                    # Original: treat draft output patch as mean of Gaussian q with fixed sigma
                    mu_q = draft_out[:, -self.args.output_token_len:, :]
                    q_list.append(mu_q)
                    sampled = mu_q + self.args.spec_sigma * torch.randn_like(mu_q)
                    # Debug: print once per batch
                    if _k == 0 and len(q_list) == 1:  # First proposal of first round
                        is_tuple = isinstance(draft_out, tuple)
                        print(f"[DEBUG] Non-stochastic path: draft_stochastic={draft_stochastic}, is_tuple={is_tuple}, output_shape={draft_out.shape if not is_tuple else (draft_out[0].shape, draft_out[1].shape)}")
                
                proposals.append(sampled)
                context = torch.cat([context[:, self.args.input_token_len:, :], sampled], dim=1)
            
            # Verification: Prefer single-pass block verification when compatible
            # NOTE: Single-pass is currently disabled because it gives the target model
            # visibility to future proposals when extracting p_next[i], causing mismatch
            # with draft predictions. Use parallel verification instead.
            single_pass_ok = False  # (self.args.output_token_len == self.args.input_token_len)
            p_list = []  # holds p_next for each proposal index i
            if single_pass_ok:
                # Build extended input per sample: x || p1 || p2 || ... || pK
                import time as _pytime
                _prep0 = _pytime.perf_counter()
                ext_inputs = torch.cat([x] + proposals, dim=1)  # [B, L + K*P, C]
                t_prep_verify += (_pytime.perf_counter() - _prep0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _tv0 = _pytime.perf_counter()
                # AMP autocast for target verify forward
                amp_enabled = bool(getattr(self.args, 'amp', False))
                amp_dtype_arg = str(getattr(self.args, 'amp_dtype', 'bf16')).lower()
                amp_dtype = torch.bfloat16 if amp_dtype_arg in ['bf16', 'bfloat16'] else torch.float16
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    target_full = self.target_model(ext_inputs, x_mark, y_mark)  # [B, N_ext*P_out, C]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_target_verify += (_pytime.perf_counter() - _tv0)
                n_target_verify_calls += 1
                # Determine window indices for each proposal
                # For proposal i, we want the target's prediction for what comes after x || x_0 || ... || x_{i-1}
                # The original context x has length L, so those predictions start at position L + i*out_P
                out_P = self.args.output_token_len
                L = x.shape[1]
                for i in range(self.k):
                    t0 = L + i * out_P
                    t1 = t0 + out_P
                    p_list.append(target_full[:, t0:t1, :])
            else:
                # Fallback: Parallel verification by batching K contexts (original method)
                import time as _pytime
                _prep0 = _pytime.perf_counter()
                verify_inputs = []
                current = x
                for p in proposals:
                    verify_inputs.append(current)
                    current = torch.cat([current[:, self.args.input_token_len:, :], p], dim=1)
                verify_inputs = torch.cat(verify_inputs, dim=0)  # [B*K, L, C]
                t_prep_verify += (_pytime.perf_counter() - _prep0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _tv0 = _pytime.perf_counter()
                # AMP autocast for target verify forward
                amp_enabled = bool(getattr(self.args, 'amp', False))
                amp_dtype_arg = str(getattr(self.args, 'amp_dtype', 'bf16')).lower()
                amp_dtype = torch.bfloat16 if amp_dtype_arg in ['bf16', 'bfloat16'] else torch.float16
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    target_out = self.target_model(verify_inputs, x_mark.repeat(len(proposals), 1, 1), y_mark.repeat(len(proposals), 1, 1))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_target_verify += (_pytime.perf_counter() - _tv0)
                n_target_verify_calls += 1
                # Will slice per i below from target_out
            accepted_in_round = torch.zeros(B, dtype=torch.long, device=x.device)
            done_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
            attempted_per_sample = torch.zeros(B, dtype=torch.long, device=x.device)

            # process proposals sequentially with per-example acceptance
            for i in range(self.k):
                # increment attempts for all still-active samples this round
                active_mask = (~done_mask)
                if active_mask.any():
                    attempted_per_sample[active_mask] += 1
                if single_pass_ok:
                    p_next = p_list[i]
                else:
                    p_next = target_out[i*B:(i+1)*B, -self.args.output_token_len:, :]
                q_i = q_list[i]
                x_i = proposals[i]
                
                # Note: When use_norm=True, both draft and target models internally normalize inputs
                # and denormalize outputs. So x_i, p_next, q_i are already in the original scale.
                # We should NOT normalize them again here - just use them directly for comparison.
                
                import time as _pytime
                _acc0 = _pytime.perf_counter()
                
                # Handle stochastic draft (with learned variance)
                draft_stochastic = getattr(self.args, 'draft_stochastic', False)
                if draft_stochastic and isinstance(q_i, tuple):
                    mu_q, logvar_q = q_i
                    # Use learned variance from draft model
                    var_q = torch.exp(logvar_q)
                    # For target, use fixed sigma (since target doesn't output variance)
                    sigma2_p = self.args.spec_sigma ** 2
                    
                    # Simplified acceptance with learned variance
                    # Use average learned variance as the draft proposal variance
                    # This makes the math simpler and more stable
                    avg_var_q = var_q.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
                    sigma2_q = avg_var_q.squeeze()  # [B]
                    
                    # Standard speculative decoding acceptance ratio with learned sigma_q:
                    # log p(x|target) / p(x|draft) ≈ [||x-μ_q||²/(2σ²_q) - ||x-p||²/(2σ²_p)]
                    sse_p = (x_i - p_next).pow(2).sum(dim=(1, 2))  # [B]
                    sse_q = (x_i - mu_q).pow(2).sum(dim=(1, 2))  # [B]
                    
                    log_ratio = -(sse_p / (2.0 * sigma2_p)) + (sse_q / (2.0 * sigma2_q))
                    
                    # Debug: print acceptance stats once
                    if i == 0 and len(proposals) == self.k:  # First proposal after all generated
                        print(f"[DEBUG ACCEPT] sigma2_q range=[{sigma2_q.min():.4f}, {sigma2_q.max():.4f}], sigma2_p={sigma2_p:.4f}")
                        print(f"[DEBUG ACCEPT] sse_p range=[{sse_p.min():.2f}, {sse_p.max():.2f}]")
                        print(f"[DEBUG ACCEPT] sse_q range=[{sse_q.min():.2f}, {sse_q.max():.2f}]")
                        print(f"[DEBUG ACCEPT] log_ratio range=[{log_ratio.min():.2f}, {log_ratio.max():.2f}]")
                        ratio_dbg = torch.exp(torch.clamp(log_ratio, max=20.0)) * self.args.spec_accept_bias
                        alpha_dbg = torch.minimum(torch.ones_like(ratio_dbg), ratio_dbg)
                        print(f"[DEBUG ACCEPT] alpha range=[{alpha_dbg.min():.4f}, {alpha_dbg.max():.4f}], mean={alpha_dbg.mean():.4f}")
                else:
                    # Original: extract mean if tuple (but shouldn't happen in non-stochastic mode)
                    mu_q = q_i[0] if isinstance(q_i, tuple) else q_i
                    # choose sigma
                    if self.sigma_mode == 'adaptive':
                        # adapt from mean squared diff between p and q
                        # avoid zeros; cap for stability
                        mean_sq = torch.mean((p_next - mu_q) ** 2, dim=(1, 2))  # [B]
                        # per-sample sigma^2 = c * mean_sq; we broadcast later
                        sigma2_vec = torch.clamp(self.sigma_adapt_c * mean_sq, min=1e-6, max=1e6)
                        # for computation below, we will expand to [B]
                        sigma2 = sigma2_vec
                    else:
                        sigma2 = torch.full((x.shape[0],), max(self.args.spec_sigma ** 2, 1e-12), device=x.device, dtype=x.dtype)
                    log_ratio = (-(x_i - p_next).pow(2).sum(dim=(1, 2)) + (x_i - mu_q).pow(2).sum(dim=(1, 2))) / (2.0 * sigma2)
                ratio = torch.exp(torch.clamp(log_ratio, max=20.0))
                ratio = ratio * max(self.args.spec_accept_bias, 1.0)
                alpha = torch.minimum(torch.ones_like(ratio), ratio)
                r = torch.rand_like(alpha)
                if self.args.spec_accept_mse_tol > 0:
                    mse = torch.mean((p_next - x_i) ** 2, dim=(1, 2))
                    tol_accept = (mse <= self.args.spec_accept_mse_tol)
                else:
                    tol_accept = torch.zeros_like(r).bool()
                accept_mask = ((r <= alpha) | tol_accept) & (~done_mask)
                # debug log
                if self.debug_accept and self._debug_batches_logged < self.debug_max_batches and i < self.debug_max_rounds:
                    try:
                        import csv
                        # compute per-sample diagnostics (no extra normalization)
                        # SSE_p = ||x_i - mu_p||^2, SSE_q = ||x_i - mu_q||^2
                        draft_stochastic = getattr(self.args, 'draft_stochastic', False)
                        if draft_stochastic and isinstance(q_i, tuple):
                            mu_q_dbg, logvar_q_dbg = q_i
                            sse_q = (x_i - mu_q_dbg).pow(2).sum(dim=(1, 2))
                        else:
                            mu_q_dbg = q_i[0] if isinstance(q_i, tuple) else q_i
                            sse_q = (x_i - mu_q_dbg).pow(2).sum(dim=(1, 2))
                        sse_p = (x_i - p_next).pow(2).sum(dim=(1, 2))
                        # log densities up to additive const: log p ~ -SSE_p/(2 sigma^2), log q ~ -SSE_q/(2 sigma^2)
                        if draft_stochastic and isinstance(q_i, tuple):
                            sigma2_dbg = self.args.spec_sigma ** 2
                            denom_vec = 2.0 * sigma2_dbg
                        else:
                            denom_vec = 2.0 * sigma2
                        logp = -sse_p / denom_vec
                        logq = -sse_q / denom_vec
                        log_ratio_dbg = logp - logq
                        # compute scalar diagnostics per sample
                        with open(self.debug_out, 'a', newline='') as f:
                            writer = csv.writer(f)
                            # header once
                            if f.tell() == 0:
                                writer.writerow([
                                    'batch_idx','round_i','sample_idx','kind',
                                    'alpha','r','tol_accept','accept',
                                    'mse','sse_p','sse_q','logp','logq','log_ratio','ratio',
                                    'sigma_eff','sigma_mode','bias',
                                    'mean_xi_minus_q','var_xi_minus_q','mean_xi_minus_p','var_xi_minus_p'
                                ])
                            # pick first N active samples to log
                            active_indices = torch.where(~done_mask)[0].tolist()
                            for b in active_indices[:self.debug_n]:
                                mse_b = torch.mean((p_next[b] - x_i[b]) ** 2).item()
                                ratio_b = float(torch.exp(torch.clamp(log_ratio_dbg[b], max=20.0)).item())
                                sigma_eff_b = float(torch.sqrt(sigma2[b]).item())
                                # empirical mean/var of proposal residuals
                                _dq = (x_i[b] - q_i[b]).reshape(-1)
                                _dp = (x_i[b] - p_next[b]).reshape(-1)
                                mean_dq = float(torch.mean(_dq).item())
                                var_dq = float(torch.var(_dq, unbiased=False).item())
                                mean_dp = float(torch.mean(_dp).item())
                                var_dp = float(torch.var(_dp, unbiased=False).item())
                                writer.writerow([
                                    int(self._debug_batches_logged), int(i), int(b), 'proposal',
                                    float(alpha[b].item()), float(r[b].item()), bool(tol_accept[b].item()), bool(accept_mask[b].item()),
                                    float(mse_b), float(sse_p[b].item()), float(sse_q[b].item()), float(logp[b].item()), float(logq[b].item()), float(log_ratio_dbg[b].item()),
                                    ratio_b,
                                    sigma_eff_b, str(self.sigma_mode), float(self.args.spec_accept_bias),
                                    mean_dq, var_dq, mean_dp, var_dp
                                ])
                    except Exception:
                        pass
                _acc1 = _pytime.perf_counter()
                t_accept_cpu += (_acc1 - _acc0)

                # update accepted samples
                if accept_mask.any():
                    # shift-append proposal for accepted ones
                    for b in torch.where(accept_mask)[0].tolist():
                        x[b] = torch.cat([x[b, self.args.input_token_len:, :], x_i[b]], dim=0)
                        preds_per_sample[b].append(x_i[b])
                        accepted_in_round[b] += 1

                # handle first rejection at this i for samples not yet done
                reject_mask = (~accept_mask) & (~done_mask)
                if reject_mask.any():
                    # draw t ~ N(mu_p=p_next, sigma)
                    noise = self.args.spec_sigma * torch.randn_like(p_next)
                    t_full = p_next + noise
                    # optional debug for target draws: empirical mean/var of noise
                    if self.debug_accept and self._debug_batches_logged < self.debug_max_batches and i < self.debug_max_rounds:
                        try:
                            import csv
                            with open(self.debug_out, 'a', newline='') as f:
                                writer = csv.writer(f)
                                active_indices = torch.where(reject_mask)[0].tolist()
                                for b in active_indices[:self.debug_n]:
                                    sigma_eff_b = float(torch.sqrt(sigma2[b]).item())
                                    _dp_draw = (t_full[b] - p_next[b]).reshape(-1)
                                    mean_dp_draw = float(torch.mean(_dp_draw).item())
                                    var_dp_draw = float(torch.var(_dp_draw, unbiased=False).item())
                                    writer.writerow([
                                        int(self._debug_batches_logged), int(i), int(b), 'target_draw',
                                        float('nan'), float('nan'), '', '',
                                        float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                                        float('nan'),
                                        sigma_eff_b, str(self.sigma_mode), float(self.args.spec_accept_bias),
                                        float('nan'), float('nan'), mean_dp_draw, var_dp_draw
                                    ])
                        except Exception:
                            pass
                    for b in torch.where(reject_mask)[0].tolist():
                        x[b] = torch.cat([x[b, self.args.input_token_len:, :], t_full[b]], dim=0)
                        preds_per_sample[b].append(t_full[b])
                        done_mask[b] = True

                # If all samples are done for this round, early stop
                if done_mask.all():
                    break

            # For samples that accepted all K proposals (not done), append one more target draw at K+1
            all_accept_mask = (~done_mask)
            if all_accept_mask.any():
                idx = torch.where(all_accept_mask)[0]
                x_all = x.index_select(0, idx)
                xm_all = x_mark.index_select(0, idx)
                ym_all = y_mark.index_select(0, idx)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                import time as _pytime
                _ts0 = _pytime.perf_counter()
                # AMP autocast for target draw forward
                amp_enabled = bool(getattr(self.args, 'amp', False))
                amp_dtype_arg = str(getattr(self.args, 'amp_dtype', 'bf16')).lower()
                amp_dtype = torch.bfloat16 if amp_dtype_arg in ['bf16', 'bfloat16'] else torch.float16
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    target_step = self.target_model(x_all, xm_all, ym_all)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_target_sample += (_pytime.perf_counter() - _ts0)
                n_target_sample_calls += 1
                mu_next = target_step[:, -self.args.output_token_len:, :]
                t_all = mu_next + self.args.spec_sigma * torch.randn_like(mu_next)
                # update x and preds
                for j, b in enumerate(idx.tolist()):
                    x[b] = torch.cat([x[b, self.args.input_token_len:, :], t_all[j]], dim=0)
                    preds_per_sample[b].append(t_all[j])
                accepted_in_round[all_accept_mask] += 0  # proposals already counted

            # accounting for adaptation
            accepted += int(accepted_in_round.sum().item())
            attempted += int(attempted_per_sample.sum().item())
            # bump batch debug counter once per outer loop
            if self.debug_accept and self._debug_batches_logged < self.debug_max_batches:
                self._debug_batches_logged += 1

            # Adapt K using global acceptance rate
            if self.adaptive and attempted > 0:
                acc_rate = accepted / attempted
                if acc_rate > 0.7:
                    self.k = min(self.k + 1, 8)
                elif acc_rate < 0.3:
                    self.k = max(self.k - 1, 1)

        # collate per-sample predictions to fixed length [B, steps*P, C]
        out_list = []
        for b in range(B):
            if len(preds_per_sample[b]) == 0:
                # unlikely, but pad zeros
                out_b = torch.zeros((0, x.shape[-1]), device=x.device, dtype=x.dtype)
            else:
                out_b = torch.cat(preds_per_sample[b], dim=0)  # [T, C]
            # truncate to max_return_len
            if out_b.shape[0] > max_return_len:
                out_b = out_b[:max_return_len, :]
            # pad by repeating last patch if needed
            if out_b.shape[0] < max_return_len:
                if out_b.shape[0] == 0:
                    pad = torch.zeros((max_return_len, out_b.shape[-1]), device=x.device, dtype=x.dtype)
                else:
                    last = out_b[-1:, :].expand(max_return_len - out_b.shape[0], -1)
                    pad = last
                out_b = torch.cat([out_b, pad], dim=0)
            out_list.append(out_b)
        breakdown = {
            't_draft': float(t_draft),
            't_prep_verify': float(t_prep_verify),
            't_target_verify': float(t_target_verify),
            't_target_sample': float(t_target_sample),
            't_accept_cpu': float(t_accept_cpu),
            'n_draft_calls': int(n_draft_calls),
            'n_target_verify_calls': int(n_target_verify_calls),
            'n_target_sample_calls': int(n_target_sample_calls),
        }
        return torch.stack(out_list, dim=0), int(accepted), int(attempted), breakdown


