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
        
        # Compute normalization statistics from initial input (track timing)
        import time as _pytime
        _tn0 = _pytime.perf_counter()
        self.norm_means, self.norm_stdev = self._compute_norm_stats(x_init)
        t_norm = _pytime.perf_counter() - _tn0

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
                # Treat draft output patch as mean of Gaussian q with fixed sigma (continuous tokens)
                mu_q = draft_out[:, -self.args.output_token_len:, :]
                q_list.append(mu_q)
                sampled = mu_q + self.args.spec_sigma * torch.randn_like(mu_q)
                proposals.append(sampled)
                context = torch.cat([context[:, self.args.input_token_len:, :], sampled], dim=1)
            
            # Verification: Prefer single-pass block verification when compatible
            single_pass_ok = (self.args.output_token_len == self.args.input_token_len)
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
                in_P = self.args.input_token_len
                out_P = self.args.output_token_len
                base_N = x.shape[1] // in_P  # number of input windows in the current context
                for i in range(self.k):
                    j_idx = base_N - 1 + i
                    t0 = j_idx * out_P
                    t1 = (j_idx + 1) * out_P
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
                
                # Normalize x_i using training dataset statistics for consistent comparison (track timing)
                _tn1 = _pytime.perf_counter()
                x_i_norm = self._normalize_patch(x_i)
                p_next_norm = self._normalize_patch(p_next)
                q_i_norm = self._normalize_patch(q_i)
                t_norm += (_pytime.perf_counter() - _tn1)

                import time as _pytime
                _acc0 = _pytime.perf_counter()
                # choose sigma
                if self.sigma_mode == 'adaptive':
                    # adapt from mean squared diff between p and q (normalized space)
                    # avoid zeros; cap for stability
                    mean_sq = torch.mean((p_next_norm - q_i_norm) ** 2, dim=(1, 2))  # [B]
                    # per-sample sigma^2 = c * mean_sq; we broadcast later
                    sigma2_vec = torch.clamp(self.sigma_adapt_c * mean_sq, min=1e-6, max=1e6)
                    # for computation below, we will expand to [B]
                    sigma2 = sigma2_vec
                else:
                    sigma2 = torch.full((x.shape[0],), max(self.args.spec_sigma ** 2, 1e-12), device=x.device, dtype=x.dtype)
                log_ratio = (-(x_i_norm - p_next_norm).pow(2).sum(dim=(1, 2)) + (x_i_norm - q_i_norm).pow(2).sum(dim=(1, 2))) / (2.0 * sigma2)
                ratio = torch.exp(torch.clamp(log_ratio, max=20.0))
                ratio = ratio * max(self.args.spec_accept_bias, 1.0)
                alpha = torch.minimum(torch.ones_like(ratio), ratio)
                r = torch.rand_like(alpha)
                if self.args.spec_accept_mse_tol > 0:
                    mse = torch.mean((p_next_norm - x_i_norm) ** 2, dim=(1, 2))
                    tol_accept = (mse <= self.args.spec_accept_mse_tol)
                else:
                    tol_accept = torch.zeros_like(r).bool()
                accept_mask = ((r <= alpha) | tol_accept) & (~done_mask)
                # debug log
                if self.debug_accept and self._debug_batches_logged < self.debug_max_batches and i < self.debug_max_rounds:
                    try:
                        import csv
                        # compute per-sample diagnostics in normalized space
                        # SSE_p = ||x_i - mu_p||^2, SSE_q = ||x_i - mu_q||^2
                        sse_p = (x_i_norm - p_next_norm).pow(2).sum(dim=(1, 2))
                        sse_q = (x_i_norm - q_i_norm).pow(2).sum(dim=(1, 2))
                        # log densities up to additive const: log p ~ -SSE_p/(2 sigma^2), log q ~ -SSE_q/(2 sigma^2)
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
                                mse_b = torch.mean((p_next_norm[b] - x_i_norm[b]) ** 2).item()
                                ratio_b = float(torch.exp(torch.clamp(log_ratio_dbg[b], max=20.0)).item())
                                sigma_eff_b = float(torch.sqrt(sigma2[b]).item())
                                # empirical mean/var of proposal residuals
                                _dq = (x_i_norm[b] - q_i_norm[b]).reshape(-1)
                                _dp = (x_i_norm[b] - p_next_norm[b]).reshape(-1)
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
                            _tn2 = _pytime.perf_counter()
                            t_full_norm = self._normalize_patch(t_full)
                            t_norm += (_pytime.perf_counter() - _tn2)
                            with open(self.debug_out, 'a', newline='') as f:
                                writer = csv.writer(f)
                                active_indices = torch.where(reject_mask)[0].tolist()
                                for b in active_indices[:self.debug_n]:
                                    sigma_eff_b = float(torch.sqrt(sigma2[b]).item())
                                    _dp_draw = (t_full_norm[b] - p_next_norm[b]).reshape(-1)
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
            't_norm': float(t_norm),
            'n_draft_calls': int(n_draft_calls),
            'n_target_verify_calls': int(n_target_verify_calls),
            'n_target_sample_calls': int(n_target_sample_calls),
        }
        return torch.stack(out_list, dim=0), int(accepted), int(attempted), breakdown


