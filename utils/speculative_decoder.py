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
        
        # Compute normalization statistics from initial input
        self.norm_means, self.norm_stdev = self._compute_norm_stats(x_init)

        # accumulate variable number of patches per sample, then stack
        B = x.shape[0]
        preds_per_sample = [[] for _ in range(B)]
        patch_len = self.args.output_token_len
        max_return_len = steps * patch_len

        for _ in range(steps):
            # Draft speculation: propose K patches (x1..xk) from q1..qk
            proposals = []  # [x1, ..., xk]
            q_list = []     # [q1, ..., qk]
            context = x
            for _k in range(self.k):
                draft_out = self.draft_model(context, x_mark, y_mark)
                # Treat draft output patch as mean of Gaussian q with fixed sigma (continuous tokens)
                mu_q = draft_out[:, -self.args.output_token_len:, :]
                q_list.append(mu_q)
                sampled = mu_q + self.args.spec_sigma * torch.randn_like(mu_q)
                proposals.append(sampled)
                context = torch.cat([context[:, self.args.input_token_len:, :], sampled], dim=1)

            # Parallel verification: batch context prefixes
            # Build contexts: x0, x0+spec1, x0+spec1+spec2, ...
            verify_inputs = []
            current = x
            for p in proposals:
                verify_inputs.append(current)
                current = torch.cat([current[:, self.args.input_token_len:, :], p], dim=1)
            verify_inputs = torch.cat(verify_inputs, dim=0)  # [B*K, L, C]

            target_out = self.target_model(verify_inputs, x_mark.repeat(len(proposals), 1, 1), y_mark.repeat(len(proposals), 1, 1))
            # Extract p1..pK (target next-patch means for each prefix)
            accepted_in_round = torch.zeros(B, dtype=torch.long, device=x.device)
            done_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
            attempted_per_sample = torch.zeros(B, dtype=torch.long, device=x.device)

            # process proposals sequentially with per-example acceptance
            for i in range(self.k):
                # increment attempts for all still-active samples this round
                active_mask = (~done_mask)
                if active_mask.any():
                    attempted_per_sample[active_mask] += 1
                p_next = target_out[i*B:(i+1)*B, -self.args.output_token_len:, :]
                q_i = q_list[i]
                x_i = proposals[i]
                
                # Normalize x_i using training dataset statistics for consistent comparison
                x_i_norm = self._normalize_patch(x_i)
                p_next_norm = self._normalize_patch(p_next)
                q_i_norm = self._normalize_patch(q_i)

                sigma2 = max(self.args.spec_sigma ** 2, 1e-12)
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
                target_step = self.target_model(x_all, xm_all, ym_all)
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
        return torch.stack(out_list, dim=0), int(accepted), int(attempted)


