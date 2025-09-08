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

    @torch.no_grad()
    def generate(self, x_init, x_mark, y_mark, steps):
        # x_init: [B, L, C]
        device = next(self.target_model.parameters()).device
        x = x_init.clone()
        accepted = 0
        attempted = 0

        preds = []
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
            # Extract p1..pk+1 by reading the next patch predictions for each prefix
            B = x.shape[0]
            accepted_in_round = 0
            for i in range(self.k):
                # p_{i+1} corresponds to prefix up to i proposals; target predicts next patch for that prefix
                p_next = target_out[i*B:(i+1)*B, -self.args.output_token_len:, :]
                q_i = q_list[i]
                x_i = proposals[i]
                # acceptance probability alpha = min(1, p(x_i)/q(x_i)) under isotropic Gaussian approx
                # log p ~ -||x-mu_p||^2/(2*sigma^2), log q ~ -||x-mu_q||^2/(2*sigma^2)
                sigma2 = max(self.args.spec_sigma ** 2, 1e-12)
                log_ratio = (-(x_i - p_next).pow(2).sum(dim=(1, 2)) + (x_i - q_i).pow(2).sum(dim=(1, 2))) / (2.0 * sigma2)
                ratio = torch.exp(torch.clamp(log_ratio, max=20.0))  # clamp for stability
                # optional multiplicative bias to relax acceptance (>=1 accepts more)
                ratio = ratio * max(self.args.spec_accept_bias, 1.0)
                alpha = torch.minimum(torch.ones_like(ratio), ratio)
                r = torch.rand_like(alpha)
                # optional MSE tolerance shortcut: accept if MSE <= tol
                if self.args.spec_accept_mse_tol > 0:
                    mse = torch.mean((p_next - x_i) ** 2, dim=(1, 2))
                    tol_accept = (mse <= self.args.spec_accept_mse_tol)
                else:
                    tol_accept = torch.zeros_like(r).bool()
                accept_mask = (r <= alpha) | tol_accept
                if torch.all(accept_mask):
                    x = torch.cat([x[:, self.args.input_token_len:, :], x_i], dim=1)
                    preds.append(x_i)
                    accepted += 1
                    accepted_in_round += 1
                else:
                    # rejection triggers resampling from p'(x) ~ max(0, p - q)
                    # continuous analogue: take a convex update toward p and away from q
                    residual = torch.relu(p_next - q_i)
                    denom = (residual.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)
                    residual = residual / denom
                    t = p_next - residual * self.args.spec_sigma
                    x = torch.cat([x[:, self.args.input_token_len:, :], t], dim=1)
                    preds.append(t)
                    # stop accepting more guesses this round per classical algorithm
                    break
            attempted += self.k

            # Fallback to target one-step if none accepted
            if accepted_in_round == 0:
                target_step = self.target_model(x, x_mark, y_mark)
                next_patch = target_step[:, -self.args.output_token_len:, :]
                x = torch.cat([x[:, self.args.input_token_len:, :], next_patch], dim=1)
                preds.append(next_patch)

            # Adapt K
            if self.adaptive and attempted > 0:
                acc_rate = accepted / attempted
                if acc_rate > 0.7:
                    self.k = min(self.k + 1, 8)
                elif acc_rate < 0.3:
                    self.k = max(self.k - 1, 1)

        return torch.cat(preds, dim=1)


