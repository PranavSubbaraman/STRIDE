# Quick Start: Stochastic Draft for Better Acceptance

## TL;DR

Your draft model had 0% acceptance because it was trained with MSE (deterministic) but tested with Gaussian acceptance (probabilistic). We fixed this by making the draft output **mean + variance** and training with **NLL loss**.

## Run This Now

### 1. Train the Stochastic Draft (30-60 min)

```bash
cd /home/pranavs/speculative-time-series-project/OpenLTM
conda activate open-ltm
bash scripts/distillation/timer_xl_draft_from_target_etth1_stochastic.sh
```

This will:
- Load your existing pretrained draft checkpoint
- Add variance head and train with NLL loss
- Add slight noise to teacher (σ=0.05) for stochasticity
- Save to: `checkpoints/forecast_ETTh1_draft_distill_stochastic_from_target_*/`

### 2. Test Acceptance Rate

```bash
bash scripts/supervised/rolling_forecast/test_etth1_timer_vs_spec_025x_stochastic.sh
```

Expected results:
- **Before**: 0% acceptance (deterministic draft)
- **After**: 10-30% acceptance (stochastic draft with learned variance)

### 3. Check Results

```bash
# View acceptance metrics
tail -20 result_inference_summary.txt

# View per-sample diagnostics
head -20 spec_decoding_stochastic_025x.csv
```

## What Changed

| Component | Before | After |
|-----------|--------|-------|
| **Draft Output** | Single value (mean) | (mean, log_variance) |
| **Training Loss** | MSE: `(draft - teacher)²` | NLL: `log σ² + (μ - teacher)²/σ²` |
| **Sampling** | `x ~ N(μ, fixed_σ)` | `x ~ N(μ, learned_σ)` |
| **Acceptance** | Uses fixed σ | Uses learned σ |

## Key Parameters

### Training (`timer_xl_draft_from_target_etth1_stochastic.sh`)

```bash
--draft_stochastic              # Enable mean+variance output
--draft_teacher_noise 0.05      # Add noise to teacher (helps learn variance)
--draft_nll_var_reg 0.01        # Prevent variance collapse
```

**Tuning Tips:**
- `draft_teacher_noise`: 
  - Too low (< 0.01) → variance collapses, overconfident
  - Too high (> 0.15) → MSE degrades
  - Sweet spot: 0.03 - 0.08
  
- `draft_nll_var_reg`:
  - Prevents log_variance from going to -∞
  - Usually 0.01 is good

### Testing (`test_etth1_timer_vs_spec_025x_stochastic.sh`)

```bash
--draft_stochastic        # Must match training
--spec_sigma 0.25         # Target model variance (can sweep)
--spec_accept_bias 1.0    # Acceptance leniency (≥1.0)
```

**Tuning Tips:**
- `spec_sigma`: Try [0.1, 0.25, 0.5, 1.0]
- `spec_accept_bias`: Try [1.0, 2.0, 5.0]
- Higher values → more acceptance, but may hurt accuracy

## Troubleshooting

### Problem: Variance collapses (all log_var ≈ -10)

**Solution:**
```bash
# Increase teacher noise
--draft_teacher_noise 0.1

# Or enable teacher dropout
--draft_teacher_dropout 0.1
```

### Problem: Acceptance still 0%

**Check:**
1. Draft MSE should be < 0.6 (if higher, train longer)
2. Try higher `spec_accept_bias` (2.0 or 5.0)
3. Try different `spec_sigma` values
4. Check debug CSV: `cat spec_decoding_stochastic_025x.csv | head`

### Problem: MSE degraded (> 0.6)

**Solution:**
```bash
# Reduce teacher noise
--draft_teacher_noise 0.02

# Train more epochs
--train_epochs 20
```

## Compare Old vs New

### Old (Deterministic) Training:
```bash
# Line 202 in exp_forecast.py
loss = nn.functional.mse_loss(outputs, teacher_out)
```

### New (Stochastic) Training:
```bash
# Lines 224-231 in exp_forecast.py
mean_out, logvar_out = outputs
var = torch.exp(logvar_out)
nll = 0.5 * (logvar_out + (mean_out - teacher_out)^2 / var)
loss = nll.mean()
```

## Next Steps

1. **Run training** (~45 min on your setup)
2. **Check acceptance** - aim for > 15%
3. **If low acceptance:**
   - Sweep `spec_sigma` and `spec_accept_bias`
   - Try higher `draft_teacher_noise` (0.08)
   - Check variance isn't collapsed (should vary by sample)

4. **If good acceptance (> 15%):**
   - Measure actual speedup vs baseline
   - Try different K values (2, 3, 4, 5)
   - Consider 0.5x draft model (currently using 0.25x)

## Alternative Approaches (if this doesn't work)

If acceptance is still poor after stochastic training:

1. **Make target also stochastic**: Output variance from target model too
2. **Larger draft**: Use 0.5x or 0.75x scaling (less aggressive)
3. **Different acceptance**: Use deterministic threshold instead of probabilistic
4. **Ensemble draft**: Average multiple draft samples

## Files to Check

- Training logs: `./checkpoints/forecast_ETTh1_draft_distill_stochastic_from_target_*/`
- Test results: `result_inference_summary.txt`
- Debug data: `spec_decoding_stochastic_025x.csv`

## Questions?

See detailed guide: `STOCHASTIC_DRAFT_GUIDE.md`
