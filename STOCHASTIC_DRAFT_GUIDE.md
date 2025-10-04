# Stochastic Draft Model for Improved Acceptance Rate

## Problem Statement

The original draft model was trained with MSE loss to match the teacher's deterministic output. However, at inference time, the speculative decoding acceptance criterion assumes both models output samples from Gaussian distributions. This mismatch resulted in 0% acceptance rate because:

1. **Deterministic Training**: Draft was trained to match teacher's mean via MSE loss
2. **Probabilistic Inference**: Acceptance criterion uses Gaussian likelihood with fixed variance
3. **Mismatch**: Draft cannot learn to model uncertainty, leading to poor acceptance

## Solution

We implemented a **stochastic draft model** that:
1. Outputs both **mean** and **log_variance** (dual-head architecture)
2. Trains with **Gaussian NLL (Negative Log-Likelihood) loss** instead of MSE
3. Learns to estimate uncertainty in its predictions
4. Uses learned variance during speculative decoding acceptance

Optionally, we also support:
- Adding Gaussian noise to teacher outputs during distillation
- Enabling dropout on the teacher during distillation
- Variance regularization to prevent collapse

## Changes Made

### 1. Model Architecture (`models/timer_xl_draft.py`)

Added dual-head output when `draft_stochastic=True`:
- `head_mean`: outputs prediction mean
- `head_logvar`: outputs log of variance (log σ²)
- Returns tuple `(mean, logvar)` instead of single tensor

Key features:
- Log variance clamped to [-10, 10] for numerical stability
- Variance properly scaled with normalization (scales with stdev²)
- Backward compatible (defaults to single head when flag is off)

### 2. Training Loss (`exp/exp_forecast.py`)

For stochastic distillation:
- **NLL Loss**: `loss = 0.5 * (log σ² + (μ - y)² / σ²)`
- **Variance Regularization**: Optional penalty on very small variances
- **Teacher Noise**: Optional Gaussian noise added to teacher outputs
- **Teacher Dropout**: Optional dropout enabled during distillation

### 3. Speculative Decoder (`utils/speculative_decoder.py`)

Updated to use learned variance:
- Draft samples using `x ~ N(μ_q, σ²_q)` with learned σ²_q
- Acceptance ratio uses proper Gaussian likelihood:
  ```
  log_ratio = -||x - μ_p||²/(2σ²_p) + ||x - μ_q||²/(2σ²_q) + 0.5*log σ²_q
  ```
- Backward compatible with non-stochastic mode

### 4. Command-Line Arguments (`run.py`)

New flags:
- `--draft_stochastic`: Enable stochastic draft (mean + variance output)
- `--draft_nll_var_reg`: Variance regularization weight (default: 0.0)
- `--draft_teacher_noise`: Gaussian noise std for teacher (default: 0.0)
- `--draft_teacher_dropout`: Dropout rate for teacher (default: 0.0)

## Usage

### Step 1: Train Stochastic Draft Model

Use the new distillation script:

```bash
cd /home/pranavs/speculative-time-series-project/OpenLTM
bash scripts/distillation/timer_xl_draft_from_target_etth1_stochastic.sh
```

Key parameters in the script:
- `--draft_stochastic`: Enables NLL training
- `--draft_nll_var_reg 0.01`: Small regularization to prevent variance collapse
- `--draft_teacher_noise 0.05`: Adds slight noise to teacher for stochasticity

You can adjust these parameters:
- **Teacher noise** (0.0 - 0.1): Higher = more uncertainty learned, but may hurt accuracy
- **Var regularization** (0.0 - 0.05): Prevents variance from collapsing to zero
- **Teacher dropout** (0.0 - 0.2): Alternative to noise, uses dropout on teacher

### Step 2: Test with Speculative Decoding

Run the test script:

```bash
bash scripts/supervised/rolling_forecast/test_etth1_timer_vs_spec_025x_stochastic.sh
```

Key parameters:
- `--draft_stochastic`: Must match training (enables variance usage)
- `--spec_sigma 0.25`: Target model variance (can tune this)
- `--spec_accept_bias 1.0`: Acceptance bias (≥1.0, higher = more lenient)

### Step 3: Sweep Parameters

After training, you may want to sweep `spec_sigma` and `spec_accept_bias`:

```bash
# Try different sigma values
for sigma in 0.1 0.25 0.5 1.0; do
  for bias in 1.0 2.0 5.0; do
    echo "Testing sigma=$sigma, bias=$bias"
    python -u run.py \
      --task_name forecast \
      --is_training 0 \
      --model timer_xl \
      --use_speculative \
      --spec_draft_model timer_xl_draft \
      --draft_stochastic \
      --spec_sigma $sigma \
      --spec_accept_bias $bias \
      # ... (other args)
  done
done
```

## Expected Improvements

With stochastic training:
1. **Higher Acceptance Rate**: Draft learns when it's uncertain, improving matches
2. **Better Calibration**: Variance reflects actual prediction uncertainty
3. **Tunable Trade-off**: Can adjust sigma/bias to balance acceptance vs. accuracy

## Theory: Why This Works

### Original Problem
- **Draft training**: min MSE(draft, teacher) → learns E[teacher|x]
- **Acceptance**: p(x|target) / p(x|draft) with fixed σ²
- **Issue**: Draft doesn't model P(teacher|x), just E[teacher|x]

### Stochastic Solution
- **Draft training**: max log p(teacher | draft_μ, draft_σ²)
- **Draft learns**: μ ≈ E[teacher|x] and σ² ≈ Var[teacher|x]
- **Acceptance**: Uses learned σ² → properly models distribution

### With Teacher Noise
- **Teacher becomes**: y ~ N(teacher_mean, noise_σ²)
- **Draft learns**: Distribution that explains noisy teacher
- **Benefit**: Prevents overconfident (σ² → 0) predictions

## Troubleshooting

### Variance Collapses to Zero
- Increase `--draft_teacher_noise` (try 0.1)
- Increase `--draft_nll_var_reg` (try 0.05)
- Enable `--draft_teacher_dropout 0.1`

### Acceptance Still Low
- Check draft MSE is reasonable (< 0.6)
- Try higher `--spec_accept_bias` (2.0 - 5.0)
- Try different `--spec_sigma` values
- Check `spec_debug_out` CSV for diagnostics

### MSE Degrades Too Much
- Reduce `--draft_teacher_noise` (try 0.01)
- Reduce `--draft_nll_var_reg` to 0
- Train longer (more epochs)

## Files Modified

1. `models/timer_xl_draft.py` - Dual-head architecture
2. `exp/exp_forecast.py` - NLL loss and teacher noise
3. `utils/speculative_decoder.py` - Learned variance usage
4. `run.py` - New command-line arguments
5. `scripts/distillation/timer_xl_draft_from_target_etth1_stochastic.sh` - Training script
6. `scripts/supervised/rolling_forecast/test_etth1_timer_vs_spec_025x_stochastic.sh` - Test script

## Backward Compatibility

All changes are backward compatible:
- Without `--draft_stochastic` flag, behaves exactly as before
- Existing checkpoints continue to work
- Can mix stochastic and non-stochastic runs

## References

- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
- [Knowledge Distillation with Uncertainty](https://arxiv.org/abs/1910.12656)
- Timer-XL: Long-Context Transformers for Unified Time Series Forecasting

## Next Steps

1. **Train the stochastic draft model** (15 epochs, ~30-60 min)
2. **Test acceptance rate** - should be > 10% (vs 0% before)
3. **Sweep parameters** to find optimal sigma/bias
4. **Compare MSE** - should be similar to non-stochastic draft
5. **Measure speedup** - higher acceptance → faster inference

If acceptance is still low after these changes, the next steps would be:
- Make target model also stochastic (output variance)
- Use larger draft model (0.5x instead of 0.25x)
- Try different distillation objectives (KL divergence)
