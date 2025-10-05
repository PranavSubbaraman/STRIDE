# Parameter Sweep Guide for Stochastic Draft

## Overview

Now that you have a working stochastic draft model with **79% acceptance rate**, this sweep will help you find the optimal hyperparameters for your specific use case.

## Quick Start

### 1. Run the Parameter Sweep

```bash
cd /home/pranavs/speculative-time-series-project/OpenLTM
conda activate open-ltm

# Run sweep (will take ~2-3 hours for all 80 configurations)
bash scripts/sweeps/stochastic_draft_param_sweep.sh
```

This will:
- Test 80 configurations (4 sigmas × 4 biases × 5 K values)
- Save results to `stochastic_sweep_results.csv`
- Log progress to `stochastic_sweep.log`
- Print top configurations at the end

### 2. Analyze Results

```bash
# Generate plots and detailed analysis
python scripts/sweeps/analyze_sweep_results.py stochastic_sweep_results.csv
```

This will:
- Print summary statistics
- Show top configurations by acceptance and MSE
- Generate 5 visualization plots in `sweep_plots/`
- Provide recommendations

### 3. View Results

```bash
# Quick view of CSV
column -t -s',' stochastic_sweep_results.csv | less -S

# View log
tail -100 stochastic_sweep.log

# View plots
ls sweep_plots/
# - acceptance_vs_mse.png
# - acceptance_heatmap.png
# - mse_heatmap.png
# - k_effect.png
# - pareto_frontier.png
```

## Parameters Being Swept

### `spec_sigma` (Target Model Variance)
- **Values**: [0.1, 0.25, 0.5, 1.0]
- **Effect**: Controls how much variance the target model is assumed to have
- **Lower**: More strict acceptance (draft must be very close)
- **Higher**: More lenient acceptance (allows more variance)
- **Current best**: 0.25 gave 79% acceptance

### `spec_accept_bias` (Acceptance Leniency)
- **Values**: [1.0, 2.0, 5.0, 10.0]
- **Effect**: Multiplies acceptance probability
- **Lower**: Stricter acceptance (better quality, lower rate)
- **Higher**: More lenient (higher acceptance, may degrade quality)
- **Current best**: 1.0 gave 79% acceptance

### `spec_k` (Speculation Length)
- **Values**: [1, 2, 3, 4, 5]
- **Effect**: Number of tokens to speculate ahead
- **Lower**: Less speculative, more frequent verification
- **Higher**: More speculative, potentially faster if accepted
- **Current**: 3 gave 79% acceptance

## What to Look For

### 1. High Acceptance with Good MSE
- **Goal**: Acceptance >= 70%, MSE close to baseline (0.475)
- **Why**: High acceptance → faster inference
- **Trade-off**: May need to accept slightly higher MSE

### 2. Pareto Optimal Points
- Configurations on the Pareto frontier give best acceptance-MSE trade-offs
- No other config can improve both metrics simultaneously

### 3. Variance Analysis
- Check if certain parameters consistently perform well
- Look for patterns (e.g., "sigma=0.25 always good")

## Customizing the Sweep

### Modify Parameter Ranges

Edit `scripts/sweeps/stochastic_draft_param_sweep.sh`:

```bash
# Line 47-49: Change these arrays
SIGMA_VALUES=(0.1 0.25 0.5 1.0 2.0)     # Add 2.0
BIAS_VALUES=(1.0 2.0 5.0 10.0 20.0)     # Add 20.0
K_VALUES=(1 2 3 4 5 6)                  # Add 6
```

### Run Subset for Quick Test

```bash
# Test only one K value (much faster)
SIGMA_VALUES=(0.1 0.25 0.5)
BIAS_VALUES=(1.0 2.0)
K_VALUES=(3)  # Only K=3
# → 6 configs instead of 80
```

### Parallel Execution

If you have multiple GPUs:

```bash
# Terminal 1 (GPU 0)
export CUDA_VISIBLE_DEVICES=0
# Run sweep with sigma=[0.1, 0.25]

# Terminal 2 (GPU 1)
export CUDA_VISIBLE_DEVICES=1
# Run sweep with sigma=[0.5, 1.0]

# Merge results later
cat sweep1.csv sweep2.csv > stochastic_sweep_results.csv
```

## Expected Results

Based on your current 79% acceptance with sigma=0.25, bias=1.0, k=3:

### Likely Outcomes:
1. **Lower sigma (0.1)**: 
   - Higher acceptance (draft is more confident)
   - May slightly degrade MSE
   
2. **Higher sigma (0.5, 1.0)**:
   - Lower acceptance (looser criteria)
   - Possibly better MSE

3. **Higher bias (2.0, 5.0)**:
   - Higher acceptance (more lenient)
   - Watch for MSE degradation

4. **Different K values**:
   - K=1,2: Lower latency per round, but more rounds
   - K=4,5: Higher latency if rejected, but fewer rounds if accepted

### Decision Criteria:

| Scenario | Recommended Config |
|----------|-------------------|
| **Maximize Speed** | Highest acceptance, K=4-5 |
| **Maintain Quality** | MSE ≤ 0.50, acceptance ≥ 60% |
| **Balanced** | MSE ≤ 0.52, acceptance ≥ 75% |

## Troubleshooting

### Sweep Taking Too Long
- Reduce K_VALUES to just [3]
- Use smaller sigma/bias ranges
- Test on smaller dataset first

### Out of Memory
- Reduce batch_size in sweep script (line 35)
- Use smaller K values

### All Configs Have Low Acceptance
- Your draft model quality might be poor
- Retrain with higher `draft_teacher_noise` (try 0.1)
- Try larger draft model (0.5x instead of 0.25x)

### MSE Too High Everywhere
- Draft model undertrained
- Teacher noise too high during distillation
- Need more training epochs

## After the Sweep

### 1. Select Best Configuration
```bash
# From analysis output, pick recommended config
# Example: sigma=0.25, bias=2.0, k=4
```

### 2. Update Test Script
```bash
# Edit test_etth1_timer_vs_spec_025x_stochastic.sh
--spec_sigma 0.25 \
--spec_accept_bias 2.0 \
--spec_k 4 \
```

### 3. Measure Actual Speedup
```bash
# Run both baseline and speculative, compare times
bash test_etth1_timer_vs_spec_025x_stochastic.sh
# Check result_inference_summary.txt for timing
```

### 4. Document Your Findings
```bash
# Add to your results
echo "Optimal: sigma=0.25, bias=2.0, k=4" >> RESULTS.md
echo "Acceptance: 85%, MSE: 0.505, Speedup: 1.8x" >> RESULTS.md
```

## Advanced: Sweeping Training Parameters

If sweep results are unsatisfactory, consider re-training draft with different parameters:

```bash
# In timer_xl_draft_from_target_etth1_stochastic.sh, try:

# Option 1: More noise (higher variance learning)
--draft_teacher_noise 0.1

# Option 2: Higher regularization (prevent variance collapse)
--draft_nll_var_reg 0.05

# Option 3: Teacher dropout (stochastic teacher)
--draft_teacher_dropout 0.1

# Option 4: More epochs
--train_epochs 20
```

Then re-run sweep with new checkpoint.

## Questions?

See:
- `STOCHASTIC_DRAFT_GUIDE.md` - Technical details
- `QUICKSTART_STOCHASTIC.md` - Quick start guide
- `stochastic_sweep.log` - Detailed sweep logs
