# Accelerating Time Series Foundation Models with Speculative Decoding

## Table of Contents
- [Setup](#setup)
- [Datasets](#datasets)
- [Experiments](#experiments)
  - [1. Pretraining Draft Models](#1-pretraining-draft-models)
  - [2. Fine-tuning Target Models](#2-fine-tuning-target-models)
  - [3. Distillation](#3-distillation)
  - [4. Speculative Decoding Tests](#4-speculative-decoding-tests)
  - [5. Parameter Sweeps](#5-parameter-sweeps)
- [Key Parameters](#key-parameters)
- [Results and Outputs](#results-and-outputs)

## Setup

### Environment Setup

1. Create and activate the conda environment:
```bash
conda create -n my-env python=3.10
conda activate my-env
```

2. Install dependencies:
```bash
pip install einops==0.8.0 matplotlib==3.9.2 numpy==1.26.0 pandas==2.2.3 scikit-learn==1.5.2 transformers
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

## Datasets

The following datasets are available in the `dataset/` directory:

- **ETT-small**: `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`
- **Weather**: `weather.csv`
- **Traffic**: `traffic.csv`
- **Electricity**: `electricity.csv`
- **Solar**: `solar_AL.txt`
- **PEMS**: `PEMS03.npz`, `PEMS04.npz`, `PEMS07.npz`, `PEMS08.npz`

## Experiments

### 1. Pretraining Draft Models

Pretrain a smaller draft model on a large dataset (e.g., electricity, weather, traffic).

**Example: Pretrain 0.25x draft model on weather**
```bash
bash scripts/pretrain/increased_batch_and_seq_len/timer_xl_draft_weather_025x.sh
```

**Key configurations:**
- `--draft_scale_d_model 0.25`: Model dimension scaled to 25% of target
- `--draft_scale_n_heads 0.25`: Number of attention heads scaled to 25%
- `--draft_scale_d_ff 0.25`: Feed-forward dimension scaled to 25%
- `--draft_scale_e_layers 0.25`: Number of layers scaled to 25%

**Other pretrain scripts:**
- `timer_xl_draft_traffic_025x.sh`: Traffic dataset
- `timer_xl_draft_ecl_025x.sh`: Electricity dataset
- `timer_xl_draft_weather_001x.sh`: Smaller 0.01x draft model

**Output:** Checkpoints saved to `./checkpoints/draft_pretrain_*`

### 2. Fine-tuning Target Models

Train or test the full Timer-XL target model on specific benchmarks.

**Example: Test Timer-XL on weather dataset**
```bash
bash scripts/supervised/rolling_forecast/timer_xl_weather.sh
```

**Example: Test Timer-XL on ETTm2 dataset**
```bash
bash scripts/supervised/rolling_forecast/timer_xl_ettm2.sh
```

**Available benchmark scripts:**
- `timer_xl_etth1.sh`, `timer_xl_etth2.sh`
- `timer_xl_ettm1.sh`, `timer_xl_ettm2.sh`
- `timer_xl_traffic.sh`, `timer_xl_weather.sh`
- `timer_xl_ecl.sh`

**Training mode:** Uncomment the training section in the script (currently commented out in most scripts).

**Output:** Checkpoints in `./checkpoints/forecast_*_timer_xl_*`

### 3. Distillation

Fine-tune the pretrained draft model to mimic the target model distribution using distillation.

**Example: Distill draft model on ETTm2 (stochastic)**
```bash
conda activate open-ltm
bash scripts/distillation/updated_distillation/ettm2_timer_xl_draft_from_target_ettm2_stochastic.sh
```

**What distillation does:**
- Loads pretrained draft model (from `--pretrain_model_path`)
- Loads target teacher model (from `--distill_target_ckpt`)
- Trains draft to match teacher's outputs using NLL loss
- Uses stochastic mode (`--draft_stochastic`) to output mean + variance
- Adds noise to teacher (`--draft_teacher_noise 0.05`) for better variance learning

**Key parameters:**
- `--distill`: Enable distillation mode
- `--distill_target_model timer_xl`: Target model architecture
- `--distill_target_ckpt`: Path to target checkpoint
- `--draft_stochastic`: Enable stochastic output (mean + log-variance)
- `--draft_nll_var_reg 0.01`: Variance regularization
- `--draft_teacher_noise 0.05`: Teacher output noise level

**Available distillation scripts:**
- `etth1_timer_xl_draft_from_target_etth1_stochastic.sh`
- `etth2_timer_xl_draft_from_target_etth2_stochastic.sh`
- `ettm1_timer_xl_draft_from_target_ettm1_stochastic.sh`
- `ettm2_timer_xl_draft_from_target_ettm2_stochastic.sh`
- `traffic_timer_xl_draft_from_target_traffic_stochastic.sh`

**Output:** Checkpoints in `./checkpoints/forecast_*_draft_distill_stochastic_from_target_*`

### 4. Speculative Decoding Tests

Run inference comparing baseline Timer-XL against speculative decoding with draft model.

**Example: Test speculative decoding on weather**
```bash
bash scripts/supervised/rolling_forecast/spec_decoding_tests/test_weather_timer_vs_spec_025x_stochastic.sh
```

**Example: Test on ETTm2**
```bash
bash scripts/supervised/rolling_forecast/spec_decoding_tests/test_ettm2_timer_vs_spec_025x_stochastic.sh
```

**What these scripts do:**
1. (Optional) Run baseline Timer-XL inference
2. Run speculative decoding with Timer-XL + draft model
3. Report acceptance rate and speedup metrics

**Key speculative decoding parameters:**
- `--use_speculative`: Enable speculative decoding
- `--spec_draft_model timer_xl_draft`: Draft model architecture
- `--spec_draft_ckpt`: Path to draft checkpoint
- `--spec_k 3`: Number of draft tokens to generate before verification
- `--spec_sigma 0.7`: Target model variance for acceptance criterion
- `--spec_accept_bias 1.0`: Acceptance threshold adjustment (≥1.0)
- `--draft_stochastic`: Use stochastic draft (must match training)
- `--spec_debug_accept`: Enable acceptance debugging
- `--spec_debug_out`: CSV file for detailed acceptance metrics

**Available test scripts:**
- `test_etth1_timer_vs_spec_025x_stochastic.sh`
- `test_etth2_timer_vs_spec_025x_stochastic.sh`
- `test_ettm1_timer_vs_spec_025x_stochastic.sh`
- `test_ettm2_timer_vs_spec_025x_stochastic.sh`
- `test_traffic_timer_vs_spec_025x_stochastic.sh`
- `test_weather_timer_vs_spec_025x_stochastic.sh`

**Output:** 
- Timing results in `result_inference_summary.txt`
- Detailed metrics in `result_inference_breakdown.txt`
- Per-sample acceptance in debug CSV (e.g., `spec_decoding_new_stochastic_025x.csv`)

### 5. Parameter Sweeps

**Example: Sweep spec_sigma and spec_accept_bias on ETTm1**
```bash
bash scripts/supervised/rolling_forecast/spec_decoding_tests/sweep_ettm1_spec_025x_stochastic.sh
```

**Example: Analyze sweep results**
```bash
python scripts/analysis/parse_sweep_results.py
```

**Available sweep scripts:**
- `sweep_ettm1_spec_025x_stochastic.sh`: Full parameter sweep
- `sweep_ettm2_spec_025x_stochastic.sh`: Full parameter sweep
- `stochastic_draft_param_sweep.sh`: Sweep stochastic parameters
- `tol_bias_sweep_etth1.sh`: Sweep tolerance and bias only

**Output:** Results in `sweep_results_*/` with CSV summaries

## Key Parameters

### Model Architecture
- `--model`: Model type (`timer_xl`, `timer_xl_draft`, `timer`, `ttm`, etc.)
- `--d_model`: Model dimension (default: 1024)
- `--d_ff`: Feed-forward dimension (default: 2048)
- `--n_heads`: Number of attention heads (default: 8)
- `--e_layers`: Number of encoder layers (default: 1)

### Draft Model Scaling
- `--draft_scale_d_model`: Scale factor for model dimension (e.g., 0.25)
- `--draft_scale_n_heads`: Scale factor for attention heads
- `--draft_scale_d_ff`: Scale factor for feed-forward dimension
- `--draft_scale_e_layers`: Scale factor for number of layers

### Training
- `--task_name`: Task type (`pretrain`, `forecast`)
- `--is_training`: 1 for training, 0 for testing
- `--train_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate
- `--cosine`: Use cosine learning rate scheduler

### Data
- `--root_path`: Root directory for dataset
- `--data_path`: Dataset file name
- `--data`: Dataset type (`MultivariateDatasetBenchmark`)
- `--seq_len`: Input sequence length (default: 1536)
- `--input_token_len`: Token length (default: 96)
- `--output_token_len`: Output token length
- `--test_pred_len`: Prediction horizon for testing

### Speculative Decoding
- `--use_speculative`: Enable speculative decoding
- `--spec_k`: Number of draft tokens per iteration (try 2-5)
- `--spec_sigma`: Target model variance (try 0.1-1.0)
- `--spec_accept_bias`: Acceptance bias (try 1.0-5.0)
- `--draft_stochastic`: Use stochastic draft with variance output

### Distillation
- `--distill`: Enable distillation mode
- `--distill_target_model`: Target model architecture
- `--distill_target_ckpt`: Path to target checkpoint
- `--draft_stochastic`: Enable stochastic output
- `--draft_nll_var_reg`: Variance regularization (default: 0.01)
- `--draft_teacher_noise`: Teacher noise level (default: 0.05)

### Testing & Debugging
- `--trace_inference_breakdown`: Trace detailed timing
- `--spec_debug_accept`: Debug acceptance statistics
- `--spec_debug_out`: Output CSV for acceptance details
- `--test_dir`: Directory containing checkpoint for testing
- `--test_file_name`: Checkpoint file name (default: `checkpoint.pth`)

## Results and Outputs

### Checkpoint Locations
- Pretrained models: `./checkpoints/draft_pretrain_*`
- Fine-tuned targets: `./checkpoints/forecast_*_timer_xl_*`
- Distilled drafts: `./checkpoints/forecast_*_draft_distill_*`

### Result Files
- `result_inference_summary.txt`: High-level timing and acceptance metrics
- `result_inference_breakdown.txt`: Detailed per-batch timing breakdown
- `spec_decoding_*.csv`: Per-sample acceptance debugging
- `result_long_term_forecast.txt`: Forecast accuracy results

### Key Metrics
- **MSE/MAE**: Forecast accuracy
- **Acceptance Rate**: Percentage of draft tokens accepted
- **Speedup**: Inference time reduction vs baseline
- **Per-batch Time**: Average time per batch in seconds

## Quick Start Guide

**Full workflow for speculative decoding on ETTm2:**

1. **Pretrain draft model** (if not already done):
   ```bash
   bash scripts/pretrain/timer_xl_draft_ecl_025x.sh
   ```

2. **Fine-tune target model** (if not already done):
   ```bash
   # Uncomment training section in script first
   bash scripts/supervised/rolling_forecast/timer_xl_ettm2.sh
   ```

3. **Distill draft to target**:
   ```bash
   bash scripts/distillation/updated_distillation/ettm2_timer_xl_draft_from_target_ettm2_stochastic.sh
   ```

4. **Test speculative decoding**:
   ```bash
   bash scripts/supervised/rolling_forecast/spec_decoding_tests/test_ettm2_timer_vs_spec_025x_stochastic.sh
   ```

5. **Check results**:
   ```bash
   tail -20 result_inference_summary.txt
   ```
