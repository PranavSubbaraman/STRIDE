#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

# ETTh1 target checkpoint (Timer-XL, e_layers=1, n_heads=8, d_ff=2048)
TARGET_TEST_DIR="forecast_ETTh1_timer_xl_MultivariateDatasetBenchmark_sl672_it96_ot96_lr0.0001_bt32_wd0_el1_dm1024_dff2048_nh8_cosFalse_test_0"
TARGET_CKPT="checkpoint.pth"

# TTM draft checkpoint (fine-tuned from ECL pretraining)
TTM_DRAFT_CKPT="./checkpoints/forecast_ETTh1_draft_ttm_MultivariateDatasetBenchmark_sl1536_it1536_ot96_lr5e-05_bt32_wd0_el3_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth"

echo "[1/3] Baseline test: timer_xl (target) on ETTh1"
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model timer_xl \
  --data MultivariateDatasetBenchmark \
  --seq_len 672 \
  --input_token_len 96 \
  --output_token_len 96 \
  --test_seq_len 672 \
  --test_pred_len 96 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --d_model 1024 \
  --d_ff 2048 \
  --n_heads 8 \
  --e_layers 1 \
  --use_norm \
  --test_dir "$TARGET_TEST_DIR" \
  --test_file_name "$TARGET_CKPT"

# Print baseline per-batch timing
if [ -f result_inference_summary.txt ]; then
  echo "Baseline (Timer-XL) per_batch_s:"
  grep 'per_batch_s' result_inference_summary.txt | tail -n 1 || true
fi

echo ""
echo "=== NOTE: Cross-architecture speculation (Timer-XL + TTM) with different context lengths ==="
echo "requires enhanced context management in SpeculativeDecoder. Skipping TTM-based tests for now."
echo ""
exit 0

# DISABLED: Cross-architecture speculation needs context length adaptation
echo ""
echo "[2/3] Draft-only test: TTM (draft) on ETTh1"
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_draft \
  --model ttm \
  --data MultivariateDatasetBenchmark \
  --seq_len 1536 \
  --input_token_len 1536 \
  --output_token_len 96 \
  --test_seq_len 1536 \
  --test_pred_len 96 \
  --batch_size 32 \
  --d_model 1024 \
  --d_ff 2048 \
  --dropout 0.2 \
  --use_norm \
  --e_layers 3 \
  --layers 2 \
  --hidden_dim 64 \
  --AP_levels 3 \
  --mode common_channel \
  --d_mode common_channel \
  --patch_size 96 \
  --stride 96 \
  --use_decoder \
  --n_vars 7 \
  --test_dir forecast_ETTh1_draft_ttm_MultivariateDatasetBenchmark_sl1536_it1536_ot96_lr5e-05_bt32_wd0_el3_dm1024_dff2048_nh8_cosTrue_test_0 \
  --nonautoregressive

# Print draft-only per-batch timing
if [ -f result_inference_summary.txt ]; then
  echo "Draft-only (TTM) per_batch_s:"
  grep 'per_batch_s' result_inference_summary.txt | tail -n 1 || true
fi

echo ""
echo "[3/3] Speculative test: timer_xl (target) + ttm (draft) on ETTh1"
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_spec \
  --model timer_xl \
  --data MultivariateDatasetBenchmark \
  --seq_len 672 \
  --input_token_len 96 \
  --output_token_len 96 \
  --test_seq_len 672 \
  --test_pred_len 96 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --d_model 1024 \
  --d_ff 2048 \
  --n_heads 8 \
  --e_layers 1 \
  --use_norm \
  --test_dir "$TARGET_TEST_DIR" \
  --test_file_name "$TARGET_CKPT" \
  --use_speculative \
  --spec_draft_model ttm \
  --spec_draft_ckpt "$TTM_DRAFT_CKPT" \
  --spec_draft_seq_len 1536 \
  --spec_k 3 \
  --spec_sigma 0.5 \
  --spec_accept_bias 1.0 \
  --spec_sigma_mode adaptive \
  --spec_sigma_adapt_c 4 \
  --spec_accept_mse_tol -1.0 \
  --spec_debug_accept \
  --spec_debug_out spec_accept_debug_timer_ttm.csv \
  --spec_debug_n 3 \
  --spec_debug_max_batches 3 \
  --spec_debug_max_rounds 4 \
  --layers 2 \
  --hidden_dim 64 \
  --AP_levels 3 \
  --mode common_channel \
  --d_mode common_channel \
  --patch_size 96 \
  --stride 96 \
  --use_decoder

# Print speculative per-batch timing
if [ -f result_inference_summary.txt ]; then
  echo "Speculative (Timer-XL + TTM) per_batch_s:"
  grep 'per_batch_s' result_inference_summary.txt | tail -n 1 || true
fi

# Print acceptance rate from the latest speculative run
if [ -f result_inference_summary.txt ]; then
  echo "Acceptance rate:"
  grep 'acceptance_pct' result_inference_summary.txt | tail -n 1 || true
fi

echo ""
echo "=== Summary ==="
echo "Compare the three per_batch_s values above:"
echo "  1. Baseline (Timer-XL alone)"
echo "  2. Draft-only (TTM alone)"
echo "  3. Speculative (Timer-XL + TTM)"
echo ""
echo "Check spec_accept_debug_timer_ttm.csv for detailed acceptance diagnostics"
