#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

# ETTh1 target checkpoint (Timer-XL, e_layers=1, n_heads=8, d_ff=2048)
TARGET_TEST_DIR="forecast_ETTh1_timer_xl_MultivariateDatasetBenchmark_sl672_it96_ot96_lr0.0001_bt32_wd0_el1_dm1024_dff2048_nh8_cosFalse_test_0"
TARGET_CKPT="checkpoint.pth"

# echo "[1/2] Baseline test: timer_xl on ETTh1"
# python -u run.py \
#   --task_name forecast \
#   --is_training 0 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1 \
#   --model timer_xl \
#   --data MultivariateDatasetBenchmark \
#   --seq_len 672 \
#   --input_token_len 96 \
#   --output_token_len 96 \
#   --test_seq_len 672 \
#   --test_pred_len 336 \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --d_model 1024 \
#   --d_ff 2048 \
#   --n_heads 8 \
#   --e_layers 1 \
#   --use_norm \
#   --cosine \
#   --test_dir "$TARGET_TEST_DIR" \
#   --test_file_name "$TARGET_CKPT" \
#   --trace_inference_breakdown

# # Print baseline per-batch timing
# if [ -f result_inference_summary.txt ]; then
#   echo "Baseline per_batch_s:"
#   grep 'per_batch_s' result_inference_summary.txt | tail -n 1 || true
# fi

echo "[2/2] Speculative test: timer_xl + timer_xl_draft on ETTh1"
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model timer_xl \
  --data MultivariateDatasetBenchmark \
  --seq_len 1536 \
  --input_token_len 96 \
  --output_token_len 96 \
  --test_seq_len 1536 \
  --test_pred_len 336 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --d_model 1024 \
  --d_ff 2048 \
  --n_heads 8 \
  --e_layers 1 \
  --use_norm \
  --cosine \
  --test_dir "$TARGET_TEST_DIR" \
  --test_file_name "$TARGET_CKPT" \
  --use_speculative \
  --spec_draft_model timer_xl_draft \
  --spec_draft_ckpt ./checkpoints/pretrain_draft_pretrain_ecl_05x_small_timer_xl_draft_MultivariateDatasetBenchmark_sl1536_it96_ot96_lr0.0001_bt32_wd0_el1_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth \
  --spec_k 3 \
  --spec_sigma 0.5 \
  --spec_accept_bias 1.0 \
  --draft_scale_d_model 0.5 \
  --draft_scale_n_heads 0.5 \
  --draft_scale_d_ff 0.5 \
  --draft_scale_e_layers 0.5 \
  --trace_inference_breakdown \
  --spec_sigma_mode adaptive \
  --spec_sigma_adapt_c 4 \
  --spec_accept_mse_tol -1.0 \
  --spec_debug_accept \
  --spec_debug_out spec_accept_debug_NEW_STATS.csv \
  --spec_debug_n 3 \
  --spec_debug_max_batches 3 \
  --spec_debug_max_rounds 4

# Print speculative per-batch timing
if [ -f result_inference_summary.txt ]; then
  echo "Speculative per_batch_s:"
  grep 'per_batch_s' result_inference_summary.txt | tail -n 1 || true
fi

# Print acceptance rate from the latest speculative run
if [ -f result_inference_summary.txt ]; then
  echo "Acceptance (speculative):"
  grep 'acceptance_pct' result_inference_summary.txt | tail -n 1 || true
fi