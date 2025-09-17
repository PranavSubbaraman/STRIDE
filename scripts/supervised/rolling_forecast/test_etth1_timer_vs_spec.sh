#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

# ETTh1 target checkpoint (Timer-XL large)
TARGET_TEST_DIR="forecast_ETTh1_large_target_e8d1024_timer_xl_MultivariateDatasetBenchmark_sl672_it96_ot96_lr0.0001_bt4_wd0_el8_dm1024_dff4096_nh16_cosTrue_test_0"
TARGET_CKPT="checkpoint.pth"

echo "[1/2] Baseline test: timer_xl on ETTh1"
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
  --batch_size 4 \
  --learning_rate 0.0001 \
  --d_model 1024 \
  --d_ff 4096 \
  --n_heads 16 \
  --e_layers 8 \
  --use_norm \
  --cosine \
  --test_dir "$TARGET_TEST_DIR" \
  --test_file_name "$TARGET_CKPT" \
  --trace_inference_breakdown

echo "[2/2] Speculative test: timer_xl + timer_xl_draft on ETTh1"
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
  --batch_size 4 \
  --learning_rate 0.0001 \
  --d_model 1024 \
  --d_ff 4096 \
  --n_heads 16 \
  --e_layers 8 \
  --use_norm \
  --cosine \
  --test_dir "$TARGET_TEST_DIR" \
  --test_file_name "$TARGET_CKPT" \
  --use_speculative \
  --spec_draft_model timer_xl_draft \
  --spec_draft_ckpt ./checkpoints/forecast_ETTh1_draft_finetune_timer_xl_draft_MultivariateDatasetBenchmark_sl672_it96_ot96_lr5e-05_bt32_wd0_el2_dm512_dff2048_nh8_cosTrue_test_0/checkpoint.pth \
  --spec_k 3 \
  --spec_sigma 0.01 \
  --spec_accept_bias 1.0 \
  --draft_scale_d_model 0.328125 \
  --draft_scale_n_heads 0.375 \
  --draft_scale_d_ff 0.35 \
  --draft_scale_e_layers 0.25 \
  --trace_inference_breakdown \
  --spec_sigma_mode adaptive \
  --spec_sigma_adapt_c 4 \
  --spec_accept_mse_tol 0.25 \
  --spec_debug_accept \
  --spec_debug_out spec_accept_debug_NEW.csv \
  --spec_debug_n 3 \
  --spec_debug_max_batches 3 \
  --spec_debug_max_rounds 4

# Print acceptance rate from the latest speculative run
if [ -f result_inference_summary.txt ]; then
  echo "Acceptance (speculative):"
  grep 'acceptance_pct' result_inference_summary.txt | tail -n 1 || true
fi