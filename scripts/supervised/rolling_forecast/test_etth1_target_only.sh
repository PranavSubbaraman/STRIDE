#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# ETTh1 target checkpoint (Timer-XL large)
TARGET_TEST_DIR="forecast_ETTh1_target_finetune_timer_xl_MultivariateDatasetBenchmark_sl672_it96_ot96_lr5e-05_bt4_wd0_el8_dm1024_dff4096_nh16_cosTrue_test_0"
TARGET_CKPT="checkpoint.pth"

echo "[Target only] timer_xl on ETTh1 (no speculative decoding)"
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_target_only \
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

# Print timing summaries
if [ -f result_inference_summary.txt ]; then
  echo "Timing (summary):"
  grep 'per_batch_s' result_inference_summary.txt | tail -n 1 || true
fi
if [ -f result_inference_breakdown.txt ]; then
  echo "Timing (baseline_forward total & calls):"
  grep -A0 't_baseline_forward_total' result_inference_breakdown.txt | tail -n 1 || true
fi


