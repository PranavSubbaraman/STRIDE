#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# ETTh1 draft checkpoint (no speculation)
DRAFT_TEST_DIR="forecast_ETTh1_draft_finetune_timer_xl_draft_MultivariateDatasetBenchmark_sl672_it96_ot96_lr5e-05_bt32_wd0_el2_dm512_dff2048_nh8_cosTrue_test_0"
DRAFT_CKPT="checkpoint.pth"

echo "[Draft only] timer_xl_draft on ETTh1 (no speculative decoding)"
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_draft_only \
  --model timer_xl_draft \
  --data MultivariateDatasetBenchmark \
  --seq_len 672 \
  --input_token_len 96 \
  --output_token_len 96 \
  --test_seq_len 672 \
  --test_pred_len 96 \
  --batch_size 32 \
  --learning_rate 5e-05 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --e_layers 2 \
  --use_norm \
  --cosine \
  --test_dir "$DRAFT_TEST_DIR" \
  --test_file_name "$DRAFT_CKPT" \
  --draft_scale_d_model 0.7 \
  --draft_scale_n_heads 0.7 \
  --draft_scale_d_ff 0.7 \
  --draft_scale_e_layers 0.7 \
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


