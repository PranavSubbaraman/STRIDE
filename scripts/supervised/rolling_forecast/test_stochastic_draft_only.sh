#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4

# Stochastic draft checkpoint (already trained)
DRAFT_TEST_DIR="forecast_ETTh1_draft_distill_stochastic_from_target_timer_xl_draft_MultivariateDatasetBenchmark_sl1536_it96_ot96_lr5e-05_bt64_wd0_el1_dm1024_dff2048_nh8_cosTrue_test_0"
DRAFT_CKPT="checkpoint.pth"

echo "[1/1] Test stochastic draft model standalone (without speculative decoding)"
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_stochastic_draft_only \
  --model timer_xl_draft \
  --data MultivariateDatasetBenchmark \
  --seq_len 1536 \
  --input_token_len 96 \
  --output_token_len 96 \
  --test_seq_len 1536 \
  --test_pred_len 336 \
  --batch_size 64 \
  --learning_rate 5e-05 \
  --d_model 1024 \
  --d_ff 2048 \
  --n_heads 8 \
  --e_layers 1 \
  --use_norm \
  --draft_scale_d_model 0.25 \
  --draft_scale_n_heads 0.25 \
  --draft_scale_d_ff 0.25 \
  --draft_scale_e_layers 0.25 \
  --draft_stochastic \
  --test_dir "$DRAFT_TEST_DIR" \
  --test_file_name "$DRAFT_CKPT"

echo "Test completed! Check result_inference_summary.txt for metrics."
