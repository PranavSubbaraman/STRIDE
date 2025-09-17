#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/pranavs/speculative-time-series-project/OpenLTM"
cd "$ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate open-ltm

TARGET_TEST_DIR="forecast_ETTh1_large_target_e8d1024_timer_xl_MultivariateDatasetBenchmark_sl672_it96_ot96_lr0.0001_bt4_wd0_el8_dm1024_dff4096_nh16_cosTrue_test_0"
TARGET_CKPT="checkpoint.pth"
DRAFT_CKPT="$ROOT/checkpoints/forecast_ETTh1_draft_finetuned_timer_xl_draft_MultivariateDatasetBenchmark_sl672_it96_ot96_lr5e-05_bt16_wd0_el2_dm512_dff2048_nh8_cosFalse_test_0/checkpoint.pth"

COMMON_ARGS=(
  --task_name forecast
  --is_training 0
  --root_path "$ROOT/dataset/ETT-small/"
  --data_path ETTh1.csv
  --model timer_xl
  --data MultivariateDatasetBenchmark
  --seq_len 672
  --input_token_len 96
  --output_token_len 96
  --test_seq_len 672
  --test_pred_len 192
  --batch_size 4
  --d_model 1024
  --d_ff 4096
  --n_heads 16
  --e_layers 8
  --use_norm
  --gpu 0
  --test_dir "$TARGET_TEST_DIR"
  --test_file_name "$TARGET_CKPT"
  --use_speculative
  --spec_draft_model timer_xl_draft
  --spec_draft_ckpt "$DRAFT_CKPT"
  --draft_scale_d_model 0.328125
  --draft_scale_n_heads 0.375
  --draft_scale_d_ff 0.35
  --draft_scale_e_layers 0.25
  --spec_k 3
  --spec_sigma_mode adaptive
  --spec_sigma_adapt_c 4
)

echo "=== TOLBIAS SWEEP START $(date) ===" >> result_inference_summary.txt

for TOL in 0.15 0.20 0.25; do
  for BIAS in 1.5 2.0; do
    MODEL_ID="ETTh1_specK3_adapt_c4_tol${TOL//./p}_bias${BIAS//./p}"
    echo "[RUN] $MODEL_ID" | tee -a result_inference_summary.txt
    CUDA_VISIBLE_DEVICES=0 python -u run.py "${COMMON_ARGS[@]}" \
      --model_id "$MODEL_ID" \
      --spec_accept_mse_tol "$TOL" \
      --spec_accept_bias "$BIAS" \
      --des "tol_${TOL}_bias_${BIAS}" | cat
  done
done

echo "=== TOLBIAS SWEEP END $(date) ===" >> result_inference_summary.txt


