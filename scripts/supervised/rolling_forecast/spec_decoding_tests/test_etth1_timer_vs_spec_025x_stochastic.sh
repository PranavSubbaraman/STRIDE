#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4

# ETTh1 target checkpoint (Timer-XL standard config)
TARGET_TEST_DIR="forecast_ETTh1_timer_xl_MultivariateDatasetBenchmark_sl1536_it96_ot96_lr0.0001_bt64_wd0_el1_dm1024_dff2048_nh8_cosFalse_test_0"
TARGET_CKPT="checkpoint.pth"


# Stochastic draft checkpoint (will be generated after training)
DRAFT_TEST_DIR="forecast_ETTh1_draft_distill_stochastic_from_target_timer_xl_draft_MultivariateDatasetBenchmark_sl1536_it96_ot96_lr5e-05_bt64_wd0_el1_dm1024_dff2048_nh8_cosTrue_test_0"
DRAFT_CKPT="checkpoint.pth"

# echo "[1/2] Baseline test: timer_xl on ETTh1"
# python -u run.py \
#   --task_name forecast \
#   --is_training 0 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1 \
#   --model timer_xl \
#   --data MultivariateDatasetBenchmark \
#   --seq_len 1536 \
#   --input_token_len 96 \
#   --output_token_len 96 \
#   --test_seq_len 1536 \
#   --test_pred_len 336 \
#   --batch_size 64 \
#   --learning_rate 0.0001 \
#   --d_model 1024 \
#   --d_ff 2048 \
#   --n_heads 8 \
#   --e_layers 1 \
#   --use_norm \
#   --test_dir "$TARGET_TEST_DIR" \
#   --test_file_name "$TARGET_CKPT" \
#   --trace_inference_breakdown

echo "[2/2] Speculative test with stochastic 0.25x draft: timer_xl + timer_xl_draft (NLL-trained)"
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_stochastic \
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
  --test_dir "$TARGET_TEST_DIR" \
  --test_file_name "$TARGET_CKPT" \
  --use_speculative \
  --spec_draft_model timer_xl_draft \
  --spec_draft_ckpt ./checkpoints/${DRAFT_TEST_DIR}/${DRAFT_CKPT} \
  --spec_k 3 \
  --spec_sigma 0.3 \
  --spec_accept_bias 3 \
  --draft_scale_d_model 0.25 \
  --draft_scale_n_heads 0.25 \
  --draft_scale_d_ff 0.25 \
  --draft_scale_e_layers 0.25 \
  --draft_stochastic \
  --spec_debug_accept \
  --spec_debug_out spec_decoding_stochastic_025x.csv \
  --trace_inference_breakdown \

# Print acceptance rate from the latest speculative run
if [ -f result_inference_summary.txt ]; then
  grep 'acceptance_pct' result_inference_summary.txt | tail -n 1 || true
fi


