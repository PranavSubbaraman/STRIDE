#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

# Wike2000 target checkpoint (Timer-XL)
# PLEASE UPDATE AFTER TRAINING TARGET
TARGET_TEST_DIR="forecast_Wike2000_full_shot_timer_xl_UnivariateDatasetBenchmark_sl96_it96_ot96_lr5e-06_bt2048_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0"
TARGET_CKPT="checkpoint.pth"

# Wike2000 draft checkpoint (Distilled)
# PLEASE UPDATE AFTER TRAINING DRAFT
DRAFT_TEST_DIR="forecast_Wike2000_draft_distill_from_target_timer_xl_draft_UnivariateDatasetBenchmark_sl96_it96_ot96_lr5e-05_bt2048_wd0_el2_dm256_dff512_nh2_cosTrue_test_0"
DRAFT_CKPT="checkpoint.pth"

echo "[Speculative] Test Wike2000: timer_xl + timer_xl_draft (0.25x)"
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/Wike2000/ \
  --data_path Wike2000.csv \
  --model_id Wike2000_spec_test \
  --model timer_xl \
  --data UnivariateDatasetBenchmark \
  --seq_len 96 \
  --input_token_len 96 \
  --output_token_len 96 \
  --test_seq_len 96 \
  --test_pred_len 96 \
  --batch_size 2048 \
  --learning_rate 5e-6 \
  --d_model 1024 \
  --d_ff 2048 \
  --n_heads 8 \
  --e_layers 8 \
  --use_norm \
  --test_dir "$TARGET_TEST_DIR" \
  --test_file_name "$TARGET_CKPT" \
  --use_speculative \
  --spec_draft_model timer_xl_draft \
  --spec_draft_ckpt ./checkpoints/${DRAFT_TEST_DIR}/${DRAFT_CKPT} \
  --spec_k 3 \
  --spec_sigma 0.25 \
  --spec_accept_bias 1.0 \
  --draft_scale_d_model 0.25 \
  --draft_scale_n_heads 0.25 \
  --draft_scale_d_ff 0.25 \
  --draft_scale_e_layers 0.25 \
  --spec_debug_accept \
  --spec_debug_out spec_decoding_wike2000.csv \
  --trace_inference_breakdown

if [ -f result_inference_summary.txt ]; then
  echo "Acceptance (speculative):"
  grep 'acceptance_pct' result_inference_summary.txt | tail -n 1 || true
fi

