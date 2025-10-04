#!/usr/bin/env bash
set -euo pipefail

# Activate env and go to repo
conda activate open-ltm
cd /home/pranavs/speculative-time-series-project/OpenLTM
export CUDA_VISIBLE_DEVICES=0

# Teacher (target timer_xl) checkpoint
TEACHER_CKPT="./checkpoints/forecast_ETTh1_large_target_e8d1024_timer_xl_MultivariateDatasetBenchmark_sl672_it96_ot96_lr0.0001_bt4_wd0_el8_dm1024_dff4096_nh16_cosTrue_test_0/checkpoint.pth"

model_name=timer_xl_draft
token_num=7
token_len=96
seq_len=$((token_num * token_len))

echo "[Distillation 0.5x] Train $model_name (0.5x of target) to mimic timer_xl on ETTh1"
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_draft_distill0p5 \
  --model $model_name \
  --data MultivariateDatasetBenchmark \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --batch_size 16 \
  --learning_rate 5e-05 \
  --train_epochs 10 \
  --d_model 1024 \
  --d_ff 4096 \
  --n_heads 16 \
  --e_layers 8 \
  --use_norm \
  --cosine \
  --tmax 10 \
  --valid_last \
  --draft_scale_d_model 0.5 \
  --draft_scale_n_heads 0.5 \
  --draft_scale_d_ff 0.5 \
  --draft_scale_e_layers 0.5 \
  --distill \
  --distill_target_model timer_xl \
  --distill_target_ckpt "$TEACHER_CKPT" \
  --distill_target_d_model 1024 \
  --distill_target_n_heads 16 \
  --distill_target_d_ff 4096 \
  --distill_target_e_layers 8


