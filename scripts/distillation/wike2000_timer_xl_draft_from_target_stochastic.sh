#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# Teacher (target timer_xl) checkpoint - Needs to be trained first!
# Assuming the user will train the target model using STRIDE/scripts/adaptation/full_shot/timer_xl_wike2000.sh
# and the checkpoint will be located in a path similar to below.
# PLEASE UPDATE TEACHER_CKPT after training the target model.
TEACHER_CKPT="./checkpoints/forecast_Wike2000_full_shot_timer_xl_UnivariateDatasetBenchmark_sl96_it96_ot96_lr5e-06_bt2048_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth"

model_name=timer_xl_draft
token_num=1
token_len=96
seq_len=$((token_num * token_len))

echo "[Distillation] Train $model_name to mimic timer_xl on Wike2000"
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/Wike2000/ \
  --data_path Wike2000.csv \
  --model_id Wike2000_draft_distill_from_target \
  --model $model_name \
  --data UnivariateDatasetBenchmark \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --batch_size 2048 \
  --learning_rate 5e-05 \
  --train_epochs 15 \
  --d_model 1024 \
  --d_ff 2048 \
  --n_heads 8 \
  --e_layers 8 \
  --use_norm \
  --cosine \
  --tmax 10 \
  --adaptation \
  --pretrain_model_path checkpoints/timer_xl/checkpoint.pth \
  --draft_scale_d_model 0.25 \
  --draft_scale_n_heads 0.25 \
  --draft_scale_d_ff 0.25 \
  --draft_scale_e_layers 0.25 \
  --distill \
  --distill_target_model timer_xl \
  --distill_target_ckpt "$TEACHER_CKPT" \
  --distill_target_d_model 1024 \
  --distill_target_n_heads 8 \
  --distill_target_d_ff 2048 \
  --distill_target_e_layers 8

