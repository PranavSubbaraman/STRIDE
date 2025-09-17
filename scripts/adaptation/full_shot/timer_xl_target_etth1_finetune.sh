#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

model_name=timer_xl
token_num=7
token_len=96
seq_len=$((token_num * token_len))

# Pretrained target checkpoint to adapt from (update if needed)
PRETRAIN_CKPT="./checkpoints/forecast_ETTh1_large_target_e8d1024_timer_xl_MultivariateDatasetBenchmark_sl672_it96_ot96_lr0.0001_bt4_wd0_el8_dm1024_dff4096_nh16_cosTrue_test_0/checkpoint.pth"

echo "[1/2] Finetuning target timer_xl on ETTh1"
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_target_finetune \
  --model $model_name \
  --data MultivariateDatasetBenchmark \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --batch_size 4 \
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
  --adaptation \
  --pretrain_model_path "$PRETRAIN_CKPT"

#testing the finetuned model
echo "[2/2] Evaluating finetuned target model"
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_target_finetune \
  --model $model_name \
  --data MultivariateDatasetBenchmark \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --batch_size 4 \
  --learning_rate 5e-05 \
  --d_model 1024 \
  --d_ff 4096 \
  --n_heads 16 \
  --e_layers 8 \
  --use_norm \
  --cosine \
  --test_dir forecast_ETTh1_target_finetune_timer_xl_MultivariateDatasetBenchmark_sl672_it96_ot96_lr5e-05_bt4_wd0_el8_dm1024_dff4096_nh16_cosTrue_test_0 \
  --test_file_name checkpoint.pth \
  --trace_inference_breakdown


