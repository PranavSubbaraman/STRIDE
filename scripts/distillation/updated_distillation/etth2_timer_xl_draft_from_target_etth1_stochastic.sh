#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4

# Teacher (target timer_xl) checkpoint
TEACHER_CKPT="./checkpoints/forecast_ETTh2_timer_xl_MultivariateDatasetBenchmark_sl1536_it96_ot96_lr0.0001_bt64_wd0_el1_dm1024_dff2048_nh8_cosFalse_test_0/checkpoint.pth"

model_name=timer_xl_draft
token_num=16
token_len=96
seq_len=$((token_num * token_len))

echo "[Stochastic Distillation] Train $model_name to mimic timer_xl on ETTh2 with NLL loss"
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_draft_distill_stochastic_from_target \
  --model $model_name \
  --data MultivariateDatasetBenchmark \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 336 \
  --batch_size 64 \
  --learning_rate 5e-05 \
  --train_epochs 15 \
  --d_model 1024 \
  --d_ff 2048 \
  --n_heads 8 \
  --e_layers 1 \
  --use_norm \
  --cosine \
  --tmax 10 \
  --valid_last \
  --adaptation \
  --pretrain_model_path ./checkpoints/forecast_ETTh2_draft_timer_xl_draft_MultivariateDatasetBenchmark_sl1536_it96_ot96_lr5e-05_bt64_wd0_el1_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth \
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
  --distill_target_e_layers 1 \
  --draft_stochastic \
  --draft_nll_var_reg 0.01 \
  --draft_teacher_noise 0.05


