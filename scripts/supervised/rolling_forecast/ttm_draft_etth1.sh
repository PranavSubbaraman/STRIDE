export CUDA_VISIBLE_DEVICES=0
model_name=ttm
token_num=16        # Match pretraining: 16 patches for transfer learning
token_len=96
seq_len=$[$token_num*$token_len]

# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_draft \
#   --model $model_name \
#   --data MultivariateDatasetBenchmark \
#   --seq_len $seq_len \
#   --input_token_len $seq_len \
#   --output_token_len $token_len \
#   --test_seq_len $seq_len \
#   --test_pred_len 96 \
#   --batch_size 32 \
#   --learning_rate 5e-05 \
#   --train_epochs 10 \
#   --d_model 1024 \
#   --d_ff 2048 \
#   --dropout 0.2 \
#   --use_norm \
#   --e_layers 3 \
#   --layers 2 \
#   --hidden_dim 64 \
#   --AP_levels 3 \
#   --mode common_channel \
#   --d_mode common_channel \
#   --patch_size $token_len \
#   --stride $token_len \
#   --use_decoder \
#   --n_vars 7 \
#   --cosine \
#   --tmax 10 \
#   --valid_last \
#   --adaptation \
#   --pretrain_model_path ./checkpoints/pretrain_ttm_pretrain_ecl_ttm_MultivariateDatasetBenchmark_sl1536_it1536_ot96_lr0.0001_bt4_wd0_el3_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth \
#   --nonautoregressive

# Testing the model
for test_pred_len in 96
do
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_draft \
  --model $model_name \
  --data MultivariateDatasetBenchmark \
  --seq_len $seq_len \
  --input_token_len $seq_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len $test_pred_len \
  --batch_size 32 \
  --d_model 1024 \
  --d_ff 2048 \
  --dropout 0.2 \
  --use_norm \
  --e_layers 3 \
  --layers 2 \
  --hidden_dim 64 \
  --AP_levels 3 \
  --mode common_channel \
  --d_mode common_channel \
  --patch_size $token_len \
  --stride $token_len \
  --use_decoder \
  --n_vars 7 \
  --test_dir forecast_ETTh1_draft_ttm_MultivariateDatasetBenchmark_sl1536_it1536_ot96_lr5e-05_bt32_wd0_el3_dm1024_dff2048_nh8_cosTrue_test_0 \
  --nonautoregressive

  # print timing for this test
  if [ -f result_inference_summary.txt ]; then
    echo "Timing (summary):"
    grep 'per_batch_s' result_inference_summary.txt | tail -n 1 || true
  fi
done
