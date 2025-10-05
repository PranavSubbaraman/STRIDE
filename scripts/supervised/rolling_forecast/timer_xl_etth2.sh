export CUDA_VISIBLE_DEVICES=4
model_name=timer_xl
token_num=16
token_len=96
seq_len=$[$token_num*$token_len]
# training one model with a context length

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2 \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 336 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --d_model 1024 \
  --d_ff 2048 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 1 \
  --valid_last \

# testing the model on all forecast lengths
for test_pred_len in 336
do
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2 \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len $test_pred_len \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --d_model 1024 \
  --d_ff 2048 \
  --n_heads 8 \
  --lradj type1 \
  --use_norm \
  --e_layers 1 \
  --valid_last \
  --test_dir forecast_ETTh2_timer_xl_MultivariateDatasetBenchmark_sl1536_it96_ot96_lr0.0001_bt64_wd0_el1_dm1024_dff2048_nh8_cosFalse_test_0 \
  --trace_inference_breakdown

  # print timing for this test
  if [ -f result_inference_summary.txt ]; then
    echo "Timing (summary):"
    grep 'per_batch_s' result_inference_summary.txt | tail -n 1 || true
  fi
done
