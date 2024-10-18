export CUDA_VISIBLE_DEVICES=2
model_name=timer_lite
token_num=7
token_len=96
seq_len=$[$token_num*$token_len]
# training one model with a context length
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model $model_name \
  --data ETTh1_Multi  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --d_model 512 \
  --d_ff 2048 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 1 \
  --d_layers 1 \
  --valid_last