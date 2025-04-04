export CUDA_VISIBLE_DEVICES=0,1,2,3
model_name=autotimes
token_num=7
token_len=96
seq_len=$[$token_num*$token_len]

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ecl \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --load_time_stamp \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --batch_size 256 \
  --learning_rate 0.0005 \
  --train_epochs 10 \
  --d_model 256 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 2 \
  --valid_last \
  --mix_embeds \
  --dp \
  --devices 0,1,2,3