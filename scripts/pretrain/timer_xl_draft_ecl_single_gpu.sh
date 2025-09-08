export CUDA_VISIBLE_DEVICES=0
model_name=timer_xl_draft
token_num=32
token_len=96
seq_len=$[$token_num*$token_len]

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id draft_pretrain_ecl \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --e_layers 4 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 2048 \
  --draft_scale_d_model 0.5 \
  --draft_scale_n_heads 0.5 \
  --draft_scale_d_ff 0.5 \
  --draft_scale_e_layers 0.5 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 20 \
  --gpu 0 \
  --cosine \
  --tmax 20 \
  --use_norm


