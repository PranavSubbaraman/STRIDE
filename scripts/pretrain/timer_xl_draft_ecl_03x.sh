export CUDA_VISIBLE_DEVICES=6

model_name=timer_xl_draft
token_num=16        # because seq_len=1536 and token_len=96 → 16*96 = 1536
token_len=96
seq_len=$((token_num * token_len))

python -u run.py \
  --task_name pretrain \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id draft_pretrain_ecl_03x_small \
  --model $model_name \
  --data MultivariateDatasetBenchmark \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --e_layers 2 \
  --cosine \
  --draft_scale_d_model 0.3 \
  --draft_scale_n_heads 0.3 \
  --draft_scale_d_ff 0.3 \
  --draft_scale_e_layers 0.3