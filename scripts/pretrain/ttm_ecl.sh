model_name=ttm
token_num=16        # context_length = 16 * 96 = 1536
token_len=96        # patch_length
seq_len=$((token_num * token_len))

python -u run.py \
  --task_name pretrain \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ttm_pretrain_ecl \
  --model $model_name \
  --data MultivariateDatasetBenchmark \
  --n_vars 321 \
  --seq_len $seq_len \
  --input_token_len $seq_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len $token_len \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --d_model 1024 \
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
  --nonautoregressive \
  --valid_last \
  --cosine


