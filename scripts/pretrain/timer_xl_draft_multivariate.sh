export CUDA_VISIBLE_DEVICES=1
model_name=timer_xl_draft
token_num=32
token_len=96
seq_len=$[$token_num*$token_len]

# Use electricity dataset (present in repo). Adjust if you have another multivariate CSV.
ROOT=./dataset/electricity
FILE=electricity.csv

# starting multivariate pre-training for draft model
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path $ROOT \
  --data_path $FILE \
  --model_id multivariate_pretrain_draft \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --e_layers 4 \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --dp \
  --devices 0 \
  --draft_scale_d_model 0.5 \
  --draft_scale_n_heads 0.5 \
  --draft_scale_d_ff 0.5 \
  --draft_scale_e_layers 0.5 \
  --flash_attention


