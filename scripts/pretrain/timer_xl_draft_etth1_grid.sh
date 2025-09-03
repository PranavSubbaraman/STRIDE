export CUDA_VISIBLE_DEVICES=0
model_name=timer_xl_draft
token_num=7
token_len=96
seq_len=$[$token_num*$token_len]

for sd in 0.5 0.75; do
for se in 0.5 0.75; do
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_pretrain_draft_sd${sd}_se${se} \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --e_layers 5 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --draft_scale_d_model ${sd} \
  --draft_scale_n_heads ${sd} \
  --draft_scale_d_ff ${sd} \
  --draft_scale_e_layers ${se} \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 5 \
  --gpu 0 \
  --use_norm \
  --valid_last
done
done


