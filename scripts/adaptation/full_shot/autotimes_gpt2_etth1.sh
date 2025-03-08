# CUDA_VISiBLE_DEVICES = 4, 5, 6, 7
export CUDA_VISIBLE_DEVICES=5
model_name=autotimes

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path /data/zhiyuan/dataset/ETT-small\
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model $model_name \
  --model_name GPT2 \
  --data ETTh \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 1 \
  --learning_rate 0.002 \
  --itr 1 \
  --train_epochs 10 \
  --use_amp \
  --llm_ckp_dir /data/zhiyuan/checkpoints/GPT2 \
  --gpu 0 \
  --des 'Gpt2' \
  --cosine \
  --tmax 10 \
  --mlp_hidden_dim 512 \
