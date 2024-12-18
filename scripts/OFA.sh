model_name=OFA
token_num=7
token_len=24
seq_len=$[$token_num*$token_len]

python run.py \
    --task_name forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL \
    --model $model_name \
    --data MultivariateDatasetBenchmark  \
    --seq_len $seq_len \
    --input_token_len $token_len \
    --output_token_len $token_len \
    --test_seq_len $seq_len \
    --test_pred_len 24 \
    --e_layers 5 \
    --d_model 768 \
    --d_ff 128 \
    --batch_size 4 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --gpu 0 \
    --gpt_layers 6 \
    --is_gpt 1 \
    --kernel_size 25\
    --patch_size 1 \
    --pretrain 1 \
    --freeze 1 \
    --max_len -1 \
    --hid_dim 16 \
    --tmax 20 \
    --n_scale -1 \
    --valid_last \
    --nonautoregressive

python run.py \
    --task_name forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL \
    --model OFA \
    --data MultivariateDatasetBenchmark  \
    --seq_len 24 \
    --input_token_len 24 \
    --output_token_len 24 \
    --test_seq_len 24 \
    --test_pred_len 24 \
    --e_layers 5 \
    --d_model 768 \
    --d_ff 128 \
    --batch_size 4 \
    --learning_rate 0.001 \
    --train_epochs 1 \
    --gpu 0 \
    --gpt_layers 6 \
    --is_gpt 1 \
    --kernel_size 25\
    --patch_size 16 \
    --pretrain 1 \
    --freeze 1 \
    --max_len -1 \
    --hid_dim 16 \
    --tmax 20 \
    --n_scale -1 \
    --valid_last \
    --nonautoregressive