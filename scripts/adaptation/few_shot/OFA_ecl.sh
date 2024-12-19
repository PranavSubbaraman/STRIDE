model_name=OFA
token_len=24
seq_len=96
pred_len=24
# a smaller batch size chosen due to large memory usage

python run.py \
    --task_name forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL \
    --model OFA \
    --data MultivariateDatasetBenchmark  \
    --seq_len $seq_len \
    --input_token_len $token_len \
    --output_token_len $token_len \
    --test_seq_len $seq_len \
    --test_pred_len $pred_len \
    --e_layers 5 \
    --d_model 768 \
    --d_ff 768 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --cosine \
    --train_epochs 10 \
    --gpu 0 \
    --gpt_layers 6 \
    --is_gpt 1 \
    --kernel_size 25\
    --patch_size 16 \
    --stride 8 \
    --pretrain 1 \
    --freeze 1 \
    --max_len -1 \
    --hid_dim 16 \
    --tmax 10 \
    --n_scale -1 \
    --valid_last \
    --nonautoregressive \
    --subset_rand_ratio 0.1