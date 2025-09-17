export CUDA_VISIBLE_DEVICES=0

# Fine-tune timer_xl_draft on ETTh1 using your ECL pretrain checkpoint
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_draft_finetune \
  --model timer_xl_draft \
  --data MultivariateDatasetBenchmark \
  --seq_len 672 \
  --input_token_len 96 \
  --output_token_len 96 \
  --test_seq_len 672 \
  --test_pred_len 96 \
  --batch_size 32 \
  --learning_rate 5e-05 \
  --train_epochs 10 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --e_layers 2 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --valid_last \
  --adaptation \
  --pretrain_model_path /home/pranavs/speculative-time-series-project/OpenLTM/checkpoints/pretrain_draft_pretrain_ecl_07x_small_timer_xl_draft_MultivariateDatasetBenchmark_sl1536_it96_ot96_lr0.0001_bt8_wd0_el2_dm512_dff2048_nh8_cosTrue_test_0/checkpoint.pth \
  --draft_scale_d_model 0.7 \
  --draft_scale_n_heads 0.7 \
  --draft_scale_d_ff 0.7 \
  --draft_scale_e_layers 0.7

#testing the finetuned model
  python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_draft_finetune \
  --model timer_xl_draft \
  --data MultivariateDatasetBenchmark \
  --seq_len 672 --input_token_len 96 --output_token_len 96 \
  --test_seq_len 672 --test_pred_len 96 \
  --batch_size 32 \
  --d_model 512 --d_ff 2048 --n_heads 8 --e_layers 2 \
  --use_norm \
  --draft_scale_d_model 0.7 \
  --draft_scale_n_heads 0.7 \
  --draft_scale_d_ff 0.7 \
  --draft_scale_e_layers 0.7 \
  --test_dir forecast_ETTh1_draft_finetune_timer_xl_draft_MultivariateDatasetBenchmark_sl672_it96_ot96_lr5e-05_bt32_wd0_el2_dm512_dff2048_nh8_cosTrue_test_0 \
  --test_file_name checkpoint.pth