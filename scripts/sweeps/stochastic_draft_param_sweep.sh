#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Stochastic Draft Parameter Sweep
# ============================================================================
# Sweeps over spec_sigma, spec_accept_bias, and spec_k to find optimal
# parameters for speculative decoding with learned variance.
# ============================================================================

export CUDA_VISIBLE_DEVICES=4

# Configuration
TARGET_TEST_DIR="forecast_ETTh1_timer_xl_MultivariateDatasetBenchmark_sl1536_it96_ot96_lr0.0001_bt64_wd0_el1_dm1024_dff2048_nh8_cosFalse_test_0"
TARGET_CKPT="checkpoint.pth"
DRAFT_TEST_DIR="forecast_ETTh1_draft_distill_stochastic_from_target_timer_xl_draft_MultivariateDatasetBenchmark_sl1536_it96_ot96_lr5e-05_bt64_wd0_el1_dm1024_dff2048_nh8_cosTrue_test_0"
DRAFT_CKPT="checkpoint.pth"

# Output file
RESULTS_CSV="stochastic_sweep_results.csv"
LOG_FILE="stochastic_sweep.log"

# Create CSV header
echo "sigma,bias,k,mse,mae,acceptance_pct,accepted,attempted,total_time_s" > "$RESULTS_CSV"
echo "Parameter sweep started at $(date)" | tee -a "$LOG_FILE"

# Common arguments
COMMON_ARGS=(
  --task_name forecast
  --is_training 0
  --root_path ./dataset/ETT-small/
  --data_path ETTh1.csv
  --model timer_xl
  --data MultivariateDatasetBenchmark
  --seq_len 1536
  --input_token_len 96
  --output_token_len 96
  --test_seq_len 1536
  --test_pred_len 336
  --batch_size 64
  --learning_rate 0.0001
  --d_model 1024
  --d_ff 2048
  --n_heads 8
  --e_layers 1
  --use_norm
  --test_dir "$TARGET_TEST_DIR"
  --test_file_name "$TARGET_CKPT"
  --use_speculative
  --spec_draft_model timer_xl_draft
  --spec_draft_ckpt ./checkpoints/${DRAFT_TEST_DIR}/${DRAFT_CKPT}
  --draft_scale_d_model 0.25
  --draft_scale_n_heads 0.25
  --draft_scale_d_ff 0.25
  --draft_scale_e_layers 0.25
  --draft_stochastic
)

# Parameter ranges
SIGMA_VALUES=(0.1 0.25 0.5 1.0)
BIAS_VALUES=(1.0 2.0 5.0 10.0)
K_VALUES=(1 2 3 4 5)

# Total configurations
TOTAL=$((${#SIGMA_VALUES[@]} * ${#BIAS_VALUES[@]} * ${#K_VALUES[@]}))
COUNTER=0

echo "Total configurations to test: $TOTAL" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Sweep loop
for SIGMA in "${SIGMA_VALUES[@]}"; do
  for BIAS in "${BIAS_VALUES[@]}"; do
    for K in "${K_VALUES[@]}"; do
      COUNTER=$((COUNTER + 1))
      
      MODEL_ID="ETTh1_stoch_s${SIGMA}_b${BIAS}_k${K}"
      echo "" | tee -a "$LOG_FILE"
      echo "[$COUNTER/$TOTAL] Testing sigma=$SIGMA, bias=$BIAS, k=$K" | tee -a "$LOG_FILE"
      echo "Model ID: $MODEL_ID" | tee -a "$LOG_FILE"
      
      START_TIME=$(date +%s)
      
      # Run test and capture output
      OUTPUT=$(python -u run.py \
        "${COMMON_ARGS[@]}" \
        --model_id "$MODEL_ID" \
        --spec_sigma "$SIGMA" \
        --spec_accept_bias "$BIAS" \
        --spec_k "$K" \
        2>&1)
      
      END_TIME=$(date +%s)
      ELAPSED=$((END_TIME - START_TIME))
      
      # Extract metrics from output
      MSE=$(echo "$OUTPUT" | grep -oP 'mse:\K[0-9.]+' | tail -1 || echo "NA")
      MAE=$(echo "$OUTPUT" | grep -oP 'mae:\K[0-9.]+' | tail -1 || echo "NA")
      ACCEPTED=$(echo "$OUTPUT" | grep -oP 'accepted:\K[0-9]+' | tail -1 || echo "0")
      ATTEMPTED=$(echo "$OUTPUT" | grep -oP 'attempted:\K[0-9]+' | tail -1 || echo "0")
      ACCEPTANCE=$(echo "$OUTPUT" | grep -oP 'acceptance_pct:\K[0-9.]+' | tail -1 || echo "0.00")
      
      # Log results
      echo "  MSE: $MSE, MAE: $MAE" | tee -a "$LOG_FILE"
      echo "  Acceptance: $ACCEPTANCE% ($ACCEPTED/$ATTEMPTED)" | tee -a "$LOG_FILE"
      echo "  Time: ${ELAPSED}s" | tee -a "$LOG_FILE"
      
      # Append to CSV
      echo "$SIGMA,$BIAS,$K,$MSE,$MAE,$ACCEPTANCE,$ACCEPTED,$ATTEMPTED,$ELAPSED" >> "$RESULTS_CSV"
      
    done
  done
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Sweep completed at $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: $RESULTS_CSV" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Print summary statistics
echo "=== TOP 10 CONFIGURATIONS BY ACCEPTANCE ===" | tee -a "$LOG_FILE"
(head -1 "$RESULTS_CSV" && tail -n +2 "$RESULTS_CSV" | sort -t',' -k6 -rn | head -10) | column -t -s',' | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== TOP 10 CONFIGURATIONS BY MSE (LOW) ===" | tee -a "$LOG_FILE"
(head -1 "$RESULTS_CSV" && tail -n +2 "$RESULTS_CSV" | sort -t',' -k4 -n | head -10) | column -t -s',' | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== BEST BALANCED (ACCEPTANCE >= 70%, LOWEST MSE) ===" | tee -a "$LOG_FILE"
(head -1 "$RESULTS_CSV" && tail -n +2 "$RESULTS_CSV" | awk -F',' '$6 >= 70.0' | sort -t',' -k4 -n | head -10) | column -t -s',' | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Done! Review $RESULTS_CSV for full results." | tee -a "$LOG_FILE"
