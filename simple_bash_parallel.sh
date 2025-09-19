#!/bin/bash

# Simple bash script for serial training
# This script runs training jobs one after another

# Configuration
DATA_FOLDER="vehicle_datasets"
BASE_SAVE_DIR="models"
LOG_DIR="logs"
DOWNSAMPLE_FACTOR=10
EPOCHS=100
BATCH_SIZE=128
LR=0.001
HIDDEN_DIM=64

# Arrays of parameters to test
NUM_TARGETS=(1 2 5 8 10)
LQR_ITERS=(5 10 30 50)

# GPU to use (modify based on your system)
GPU_ID=0

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$BASE_SAVE_DIR"

# Function to run a single training
run_training() {
    local num_targets=$1
    local lqr_iter=$2
    local job_num=$3
    local total_jobs=$4
    
    local save_dir="${BASE_SAVE_DIR}/targets_${num_targets}_lqr_${lqr_iter}"
    local log_file="${LOG_DIR}/train_targets_${num_targets}_lqr_${lqr_iter}.log"
    
    echo ""
    echo "=========================================="
    echo "Starting training job $job_num/$total_jobs"
    echo "Parameters: targets=$num_targets, lqr_iter=$lqr_iter"
    echo "GPU: $GPU_ID"
    echo "Log file: $log_file"
    echo "=========================================="
    
    # Set GPU and run training
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Run training and capture exit status
    python vehicle_lstm_mpc.py \
        --mode train \
        --data_folder $DATA_FOLDER \
        --save_dir $save_dir \
        --num_targets $num_targets \
        --lqr_iter $lqr_iter \
        --downsample_factor $DOWNSAMPLE_FACTOR \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --hidden_dim $HIDDEN_DIM \
        2>&1 | tee $log_file
    
    # Check if training completed successfully
    local exit_status=${PIPESTATUS[0]}
    if [ $exit_status -eq 0 ]; then
        echo "✓ Training completed successfully: targets=$num_targets, lqr_iter=$lqr_iter" | tee -a $log_file
    else
        echo "✗ Training failed with exit code $exit_status: targets=$num_targets, lqr_iter=$lqr_iter" | tee -a $log_file
    fi
    
    echo "Finished job $job_num/$total_jobs"
    echo "=========================================="
}

# Main execution
echo "=========================================="
echo "Starting Serial Training"
echo "=========================================="

# Calculate total number of jobs
total_jobs=$((${#NUM_TARGETS[@]} * ${#LQR_ITERS[@]}))
echo "Total configurations: ${#NUM_TARGETS[@]} x ${#LQR_ITERS[@]} = $total_jobs"
echo "Using GPU: $GPU_ID"
echo ""

# Initialize job counter
job_num=0

# Record start time
start_time=$(date)
echo "Started at: $start_time"

# Loop through all combinations sequentially
for lqr_iter in "${LQR_ITERS[@]}"; do
    for num_targets in "${NUM_TARGETS[@]}"; do
        job_num=$((job_num + 1))
        
        # Run training and wait for completion
        run_training $num_targets $lqr_iter $job_num $total_jobs
        
        # Small delay between jobs
        sleep 1
    done
done

# Record end time
end_time=$(date)

echo ""
echo "=========================================="
echo "All training jobs completed!"
echo "=========================================="
echo "Started at: $start_time"
echo "Finished at: $end_time"
echo ""

# Summary of results
echo "Training Results Summary:"
echo "----------------------------------------"
for num_targets in "${NUM_TARGETS[@]}"; do
    for lqr_iter in "${LQR_ITERS[@]}"; do
        model_path="${BASE_SAVE_DIR}/targets_${num_targets}_lqr_${lqr_iter}/best_vehicle_lstm_mpc_ds${DOWNSAMPLE_FACTOR}_targets${num_targets}_lqr${lqr_iter}.pth"
        log_file="${LOG_DIR}/train_targets_${num_targets}_lqr_${lqr_iter}.log"
        
        if [ -f "$model_path" ]; then
            echo "  ✓ targets=$num_targets, lqr_iter=$lqr_iter - Model saved"
        elif [ -f "$log_file" ]; then
            if grep -q "failed\|error\|Error" "$log_file"; then
                echo "  ✗ targets=$num_targets, lqr_iter=$lqr_iter - Training failed"
            else
                echo "  ? targets=$num_targets, lqr_iter=$lqr_iter - Status unknown"
            fi
        else
            echo "  - targets=$num_targets, lqr_iter=$lqr_iter - Not started"
        fi
    done
done

echo ""
echo "Check individual log files in '$LOG_DIR/' for detailed information"
echo "Models saved in '$BASE_SAVE_DIR/' subdirectories"
echo "========================================"