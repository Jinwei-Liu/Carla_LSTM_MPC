#!/bin/bash

# Simple bash script for parallel training with screen/tmux
# This script creates separate screen sessions for each training job

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
NUM_TARGETS=(1 2 5 10)
LQR_ITERS=(5 10 30 50)

# Number of GPUs available (modify based on your system)
NUM_GPUS=1

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$BASE_SAVE_DIR"

# Function to run a single training
run_training() {
    local num_targets=$1
    local lqr_iter=$2
    local gpu_id=$3
    
    local session_name="train_t${num_targets}_l${lqr_iter}"
    local save_dir="${BASE_SAVE_DIR}/targets_${num_targets}_lqr_${lqr_iter}"
    local log_file="${LOG_DIR}/train_targets_${num_targets}_lqr_${lqr_iter}.log"
    
    echo "Starting training: targets=$num_targets, lqr_iter=$lqr_iter on GPU $gpu_id"
    
    # Create screen session and run training
    screen -dmS "$session_name" bash -c "
        export CUDA_VISIBLE_DEVICES=$gpu_id
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
        echo 'Training completed: targets=$num_targets, lqr_iter=$lqr_iter' >> $log_file
    "
}

# Main execution
echo "=========================================="
echo "Starting Parallel Training"
echo "=========================================="
echo "Total configurations: ${#NUM_TARGETS[@]} x ${#LQR_ITERS[@]} = $((${#NUM_TARGETS[@]} * ${#LQR_ITERS[@]}))"
echo "Available GPUs: $NUM_GPUS"
echo ""

# Initialize job counter and GPU assignment
job_id=0
gpu_id=0

# Loop through all combinations
for num_targets in "${NUM_TARGETS[@]}"; do
    for lqr_iter in "${LQR_ITERS[@]}"; do
        # Run training
        run_training $num_targets $lqr_iter $gpu_id
        
        # Update GPU assignment (round-robin)
        gpu_id=$(( (gpu_id + 1) % NUM_GPUS ))
        job_id=$((job_id + 1))
        
        # Small delay to avoid overwhelming the system
        sleep 2
    done
done

echo ""
echo "All training jobs started!"
echo "=========================================="
echo "Monitor progress with:"
echo "  screen -ls                # List all sessions"
echo "  screen -r train_tX_lY      # Attach to specific session"
echo "  tail -f logs/*.log         # Watch log files"
echo ""
echo "To kill all training sessions:"
echo "  screen -ls | grep train_ | cut -d. -f1 | xargs -I {} screen -X -S {} quit"
echo "=========================================="

# Optional: Wait for all jobs to complete
read -p "Press Enter to start monitoring, or Ctrl+C to exit..."

# Monitor script
while true; do
    clear
    echo "=========================================="
    echo "Training Status Monitor"
    echo "=========================================="
    echo ""
    
    # Check running sessions
    echo "Running sessions:"
    screen -ls | grep train_ || echo "  None"
    echo ""
    
    # Check completed trainings
    echo "Completed trainings (models found):"
    for num_targets in "${NUM_TARGETS[@]}"; do
        for lqr_iter in "${LQR_ITERS[@]}"; do
            model_path="${BASE_SAVE_DIR}/targets_${num_targets}_lqr_${lqr_iter}/best_vehicle_lstm_mpc_ds${DOWNSAMPLE_FACTOR}_targets${num_targets}_lqr${lqr_iter}.pth"
            if [ -f "$model_path" ]; then
                echo "  ✓ targets=$num_targets, lqr_iter=$lqr_iter"
            fi
        done
    done
    echo ""
    
    # Check for errors in logs
    echo "Recent errors (if any):"
    grep -i "error\|exception\|failed" logs/*.log 2>/dev/null | tail -5 || echo "  None"
    echo ""
    
    echo "Press Ctrl+C to exit monitor"
    sleep 30
done