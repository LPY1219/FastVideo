#!/bin/bash
# Debug version with single GPU and simplified settings
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export WANDB_API_KEY="0fa0d98600c7e9cae06a14debb71ced7b8dd2a63"
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONDONTWRITEBYTECODE=1

# Machine detection: automatically select paths based on available directories
if [ -d "/DATA/disk1/lpy_a100_4" ]; then
    echo "Detected: Default machine (A100)"
    MACHINE_NAME="a100"
    DATA_DIR=/DATA/disk1/lpy_a100_4/huggingface/mini_i2v_dataset/crush-smol_preprocessed/combined_parquet_dataset
    VALIDATION_DIR=/DATA/disk1/lpy_a100_4/huggingface/mini_i2v_dataset/crush-smol_raw/validation.json
    MODEL_PATH=/DATA/disk1/lpy_a100_4/huggingface/Wan2.1-T2V-1.3B-Diffusers
    CACHE_DIR="/DATA/disk1/lpy_a100_4/.cache"
else
    echo "Detected: Current machine (BridgeVLA)"
    MACHINE_NAME="bridgevla"
    DATA_DIR=/DATA/disk1/lpy/huggingface/dataset/mini_i2v_dataset/crush-smol_preprocessed/combined_parquet_dataset
    VALIDATION_DIR=/DATA/disk1/lpy/huggingface/dataset/mini_i2v_dataset/crush-smol_raw/validation.json
    MODEL_PATH=/DATA/disk1/lpy/huggingface/Wan2.1-T2V-1.3B-Diffusers
    CACHE_DIR="/DATA/disk1/lpy/.cache"
fi

# Single GPU setup
NUM_GPUS=1

# Attention backend
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
# NCCL settings for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# Enable expandable segments to reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=================================="
echo "Running DEBUG version with:"
echo "  - Machine: $MACHINE_NAME"
echo "  - Single GPU (NUM_GPUS=1)"
echo "  - TORCH_SDPA attention backend"
echo "  - CUDA_LAUNCH_BLOCKING=1"
echo "  - Model: $MODEL_PATH"
echo "  - Data: $DATA_DIR"
echo "  - Cache: $CACHE_DIR"
echo "=================================="

# make sure that num_latent_t is a multiple of sp_size
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    fastvideo/training/wan_distillation_pipeline.py \
    --model-path $MODEL_PATH \
    --real-score-model-path $MODEL_PATH \
    --fake-score-model-path $MODEL_PATH \
    --inference-mode False \
    --pretrained-model-name-or-path $MODEL_PATH \
    --cache-dir "$CACHE_DIR" \
    --data-path "$DATA_DIR" \
    --validation-dataset-file  "$VALIDATION_DIR" \
    --train-batch-size 1 \
    --num-latent-t 12 \
    --sp-size 1 \
    --tp-size 1 \
    --num-gpus $NUM_GPUS \
    --hsdp-replicate-dim $NUM_GPUS \
    --hsdp-shard-dim 1 \
    --train-sp-batch-size 1 \
    --dataloader-num-workers 0 \
    --gradient-accumulation-steps 8 \
    --max-train-steps 30000 \
    --learning-rate 2e-6 \
    --mixed-precision "bf16" \
    --enable-gradient-checkpointing-type "full" \
    --training-state-checkpointing-steps 1000 \
    --weight-only-checkpointing-steps 500 \
    --validation-steps 100 \
    --validation-sampling-steps "3" \
    --log_validation \
    --checkpoints-total-limit 3 \
    --ema-start-step 0 \
    --training-cfg-rate 0.0 \
    --output-dir "outputs_dmd/wan_finetune_debug" \
    --tracker-project-name Wan_distillation_debug \
    --num-height 256 \
    --num-width 256 \
    --num-frames 45 \
    --flow-shift 8 \
    --validation-guidance-scale "6.0" \
    --master-weight-type "fp32" \
    --dit-precision "fp32" \
    --vae-precision "bf16" \
    --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --generator-update-interval 5 \
    --dmd-denoising-steps '1000,757,522' \
    --min-timestep-ratio 0.02 \
    --max-timestep-ratio 0.98 \
    --real-score-guidance-scale 3.5 \
    --seed 1024
