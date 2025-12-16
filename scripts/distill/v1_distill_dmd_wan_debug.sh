#!/bin/bash
# Debug version with single GPU and simplified settings
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export WANDB_API_KEY="0fa0d98600c7e9cae06a14debb71ced7b8dd2a63"
export TRITON_CACHE_DIR=/tmp/triton_cache
DATA_DIR=/DATA/disk1/lpy_a100_4/huggingface/mini_i2v_dataset/crush-smol_preprocessed/combined_parquet_dataset
VALIDATION_DIR=/DATA/disk1/lpy_a100_4/huggingface/mini_i2v_dataset/crush-smol_raw/validation.json

# Try single GPU first
NUM_GPUS=1

# Try without Flash Attention first
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN  # Use FLASH_ATTN like original script
export TOKENIZERS_PARALLELISM=false
# Use Gloo backend instead of NCCL for single GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1  # Disable shared memory for debugging
export CUDA_LAUNCH_BLOCKING=1  # Added for better error messages
# Disable CUDA memory optimizations that might cause issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

MODEL_PATH=/DATA/disk1/lpy_a100_4/huggingface/Wan2.1-T2V-1.3B-Diffusers

echo "=================================="
echo "Running DEBUG version with:"
echo "  - Single GPU (NUM_GPUS=1)"
echo "  - TORCH_SDPA attention backend"
echo "  - CUDA_LAUNCH_BLOCKING=1"
echo "=================================="

# make sure that num_latent_t is a multiple of sp_size
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    fastvideo/training/wan_distillation_pipeline.py \
    --model_path $MODEL_PATH \
    --real_score_model_path $MODEL_PATH \
    --fake_score_model_path $MODEL_PATH \
    --inference_mode False \
    --pretrained_model_name_or_path $MODEL_PATH \
    --cache_dir "/DATA/disk1/lpy_a100_4/.cache" \
    --data_path "$DATA_DIR" \
    --validation_dataset_file  "$VALIDATION_DIR" \
    --train_batch_size 1 \
    --num_latent_t 20 \
    --sp_size 1 \
    --tp_size 1 \
    --num_gpus $NUM_GPUS \
    --hsdp_replicate_dim 1 \
    --hsdp-shard-dim 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 30000 \
    --learning_rate 2e-6 \
    --mixed_precision "bf16" \
    --training_state_checkpointing_steps 400 \
    --validation_steps 10000 \
    --validation_sampling_steps "3" \
    --checkpoints_total_limit 3 \
    --ema_start_step 0 \
    --training_cfg_rate 0.0 \
    --output_dir "outputs_dmd/wan_finetune_debug" \
    --tracker_project_name Wan_distillation_debug \
    --num_height 448 \
    --num_width 832 \
    --num_frames 77 \
    --flow_shift 8 \
    --validation_guidance_scale "6.0" \
    --master_weight_type "fp32" \
    --dit_precision "fp32" \
    --vae_precision "bf16" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --generator_update_interval 5 \
    --dmd_denoising_steps '1000,757,522' \
    --min_timestep_ratio 0.02 \
    --max_timestep_ratio 0.98 \
    --real_score_guidance_scale 3.5 \
    --seed 1024
