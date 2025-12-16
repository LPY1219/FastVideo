#!/bin/bash
# Version using Gloo backend instead of NCCL to avoid SIGSEGV
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export WANDB_API_KEY="0fa0d98600c7e9cae06a14debb71ced7b8dd2a63"
export TRITON_CACHE_DIR=/tmp/triton_cache
DATA_DIR=/DATA/disk1/lpy_a100_4/huggingface/mini_i2v_dataset/crush-smol_preprocessed/combined_parquet_dataset
VALIDATION_DIR=/DATA/disk1/lpy_a100_4/huggingface/mini_i2v_dataset/crush-smol_raw/validation.json

# Use 8 GPUs with sequence parallelism
NUM_GPUS=8
SP_SIZE=4

export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false

# Force Gloo backend instead of NCCL to avoid CUDA version mismatch
export FASTVIDEO_DISTRIBUTED_BACKEND=gloo
export NCCL_DEBUG=WARN

MODEL_PATH=/DATA/disk1/lpy_a100_4/huggingface/Wan2.1-T2V-1.3B-Diffusers

echo "=================================="
echo "Running with Gloo backend (no NCCL):"
echo "  - $NUM_GPUS GPUs"
echo "  - sp_size=$SP_SIZE"
echo "  - gradient_checkpointing=full"
echo "  - Using Gloo instead of NCCL"
echo "=================================="

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
    --sp_size $SP_SIZE \
    --tp_size 1 \
    --num_gpus $NUM_GPUS \
    --hsdp_replicate_dim 2 \
    --hsdp-shard-dim 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 30000 \
    --learning_rate 2e-6 \
    --mixed_precision "bf16" \
    --enable_gradient_checkpointing_type "full" \
    --training_state_checkpointing_steps 400 \
    --validation_steps 10000 \
    --validation_sampling_steps "3" \
    --checkpoints_total_limit 3 \
    --ema_start_step 0 \
    --training_cfg_rate 0.0 \
    --output_dir "outputs_dmd/wan_finetune_gloo" \
    --tracker_project_name Wan_distillation_gloo \
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
