#!/bin/bash
#SBATCH --job-name=t2v
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=dmd_Wan2.2/t2v_g2e5_f1e5_%j.out
#SBATCH --error=dmd_Wan2.2/t2v_g2e5_f1e5_%j.err
#SBATCH --exclusive
set -e -x

# Environment Setup
source /home/yw/anaconda3/bin/activate
conda activate Bridgevla_fast

# Basic Info
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export MASTER_PORT=29500
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE="offline"     #online
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export WANDB_API_KEY="0fa0d98600c7e9cae06a14debb71ced7b8dd2a63"
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# Configs
NUM_GPUS=8  # Number of GPUs per node
NNODES=${NNODES:-1}  # Number of nodes (default: 1)
TOTAL_GPUS=$((NNODES * NUM_GPUS))  # Auto-calculate total GPUs

MODEL_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B-Diffusers"
REAL_SCORE_MODEL_PATH=$MODEL_PATH
FAKE_SCORE_MODEL_PATH=$MODEL_PATH
DATA_DIR="/DATA/disk1/lpy/huggingface/dataset/Wan2.2-Syn-121x704x1280_32k/train"
VALIDATION_DIR="/DATA/disk1/lpy/huggingface/dataset/mini_i2v_dataset/crush-smol_raw/validation.json"  #(example:validation_64.json)
OUTPUT_DIR="checkpoints/wan_t2v_finetune"
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

echo "Configuration: ${NNODES} nodes × ${NUM_GPUS} GPUs/node = ${TOTAL_GPUS} total GPUs"

# Training arguments
training_args=(
  --tracker_project_name Wan_distillation
  --output_dir "$OUTPUT_DIR"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 12
  --num_height 256
  --num_width 256
  --num_frames 45
  --enable_gradient_checkpointing_type "ops"
)

# Parallel arguments - use FSDP sharding to reduce memory
parallel_args=(
  --num_gpus $TOTAL_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $TOTAL_GPUS
)

# Model arguments
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DIR"
  --validation_steps 200
  --validation_sampling_steps "3"
  --validation_guidance_scale "6.0" # not used for dmd inference
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 4e-6
  --lr_scheduler "cosine_with_min_lr"
  --min_lr_ratio 0.5
  --lr_warmup_steps 100
  --fake_score_learning_rate 2e-6
  --fake_score_lr_scheduler "cosine_with_min_lr"
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 200
  --weight_decay 0.01
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --ema_start_step 0
  --flow_shift 5
  --seed 1000
)

# DMD arguments
dmd_args=(
  --dmd_denoising_steps '1000,757,522'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --generator_update_interval 5
  --real_score_guidance_scale 3
  --simulate_generator_forward 
  --log_visualization # disable if oom
)

torchrun \
--nnodes $NNODES \
--nproc_per_node $NUM_GPUS \
--node_rank $NODE_RANK \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/wan_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}"
