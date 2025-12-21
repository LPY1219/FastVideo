num_gpus=1
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export MODEL_BASE=/DATA/disk1/lpy/huggingface/FastWan2.2-TI2V-5B-FullAttn-Diffusers
# export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
# You can either use --prompt or --prompt-txt, but not both.
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --height 256 \
    --width 256 \
    --num-frames 49 \
    --num-inference-steps 3 \
    --fps 24 \
    --prompt-txt assets/prompt.txt \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 1024 \
    --output-path outputs_video_dmd/ \
    --dmd-denoising-steps "1000,757,522"
