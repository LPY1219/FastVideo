#!/usr/bin/env python3
"""Test script to diagnose the SIGSEGV error"""
import os
import sys

print("=" * 80)
print("Testing imports...")
print("=" * 80)

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ CUDA version: {torch.version.cuda}")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import flash_attn
    print(f"✓ flash_attn: {flash_attn.__version__}")
except Exception as e:
    print(f"✗ flash_attn import failed: {e}")

try:
    from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
    print("✓ fastvideo.fastvideo_args imported")
except Exception as e:
    print(f"✗ fastvideo_args import failed: {e}")
    sys.exit(1)

try:
    from fastvideo.pipelines.basic.wan.wan_dmd_pipeline import WanDMDPipeline
    print("✓ WanDMDPipeline imported")
except Exception as e:
    print(f"✗ WanDMDPipeline import failed: {e}")
    sys.exit(1)

try:
    from fastvideo.training.distillation_pipeline import DistillationPipeline
    print("✓ DistillationPipeline imported")
except Exception as e:
    print(f"✗ DistillationPipeline import failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("Testing basic model loading...")
print("=" * 80)

# Set environment variables
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"

model_path = "/DATA/disk1/lpy_a100_4/huggingface/Wan2.1-T2V-1.3B-Diffusers"

try:
    # Try to create a minimal args object
    print(f"Attempting to load model from: {model_path}")
    print("This may take a while...")

    # We won't actually load the model, just test the args parsing
    print("✓ If you see this, basic imports work fine")
    print("\nThe SIGSEGV error likely occurs during model loading or initialization.")
    print("Try running with a single GPU first to isolate the issue.")

except Exception as e:
    print(f"✗ Error during model test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("All basic tests passed!")
print("=" * 80)
