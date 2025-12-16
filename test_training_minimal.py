#!/usr/bin/env python3
"""Minimal test script to isolate SIGSEGV in training"""
import os
import sys
import torch
import gc

# Set environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

print("=" * 80)
print("Minimal Training Test - Isolating SIGSEGV")
print("=" * 80)

# Test 1: Basic imports
print("\n[Test 1] Testing imports...")
try:
    from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    print("  [OK] Imports successful")
except Exception as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Check CUDA
print("\n[Test 2] Testing CUDA...")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test 3: Load single transformer using diffusers
print("\n[Test 3] Loading single transformer (using diffusers)...")
try:
    from diffusers import WanTransformer3DModel

    model_path = "/DATA/disk1/lpy_a100_4/huggingface/Wan2.1-T2V-1.3B-Diffusers"

    # Load model using diffusers
    model = WanTransformer3DModel.from_pretrained(
        f"{model_path}/transformer",
        torch_dtype=torch.float32
    )
    model = model.to("cuda")
    model.eval()

    print(f"  [OK] Model loaded, params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"  GPU Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
except Exception as e:
    print(f"  [FAIL] Model load error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Forward pass without gradient
print("\n[Test 4] Testing forward pass (no gradient)...")
try:
    # Create dummy inputs
    # diffusers WanTransformer3DModel expects (batch, channels, time, height, width)
    batch_size = 1
    num_frames = 20  # num_latent_t
    height = 56  # 448 / 8
    width = 104  # 832 / 8
    channels = 16

    hidden_states = torch.randn(batch_size, channels, num_frames, height, width,
                                 device="cuda", dtype=torch.float32)
    encoder_hidden_states = torch.randn(batch_size, 512, 4096,
                                         device="cuda", dtype=torch.float32)
    timestep = torch.tensor([500], device="cuda", dtype=torch.long)

    print(f"  Input shapes: hidden_states={hidden_states.shape}")
    print(f"  GPU Memory before forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            return_dict=False
        )

    print(f"  Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}")
    print(f"  [OK] Forward pass (no grad) successful")
    print(f"  GPU Memory after forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
except Exception as e:
    print(f"  [FAIL] Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward pass WITH gradient (training mode)
print("\n[Test 5] Testing forward pass (with gradient - training mode)...")
try:
    model.train()

    hidden_states = torch.randn(batch_size, channels, num_frames, height, width,
                                 device="cuda", dtype=torch.float32, requires_grad=True)

    print(f"  GPU Memory before forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    output = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        return_dict=False
    )

    output_tensor = output[0] if isinstance(output, tuple) else output
    print(f"  Output shape: {output_tensor.shape}")
    print(f"  GPU Memory after forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Try backward
    print("  Computing backward...")
    loss = output_tensor.mean()
    loss.backward()

    print(f"  [OK] Forward + backward pass successful")
    print(f"  GPU Memory after backward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
except Exception as e:
    print(f"  [FAIL] Training forward/backward error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Clean up
del model, hidden_states, output, output_tensor
gc.collect()
torch.cuda.empty_cache()
print(f"\n  GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Test 6: Load multiple transformers (like distillation pipeline)
print("\n[Test 6] Loading multiple transformers (distillation setup)...")
try:
    # Load 3 transformers like in distillation pipeline:
    # 1. generator (trainable)
    # 2. real_score_transformer (frozen)
    # 3. fake_score_transformer (trainable)

    print("  Loading generator transformer...")
    generator = WanTransformer3DModel.from_pretrained(
        f"{model_path}/transformer",
        torch_dtype=torch.float32
    ).to("cuda")
    generator.train()
    print(f"    GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("  Loading real_score_transformer...")
    real_score = WanTransformer3DModel.from_pretrained(
        f"{model_path}/transformer",
        torch_dtype=torch.float32
    ).to("cuda")
    real_score.eval()
    for p in real_score.parameters():
        p.requires_grad = False
    print(f"    GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("  Loading fake_score_transformer...")
    fake_score = WanTransformer3DModel.from_pretrained(
        f"{model_path}/transformer",
        torch_dtype=torch.float32
    ).to("cuda")
    fake_score.train()
    print(f"    GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print(f"  [OK] All 3 transformers loaded")
    print(f"  Total GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
except Exception as e:
    print(f"  [FAIL] Multiple model load error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Forward pass with multiple models
print("\n[Test 7] Testing forward pass with multiple models...")
try:
    hidden_states = torch.randn(batch_size, channels, num_frames, height, width,
                                 device="cuda", dtype=torch.float32, requires_grad=True)

    print("  Running generator forward...")
    gen_output = generator(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        return_dict=False
    )[0]
    print(f"    Generator output: {gen_output.shape}")

    print("  Running real_score forward...")
    with torch.no_grad():
        real_output = real_score(
            hidden_states=hidden_states.detach(),
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            return_dict=False
        )[0]
    print(f"    Real score output: {real_output.shape}")

    print("  Running fake_score forward...")
    fake_output = fake_score(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        return_dict=False
    )[0]
    print(f"    Fake score output: {fake_output.shape}")

    print("  [OK] All forward passes successful")
except Exception as e:
    print(f"  [FAIL] Multiple model forward error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Backward pass (the actual training step)
print("\n[Test 8] Testing backward pass with loss computation...")
try:
    # Simulate DMD loss computation
    loss = (gen_output - real_output.detach()).pow(2).mean()
    loss += (fake_output - real_output.detach()).pow(2).mean()

    print(f"  Loss value: {loss.item()}")
    print("  Computing backward...")
    loss.backward()

    print(f"  [OK] Backward pass successful")
    print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
except Exception as e:
    print(f"  [FAIL] Backward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nThe SIGSEGV issue is likely in:")
print("  1. Distributed communication (NCCL) - not in model forward/backward")
print("  2. Specific attention metadata handling in set_forward_context")
print("  3. Data loading or batch processing")
print("=" * 80)
