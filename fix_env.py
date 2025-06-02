#!/usr/bin/env python3
"""
Quick fix for .env file to resolve the LOGGING_DIR error
"""

import os


def fix_env_file():
    """Fix .env file with your HF token"""
    
    print("🔧 Quick .env fix for the LOGGING_DIR error")
    print("=" * 50)
    
    # Get your HF token
    hf_token = input("Enter your Hugging Face token (hf_...): ").strip()
    
    if not hf_token.startswith('hf_'):
        print("❌ Invalid token! Must start with 'hf_'")
        return
    
    # Create complete .env file
    env_content = f"""# Fixed .env file for Diffusion Finetuning Toolkit

# Authentication
HF_TOKEN={hf_token}
HUGGINGFACE_HUB_TOKEN={hf_token}

# Required paths for old system compatibility
OUTPUT_DIR=/workspace/outputs
LOGGING_DIR=/workspace/logs
CACHE_DIR=/workspace/cache
DATA_DIR=/workspace/data

# Cache directories for new toolkit
HF_HOME=/workspace/cache/huggingface
TRANSFORMERS_CACHE=/workspace/cache/transformers
DIFFUSERS_CACHE=/workspace/cache/diffusers
HF_DATASETS_CACHE=/workspace/cache/datasets

# CUDA optimization for A40
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
TORCH_DTYPE=bfloat16
MIXED_PRECISION=bf16
TOKENIZERS_PARALLELISM=false

# Model paths
FLUX_MODEL_PATH=black-forest-labs/FLUX.1-dev
WAN21_I2V_MODEL_PATH=Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
WAN21_T2V_MODEL_PATH=Wan-AI/Wan2.1-T2V-14B-720P-Diffusers
WAN2_1_MODEL_PATH=Wan-AI/Wan2.1-I2V-14B-720P-Diffusers

# Training defaults
DEFAULT_BATCH_SIZE=1
DEFAULT_LORA_RANK=64
DEFAULT_LEARNING_RATE=1e-4
DEFAULT_MAX_STEPS=1000
GRADIENT_CHECKPOINTING=true

# Performance
OMP_NUM_THREADS=8
USE_MEMORY_EFFICIENT_ATTENTION=true
PYTHONUNBUFFERED=1

# Logging
LOG_LEVEL=INFO
LOGGING_STEPS=10

# Safety
TRUST_REMOTE_CODE=false
USE_AUTH_TOKEN=true
"""
    
    # Write .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    # Create directories
    from pathlib import Path
    dirs = [
        "/workspace/cache/huggingface",
        "/workspace/cache/transformers",
        "/workspace/cache/diffusers",
        "/workspace/cache",
        "/workspace/data",
        "/workspace/outputs", 
        "/workspace/logs"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Fixed .env file created!")
    print(f"✅ Directories created!")
    print(f"✅ HF Token: {hf_token[:10]}...")
    
    print(f"\n🎉 The LOGGING_DIR error should now be fixed!")
    print(f"\nTry running your app again:")
    print(f"  python app.py")


if __name__ == "__main__":
    fix_env_file()
