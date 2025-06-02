#!/usr/bin/env python3
"""
Super quick .env file creator
Usage: python quick_env.py hf_your_token_here
"""

import sys
import os
from pathlib import Path


def quick_env(hf_token, wandb_token=None):
    """Create .env file quickly with minimal input"""
    
    # Validate HF token
    if not hf_token.startswith('hf_'):
        print("❌ Invalid Hugging Face token! Must start with 'hf_'")
        return False
    
    # Create .env content
    env_content = f"""# Quick .env for Diffusion Finetuning Toolkit

# Authentication
HF_TOKEN={hf_token}
HUGGINGFACE_HUB_TOKEN={hf_token}
"""
    
    if wandb_token:
        env_content += f"WANDB_API_KEY={wandb_token}\n"
    
    env_content += """
# Storage (RunPod optimized)
HF_HOME=/workspace/cache/huggingface
TRANSFORMERS_CACHE=/workspace/cache/transformers
DIFFUSERS_CACHE=/workspace/cache/diffusers
DATA_DIR=/workspace/data
OUTPUT_DIR=/workspace/outputs
LOGS_DIR=/workspace/logs

# CUDA (A40 optimized)
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
TORCH_DTYPE=bfloat16
MIXED_PRECISION=bf16
TOKENIZERS_PARALLELISM=false

# Training defaults
DEFAULT_BATCH_SIZE=1
DEFAULT_LORA_RANK=64
DEFAULT_LEARNING_RATE=1e-4
DEFAULT_MAX_STEPS=1000
GRADIENT_CHECKPOINTING=true
USE_MEMORY_EFFICIENT_ATTENTION=true
OMP_NUM_THREADS=8
PYTHONUNBUFFERED=1
"""
    
    # Write file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    # Create directories
    dirs = [
        "/workspace/cache/huggingface",
        "/workspace/cache/transformers",
        "/workspace/cache/diffusers", 
        "/workspace/data",
        "/workspace/outputs",
        "/workspace/logs"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print("✅ Quick .env created!")
    print(f"📁 Location: {os.path.abspath('.env')}")
    print("🚀 Ready to train!")
    return True


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python quick_env.py hf_your_token_here")
        print("  python quick_env.py hf_your_token_here wandb_token_here")
        print("\nOr interactive mode:")
        hf_token = input("Hugging Face token (hf_...): ").strip()
        wandb_token = input("W&B token (optional): ").strip() or None
    else:
        hf_token = sys.argv[1]
        wandb_token = sys.argv[2] if len(sys.argv) > 2 else None
    
    quick_env(hf_token, wandb_token)


if __name__ == "__main__":
    main()
