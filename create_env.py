#!/usr/bin/env python3
"""
Create .env file from template with user inputs
Simple script to generate .env file for diffusion finetuning toolkit
"""

import os
import shutil
from pathlib import Path


def create_env_file():
    """Create .env file with user inputs"""
    
    print("🚀 Creating .env file for Diffusion Finetuning Toolkit")
    print("=" * 60)
    
    # Check if template exists
    if not os.path.exists('.env.template'):
        print("❌ .env.template not found!")
        print("Please make sure .env.template exists in the current directory.")
        return
    
    # Get required inputs
    print("\n🔑 REQUIRED SETTINGS:")
    
    # Hugging Face Token (required)
    hf_token = input("Enter your Hugging Face token (hf_...): ").strip()
    while not hf_token or not hf_token.startswith('hf_'):
        print("❌ Please enter a valid Hugging Face token starting with 'hf_'")
        hf_token = input("Enter your Hugging Face token (hf_...): ").strip()
    
    print("\n📁 OPTIONAL SETTINGS (press Enter for defaults):")
    
    # Optional inputs with defaults
    wandb_key = input("Weights & Biases API Key (optional): ").strip()
    workspace_path = input("Workspace path [/workspace]: ").strip() or "/workspace"
    
    # Training defaults
    print("\n🎯 TRAINING DEFAULTS:")
    batch_size = input("Batch size [1]: ").strip() or "1"
    lora_rank = input("LoRA rank [64]: ").strip() or "64"
    learning_rate = input("Learning rate [1e-4]: ").strip() or "1e-4"
    max_steps = input("Max training steps [1000]: ").strip() or "1000"
    
    # Read template
    with open('.env.template', 'r') as f:
        env_content = f.read()
    
    # Replace placeholders
    replacements = {
        'hf_your_token_here': hf_token,
        'your_wandb_key_here': wandb_key if wandb_key else '',
        '/workspace': workspace_path,
        'DEFAULT_BATCH_SIZE=1': f'DEFAULT_BATCH_SIZE={batch_size}',
        'DEFAULT_LORA_RANK=64': f'DEFAULT_LORA_RANK={lora_rank}',
        'DEFAULT_LEARNING_RATE=1e-4': f'DEFAULT_LEARNING_RATE={learning_rate}',
        'DEFAULT_MAX_STEPS=1000': f'DEFAULT_MAX_STEPS={max_steps}',
    }
    
    # Apply replacements
    for old, new in replacements.items():
        env_content = env_content.replace(old, new)
    
    # Write .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ .env file created successfully!")
    print(f"📁 Location: {os.path.abspath('.env')}")
    
    # Create directories
    create_directories(workspace_path)
    
    print(f"\n🎉 Setup complete!")
    print(f"\nNext steps:")
    print(f"1. Review .env file: cat .env")
    print(f"2. Install dependencies: pip install -r requirements_toolkit.txt")
    print(f"3. Test setup: python -c 'from toolkit.core import *; print(\"✅ Ready!\")'")


def create_directories(base_path="/workspace"):
    """Create necessary directories"""
    directories = [
        f"{base_path}/cache/huggingface",
        f"{base_path}/cache/transformers",
        f"{base_path}/cache/diffusers",
        f"{base_path}/cache/datasets",
        f"{base_path}/cache/models",
        f"{base_path}/data",
        f"{base_path}/outputs",
        f"{base_path}/logs",
        f"{base_path}/tmp",
        f"{base_path}/backups"
    ]
    
    print(f"\n📁 Creating directories in {base_path}...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")


def quick_env():
    """Create minimal .env file quickly"""
    print("🚀 Quick .env setup")
    
    hf_token = input("Hugging Face token (hf_...): ").strip()
    if not hf_token.startswith('hf_'):
        print("❌ Invalid token format")
        return
    
    env_content = f"""# Minimal .env for Diffusion Finetuning Toolkit

# Authentication
HF_TOKEN={hf_token}
HUGGINGFACE_HUB_TOKEN={hf_token}

# Storage (RunPod optimized)
HF_HOME=/workspace/cache/huggingface
TRANSFORMERS_CACHE=/workspace/cache/transformers
DIFFUSERS_CACHE=/workspace/cache/diffusers
DATA_DIR=/workspace/data
OUTPUT_DIR=/workspace/outputs
LOGS_DIR=/workspace/logs

# CUDA optimization for A40
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

# Training defaults
TORCH_DTYPE=bfloat16
MIXED_PRECISION=bf16
TOKENIZERS_PARALLELISM=false
GRADIENT_CHECKPOINTING=true
DEFAULT_BATCH_SIZE=1
DEFAULT_LORA_RANK=64
DEFAULT_LEARNING_RATE=1e-4
DEFAULT_MAX_STEPS=1000

# Performance
USE_MEMORY_EFFICIENT_ATTENTION=true
OMP_NUM_THREADS=8
PYTHONUNBUFFERED=1
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    create_directories()
    print("✅ Quick .env created!")


def validate_env():
    """Validate existing .env file"""
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        return False
    
    required_vars = ['HF_TOKEN', 'DATA_DIR', 'OUTPUT_DIR']
    
    with open('.env', 'r') as f:
        content = f.read()
    
    missing = []
    for var in required_vars:
        if f"{var}=" not in content or f"{var}=" in content and content.split(f"{var}=")[1].split('\n')[0].strip() == "":
            missing.append(var)
    
    if missing:
        print(f"❌ Missing or empty variables: {', '.join(missing)}")
        return False
    
    print("✅ .env file is valid!")
    return True


def main():
    """Main function with menu"""
    print("🔧 .env File Creator for Diffusion Finetuning Toolkit")
    print("=" * 60)
    print("1. Create .env from template (recommended)")
    print("2. Create minimal .env (quick)")
    print("3. Validate existing .env")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        create_env_file()
    elif choice == "2":
        quick_env()
    elif choice == "3":
        validate_env()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
