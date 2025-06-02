#!/usr/bin/env python3
"""
Update .env file with your actual tokens
Fixes the compatibility issue between old and new systems
"""

import os
import re


def update_env_tokens():
    """Update .env file with user tokens"""
    
    print("🔑 Update .env file with your tokens")
    print("=" * 50)
    
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Run: python quick_env.py first")
        return
    
    # Read current .env file
    with open('.env', 'r') as f:
        env_content = f.read()
    
    print("Current .env file found. Let's update your tokens:")
    
    # Get Hugging Face token
    print("\n1. Hugging Face Token (REQUIRED)")
    print("   Get from: https://huggingface.co/settings/tokens")
    
    current_hf = re.search(r'HF_TOKEN=(.+)', env_content)
    current_hf_token = current_hf.group(1) if current_hf else "your_hf_token_here"
    
    if current_hf_token == "your_hf_token_here":
        hf_token = input("   Enter HF token (hf_...): ").strip()
        
        if not hf_token or not hf_token.startswith('hf_'):
            print("❌ Invalid token! Must start with 'hf_'")
            return
        
        # Update HF tokens
        env_content = re.sub(r'HF_TOKEN=.+', f'HF_TOKEN={hf_token}', env_content)
        env_content = re.sub(r'HUGGINGFACE_HUB_TOKEN=.+', f'HUGGINGFACE_HUB_TOKEN={hf_token}', env_content)
        print(f"   ✅ Updated HF token: {hf_token[:10]}...")
    else:
        print(f"   ✅ HF token already set: {current_hf_token[:10]}...")
    
    # Get W&B token (optional)
    print("\n2. Weights & Biases Token (OPTIONAL)")
    print("   Get from: https://wandb.ai/authorize")
    print("   Press Enter to skip")
    
    current_wandb = re.search(r'WANDB_API_KEY=(.+)', env_content)
    current_wandb_token = current_wandb.group(1) if current_wandb else "your_wandb_key_here"
    
    if current_wandb_token == "your_wandb_key_here":
        wandb_token = input("   Enter W&B token (optional): ").strip()
        
        if wandb_token:
            env_content = re.sub(r'WANDB_API_KEY=.+', f'WANDB_API_KEY={wandb_token}', env_content)
            print(f"   ✅ Updated W&B token: {wandb_token[:10]}...")
        else:
            print("   ⏭️  Skipped W&B token")
    else:
        print(f"   ✅ W&B token already set: {current_wandb_token[:10]}...")
    
    # Write updated .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ .env file updated successfully!")
    
    # Create directories
    create_directories()
    
    print(f"\n🎉 Setup complete!")
    print(f"\nYour .env file now contains:")
    print(f"  ✅ All required variables for old system compatibility")
    print(f"  ✅ A40 GPU optimizations")
    print(f"  ✅ RunPod workspace paths")
    print(f"  ✅ Your authentication tokens")
    
    print(f"\nNext steps:")
    print(f"1. Test old system: python app.py")
    print(f"2. Test new toolkit: python toolkit_cli.py create-config flux --output test.yaml")
    print(f"3. Start training: python train.py --model flux --input_dir /workspace/data")


def create_directories():
    """Create all necessary directories"""
    from pathlib import Path
    
    directories = [
        "/workspace/cache/huggingface",
        "/workspace/cache/transformers",
        "/workspace/cache/diffusers",
        "/workspace/cache/datasets",
        "/workspace/cache/models",
        "/workspace/cache",
        "/workspace/data",
        "/workspace/outputs",
        "/workspace/logs",
        "/workspace/tmp",
        "/workspace/backups",
        "/workspace/models"
    ]
    
    print(f"\n📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")


def validate_env():
    """Validate .env file"""
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        return False
    
    with open('.env', 'r') as f:
        content = f.read()
    
    # Check required variables for old system
    required_old = [
        'HF_TOKEN', 'OUTPUT_DIR', 'LOGGING_DIR', 'CACHE_DIR', 'DATA_DIR'
    ]
    
    # Check required variables for new toolkit
    required_new = [
        'HF_TOKEN', 'HF_HOME', 'TRANSFORMERS_CACHE', 'DIFFUSERS_CACHE'
    ]
    
    missing_old = []
    missing_new = []
    
    for var in required_old:
        if f"{var}=" not in content:
            missing_old.append(var)
    
    for var in required_new:
        if f"{var}=" not in content:
            missing_new.append(var)
    
    if missing_old:
        print(f"❌ Missing variables for old system: {', '.join(missing_old)}")
    else:
        print("✅ Old system compatibility: OK")
    
    if missing_new:
        print(f"❌ Missing variables for new toolkit: {', '.join(missing_new)}")
    else:
        print("✅ New toolkit compatibility: OK")
    
    # Check if tokens are set
    hf_token_set = 'HF_TOKEN=your_hf_token_here' not in content
    wandb_token_set = 'WANDB_API_KEY=your_wandb_key_here' not in content
    
    print(f"🔑 HF Token: {'✅ Set' if hf_token_set else '❌ Not set'}")
    print(f"🔑 W&B Token: {'✅ Set' if wandb_token_set else '⏭️ Optional'}")
    
    return len(missing_old) == 0 and len(missing_new) == 0


def main():
    """Main function"""
    print("🔧 .env Token Updater")
    print("Fixes compatibility between old and new systems")
    print("=" * 60)
    print("1. Update tokens in .env file")
    print("2. Validate .env file")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        update_env_tokens()
    elif choice == "2":
        validate_env()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
