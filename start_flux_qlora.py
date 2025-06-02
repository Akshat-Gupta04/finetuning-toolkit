#!/usr/bin/env python3
"""
Start FLUX QLoRA training with correct arguments
Simple script that runs the exact command needed
"""

import os
import subprocess
from pathlib import Path


def start_training():
    """Start FLUX QLoRA training"""
    
    print("🚀 Starting FLUX QLoRA Training")
    print("=" * 50)
    
    # Check prerequisites
    if not os.path.exists("train.py"):
        print("❌ train.py not found! Run from project root.")
        return
    
    if not os.path.exists(".env"):
        print("❌ .env file not found! Run 'python fix_env.py' first.")
        return
    
    # Check images
    image_dir = Path("data/images")
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    # Count images
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
    images = []
    for ext in image_extensions:
        images.extend(list(image_dir.glob(f"*{ext}")))
        images.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    print(f"📁 Found {len(images)} images in {image_dir}")
    
    if len(images) == 0:
        print("❌ No images found! Add images to data/images/")
        return
    
    # Create output directory
    output_dir = Path("/workspace/finetuning-toolkit/outputs/flux_qlora")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # The correct command based on train.py arguments
    cmd = [
        "python", "train.py",
        "--model", "flux",
        "--input_dir", "data/images",
        "--prepare_dataset",
        "--config", "config/flux_qlora_config.yaml",
        "--data_dir", "/workspace/finetuning-toolkit/data/train",
        "--output_dir", "/workspace/finetuning-toolkit/outputs/flux_qlora",
        "--batch_size", "1",
        "--mixed_precision", "bf16"
    ]
    
    print(f"\n🎯 Training Configuration:")
    print(f"  Model: FLUX with QLoRA")
    print(f"  Images: {len(images)} files")
    print(f"  Config: config/flux_qlora_config.yaml")
    print(f"  Output: /workspace/finetuning-toolkit/outputs/flux_qlora")
    print(f"  Batch Size: 1")
    print(f"  Mixed Precision: bf16")
    
    print(f"\n🚀 Starting training...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Start training
    try:
        # Run the command
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print(f"\n✅ Training completed successfully!")
        else:
            print(f"\n❌ Training failed with return code: {result.returncode}")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")


if __name__ == "__main__":
    start_training()
